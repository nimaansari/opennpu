# Copyright (C) 2026, Advanced Micro Devices, Inc. (lm_gemv.py, MIT) and the OpenNPU authors.
# SPDX-License-Identifier: MIT
"""int4 GEMV device for any matrix: y[N] = W[N, K] x[K], Q4NX weights (w = q * scale + min).

From MLIR-AIR's fused_prefill/lm_gemv.py (one fixed LM-head shape) made shape-generic:

  * the cores' loop counts (row tiles per core, K packets) are RTPs and the input length is a
    kernel argument, so every shape runs on the same core program (one PDI, one hw_context);
    a shape only changes the insts, built once per (tiles, kc)
  * each column has its own input vector (X is [8, KMAX]): one dispatch can run 8 slices of one
    matrix on the same x, or the down projections of several MoE experts on their own inputs
  * each column streams its own run of W from a byte offset that is linear in the insts
    (insts.LinearInsts), so a dispatch can read 8 different places: 8 slices of one matrix, or
    the column blocks of several MoE experts, chosen per token without copying weights

32 cores (8 columns x 4 rows), each owning `tiles` (even) 64-row tiles; a column's stream interleaves
its four cores' packets [tile][k packet][row], the memtile splits them. K <= KMAX per dispatch.

This module only needs the MLIR-AIR python API (build toolchain); pack()/unpack() are numpy.
"""
import numpy as np

NC, NR = 8, 4
GN, GK = 64, 256
KMAX = 2048
FLUSH = 2                 # tiles per output transfer; a 1-tile (256 B per core) flush hangs the device
PKT = GN * GK // 2 + 2 * 2 * (GK // 32) * GN          # bytes per packet: nibbles + scale + min
PKT_S = GN * GK // 2 + 2 * (GK // 32) * GN            # q4s (symmetric, sym=True): nibbles + scale
OBJ = "gemv_any.o"
OBJ8 = "gemm8.o"
BATCH = 8                 # tokens per weight pass in the batched (prompt) device
W_BYTES = 1 << 30                                       # the W argument's declared size (offsets < 1 GiB)


def out_tiles(tiles):
    """Tiles per core in the output (an odd last tile comes out with a zero partner)."""
    return tiles + tiles % FLUSH


def pkt(sym=False):
    return PKT_S if sym else PKT


def col_bytes(tiles, kc, sym=False):
    return tiles * kc * NR * pkt(sym)


def build(tiles, kc, offs=(0,) * NC, batch=1, sym=False):
    """batch 1: the GEMV (one token, gemv_any.cc). batch 8: 8 tokens per weight pass (gemm8.cc), on
    the same packed weights: X per column is [K/8][8 tok][8] bf16, Y per column [..][nb][tok][nn].
    sym: the q4s weight format (kernels built with -DGV_SYM; smaller packets)."""
    from air import api as air
    from air.api.types import bf16, f32, i8, i32

    assert batch in (1, BATCH)
    assert 1 <= kc <= KMAX // GK and tiles >= 1 and (tiles * kc) % 2 == 0, (tiles, kc)   # ping-pong in step
    T = batch
    obj = OBJ if T == 1 else OBJ8
    PKT = pkt(sym)
    cb = col_bytes(tiles, kc, sym)
    W = air.tensor([W_BYTES], i8)
    X = air.tensor([NC, KMAX * T], bf16)
    to = out_tiles(tiles)
    Y = air.tensor([NC, to * NR * GN * T], bf16)

    if T == 1:
        expand = air.extern("gv_expand", link_with=obj, scalars=[i32])
        zero = air.extern("gv_zero2", link_with=obj)
        acc_k = air.extern("gv_acc2", link_with=obj, scalars=[i32, i32])
        flush = air.extern("gv_flush2", link_with=obj)
    else:
        prep = air.extern("gm_prep", link_with=obj, scalars=[i32])
        zero = air.extern("gm_zero2", link_with=obj)
        acc_k = air.extern("gm_acc2", link_with=obj, scalars=[i32, i32])
        flush = air.extern("gm_flush2", link_with=obj)

    w_in = air.channel("WIn", size=[NC])
    x_in = air.channel("XIn", size=[NC])
    w2l1 = air.channel("W2L1", size=[NC, NR])
    x2l1 = air.channel("X2L1", size=[NC, 1], broadcast_shape=[NC, NR])
    o2l2 = air.channel("O2L2", size=[NC, NR])
    o_out = air.channel("OOut", size=[NC])

    with air.launch(repeat=1, name="gemv_any" if T == 1 else "gemm8_any", attrs=["air.preserve_shim_dma_order"]) as lch:

        @lch.body
        def _(wave):
            for c in range(NC):
                x_in.put(X[c:c + 1, 0:KMAX * T], indices=[c])
            for c in range(NC):
                w_in.put(W[offs[c]:offs[c] + cb], indices=[c])
            for c in range(NC):
                o_out.get(Y[c:c + 1, 0:to * NR * GN * T], indices=[c])

            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    go = air.rtp(wave * 0 + 1)
                    nt = air.rtp(wave * 0 + tiles // FLUSH)
                    nk = air.rtp(wave * 0 + kc)              # packet pairs per FLUSH pair of tiles
                    kcr = air.rtp(wave * 0 + kc)
                    nl = air.rtp(wave * 0 + tiles % FLUSH)    # an odd last tile: kc / 2 packet pairs
                    nkh = air.rtp(wave * 0 + kc // 2)
                    for c in range(NC):
                        x_l2 = air.alloc([KMAX * T], bf16, scope=seg.private(), column=c)
                        x_in.get(x_l2, indices=[c])
                        x2l1.put(x_l2, indices=[c, 0])
                        w_l2 = [air.alloc([NR * PKT], i8, scope=seg.private(), column=c, split=False)
                                for _ in range(2)]
                        for pp in range(2):
                            w_in.get(w_l2[pp], indices=[c])
                        for pp in range(2):
                            for r in range(NR):
                                w2l1.put(w_l2[pp][r * PKT:r * PKT + PKT], indices=[c, r])
                        o_l2 = air.alloc([NR, FLUSH * GN * T], bf16, scope=seg.private(), column=c, split=False)
                        for r in range(NR):
                            o2l2.get(o_l2[r:r + 1, 0:FLUSH * GN * T], indices=[c, r])
                        o_out.put(o_l2, indices=[c])

                    with air.herd([range(NC), range(NR)], name="gemv", shape=(NC, NR), link_with=obj,
                                  params=[go, nt, nk, kcr, nl, nkh]) as hd:

                        @hd.body
                        def _(tx, ty):
                            xb = air.alloc([KMAX * T], bf16, scope=hd.private())
                            xe = air.alloc([KMAX * 8], bf16, scope=hd.private()) if T == 1 else xb
                            wb = [air.alloc([PKT], i8, scope=hd.private()) for _ in range(2)]
                            acc = air.alloc([2 * GN * 8], f32, scope=hd.private())   # a FLUSH pair
                            ob = air.alloc([FLUSH * GN * T], bf16, scope=hd.private())
                            for _g in air.sequential(go):
                                x2l1.get(xb, indices=[tx, ty])
                                if T == 1:
                                    expand(xb, xe, kcr)
                                else:
                                    prep(xb, kcr)       # fused expert FFN (down phase): SiLU(gate) * up
                                for _f in air.sequential(0, nt):
                                    zero(acc)
                                    for kk in air.sequential(0, nk):
                                        w2l1.get(wb[0], indices=[tx, ty])
                                        acc_k(wb[0], xe, acc, kk * 2, kcr)
                                        w2l1.get(wb[1], indices=[tx, ty])
                                        acc_k(wb[1], xe, acc, kk * 2 + 1, kcr)
                                    flush(acc, ob)
                                    o2l2.put(ob, indices=[tx, ty])
                                for _l in air.sequential(0, nl):       # last tile alone, zero partner
                                    zero(acc)
                                    for kk in air.sequential(0, nkh):
                                        w2l1.get(wb[0], indices=[tx, ty])
                                        acc_k(wb[0], xe, acc, kk * 2, kcr)
                                        w2l1.get(wb[1], indices=[tx, ty])
                                        acc_k(wb[1], xe, acc, kk * 2 + 1, kcr)
                                    flush(acc, ob)
                                    o2l2.put(ob, indices=[tx, ty])

    return lch


# ------------------------------------------------------------------ host layouts
def pack_block(q, sc, mn, tiles, kc):
    """One column's run: rows [NR * tiles * GN, K] (zero-padded) of raw Q4NX
    (q [rows, K] uint8, scale/min [rows, K/32]) -> bytes in stream order [tile][kc][row r][PKT].
    Core r of the column takes rows r*tiles*GN + [0, tiles*GN). mn None: q4s (no mins, PKT_S)."""
    from ml_dtypes import bfloat16 as _bf
    rows, K = q.shape
    R, Kp = NR * tiles * GN, kc * GK
    assert rows <= R and K <= Kp, (rows, R, K, Kp)
    qp = np.zeros((R, Kp), np.uint8)
    qp[:rows, :K] = q
    sp = np.zeros((R, Kp // 32), np.float32)
    mp = np.zeros((R, Kp // 32), np.float32)
    sp[:rows, :K // 32] = sc
    if mn is not None:
        mp[:rows, :K // 32] = mn
    # per packet: [GN/8][GK/8][8 k][8 n] nibbles, then scale, min [GK/32][GN]
    qt = qp.reshape(NR, tiles, GN // 8, 8, kc, GK // 8, 8)            # r t nb nn kc kb kk
    qt = qt.transpose(1, 4, 0, 2, 5, 6, 3).reshape(tiles, kc, NR, -1)  # t kc r [nb kb kk nn]
    qb = (qt[..., 0::2] | (qt[..., 1::2] << 4)).astype(np.uint8)

    def grp(a):
        a = np.asarray(a, _bf).reshape(NR, tiles, GN, kc, GK // 32)    # r t n kc g
        return a.transpose(1, 3, 0, 4, 2).reshape(tiles, kc, NR, -1).view(np.uint8)
    out = np.concatenate([qb, grp(sp)] + ([] if mn is None else [grp(mp)]), axis=-1)
    assert out.shape[-1] == pkt(mn is None)
    return out.reshape(-1)


def unpack_col(y_col, tiles):
    """One column's output [out_tiles/FLUSH][NR][FLUSH*GN] -> rows in core order (core r: rows
    r*tiles*GN + ...)."""
    to = out_tiles(tiles)
    y = y_col.reshape(to // FLUSH, NR, FLUSH * GN).transpose(1, 0, 2).reshape(NR, to * GN)
    return y[:, :tiles * GN].reshape(-1)


def x_block(xs):
    """[n<=8, K] inputs -> the batched device's per-column layout [K/8][8 tok][8] (zero-padded)."""
    n, K = xs.shape
    out = np.zeros((K // 8, BATCH, 8), np.float32)
    out[:, :n, :] = xs.reshape(n, K // 8, 8).transpose(1, 0, 2)
    return out.reshape(-1)


def unpack_col8(y_col, tiles):
    """One batched column's output [tiles/FLUSH][NR][FLUSH][GN/8 nb][8 tok][8 nn] -> [8 tok, rows]
    with rows in core order (core r: rows r*tiles*GN + ...)."""
    to = out_tiles(tiles)
    y = y_col.reshape(to // FLUSH, NR, FLUSH, GN // 8, BATCH, 8)
    return y.transpose(4, 1, 0, 2, 3, 5).reshape(BATCH, NR, to * GN)[:, :, :tiles * GN].reshape(BATCH, -1)
