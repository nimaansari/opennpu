# MoE on the fused decode: Qwen3-30B-A3B design note

Status: design only (read-only analysis, 2026-10-03). Nothing here has been built or run.
`FD` = `<mlir-air checkout>/programming_examples/fused_decode/fused_decode.py`.
`RT` = `~/projects/opennpu/opennpu/native/runtime.py`. Line numbers are as of today.

Target: qwen3_moe, K=2048, 48 layers, 32 q / 4 kv heads, DH=128 (attention impl 1x8x1,
N_ATTN_CU=4, KV_PER_CU=1), qk-norm, 128 experts, top-8, moe_intermediate 768, no
shared expert, vocab 151936 (untied), rope theta 1e7.

Core idea: the 8 selected experts form one dense SwiGLU FFN with INTER = 8*768 = 6144,
gate/up rows stacked per expert and down columns stacked per expert. Each expert's GLU
output is scaled by its routing weight. The host picks experts by binding 8 expert
sub-buffers (`pyxrt.bo(parent, size, offset)`, already proven) as kernel arguments.
Nothing is copied.

---------------------------------------------------------------------------------------
## 0. Facts about the current engine that drive the design

1. **One dispatch = one instruction stream, which runs the waves in `range(UNI_WAVE_LO,
   UNI_WAVE_HI)`.** The launch is `air_api.launch(repeat=...)` at FD:1578. Each wave
   reads its DDR slabs at `iv*SLAB` (`_lb`/`_lbx`, FD:2169-2181). The arm is computed
   per wave: `iv<UNI_DEC ? 1 : 0` (FD:2352-2358) and promoted by `_arm_of_wave`
   (FD:2061). `UNI_WAVE_LO/HI` (FD:1311-1312) is documented as keeping the CDO fixed
   while it restricts which waves a stream drives. Today it is a hard constant.
2. **Memtiles are static and count-free; the cores carry the phase structure.** The
   weight fan (FD:3025-3055), the X memtile `_feed_inX` (FD:2952-2964), egress assembly
   (FD:3069-3101) and the down_buffer (FD:4142-4156) are each lowered to a BD ring with
   one fixed per-step pattern. A segment-scope gate folds to the wave-0 arm (FD:2920-2935
   comment), so a memtile cannot change its pattern per arm. The proj cores loop
   `ph in [0, nph_v)` over runtime-selected `I2P/J2P/DEST` tables (FD:4261, FD:4419).
   **A new schedule must keep every memtile's per-step pattern unchanged.** Both options
   below do.
3. **The rms core is a fixed two-norm, two-residual state machine** (FD:4790-4853):
   `get x, rmsW, rmsW2` -> `norm(x,w)` -> `refeed(XN_REFEED)` -> `get o (2048)` ->
   `h=x+o` -> `norm(h,w2)` -> `refeed(REFEED[GATEUP])` -> `get down (2048)` -> `r2=h+down`
   -> `layerOut(r2)`. The o and down landings are the same size
   (`OPROJ_RNDS*PAYLOAD = DOWN_RNDS*PAYLOAD = 2048` here).
4. **The X feed is one convergent packet channel, `@xnorm`.** Its producers are the rms
   core (two collapsed refeeds that share one buffer-free lock), the o-gather memtile
   (`refeed(OPROJ_REFEED)`, FD:4028) and the down_buffer (`refeed(DOWN_REFEED)`,
   FD:4152). They are time-disjoint because each waits on data the previous phase made.
5. **Weight feed.** For each phase, `_feed_wcols` (FD:2266) calls `_feed_wcol` (FD:2221)
   for each column. With `W_DUAL_CHAN=1` (the default, FD:924) a column's slab is
   `[ch0 half | ch1 half]`. Each half goes out as **two contiguous puts** on that
   column's own pinned channel `inW{ci}c{cx}`. The pair is deliberate: the shim
   coalescer merges it and tags it `air.coalesced_shim_feed`, and that tag is the
   cross-channel phase barrier ("a single put would skip coalescing and lose that
   barrier", FD:2224-2227). The puts are issued round-major across columns because the
   cores consume in X-broadcast lockstep (FD:2589-2596).
6. **The host packs with ITERATION-major order** (`q4_0_codec.pack_q4k_cascade_fast`,
   q4_0_codec.py:146-163): stream `[cx][h][i][j][cy']`, global row-block
   `gi = i*16 + cx*4 + cy`. Iteration `i` owns dense rows `[512i, 512i+512)`, and egress
   round `r` of a phase delivers exactly those rows (`ROUNDS_PER_PH = I2*PAIR_ROWS =
   nbi_pc`). This is what makes the per-expert packing below work.
7. **The GLU core** (FD:4072-4130) takes `NGLU` slices of `GLU_SLICE = [up | gate]` off
   the gate-up demux dest and calls `glu_aie(y, x, arm)` (glu.cc:58). It ping-pongs the
   slices in pairs, so NGLU must be even. Down then reads the GLU output in slice order
   as its K.

---------------------------------------------------------------------------------------
## 1. Geometry of the new model entry

PAIR_ROWS=2, so one row unit is ROW_BLOCK*NCX*NCY*PAIR_ROWS = 1024 rows. J2 = K_phase/512.

| phase | rows (out) | K (contract) | I2P | J2P | DEST |
|---|---|---|---|---|---|
| gate-up (8 experts) | 2*6144 = 12288 | 2048 | 12 | 4 | glu |
| down (8 experts) | 2048 | 6144 | 2 | 12 | rms |
| QKV | 4096+512+512 = 5120 | 2048 | 5 | 4 | rope |
| o-proj | 2048 | DQ = 4096 | 2 | 8 | rms |

Derived values (check each with `FUSED_DECODE_PRINT_CONST`):

- `NUM_Q_HEADS = (5120-2*4*128)/128 = 32`.
- `Q_HEADS_PADDED_PER_CU = 8` (GQA_SEG=8), `DQ_PADDED = 4096`, `SSZ_BLK = 192`.
- `MIX_RNDS_QKV = 10`, so `QKV_ONE_BD` is off (10+4 <= 16).
- Gate-up rounds = 24. **Force `GLU_PKTS = 1`** (see §3): `GLU_SLICE = 512`,
  `GLU_HID = 256`, `NGLU = 24` (even), `GLU_OUT = 6144`.
- LM head: `VOCAB_SIZE_PADDED_FULL = 75*2048 = 153600`, which is 4800 row-blocks.
  `VOCAB_ROWBLKS = 32*VOCAB_I2` must divide 4800, and 4 must divide `VOCAB_RNDS =
  2*VOCAB_I2`, with `2*VOCAB_I2 <= 63`. That leaves VOCAB_I2 in {2, 6, 10, 30}.
  Take **VOCAB_CHUNK_I2=30, UNI_LM=5**.
- `RCACHE_LEN = 2*12*8 = 192`.
- `LEAN_RMS` is off: 2048*2*7 + 10240 < 64 KiB, so the default `DECODE_STACK=10240`
  fits.
- Attention 1x8x1 with KV_PER_CU=1 already exists (qwen2.5-7b, FD:381-391). The new
  combination is 1x8x1 with qk-norm and PAIR_ROWS=2; M2 must validate it.

Q4NX costs 5120 B per 8192 weights, i.e. 0.625 B/weight.

| item | weights | bytes |
|---|---|---|
| attention (QKV + o), per layer | 18.87 M | **11.80 MB** |
| one expert gate-up (1536 x 2048, 384 blocks) | | 1,966,080 B |
| one expert down (2048 x 768, 192 blocks) | | 983,040 B |
| one expert slab | | **2,949,120 B** (720 x 4 KiB, page-aligned) |
| active experts per layer (8) | 37.75 M | **23.59 MB** |
| all experts, resident (48 x 128) | | 18.1 GB |
| LM head | 314.6 M | **196.6 MB** |

The full resident footprint is about 18.9 GB of host_only BOs. Confirm the box's RAM
before M3.

---------------------------------------------------------------------------------------
## 2. Q1: dispatch structure

The router for layer k needs `post_attention_layernorm(x_k + attn_k)`. It is computed
mid-layer, and the expert choice has to reach shim BD addresses that only the host can
set. So **every layer needs one host round trip**. An instruction stream cannot wait on
the host, and BD addresses are fixed when the run is submitted.

### Option A, rotated single-arm program (RECOMMENDED)

One program, one decode arm (the existing arm 1), one wave per dispatch (`UNI_DEC=1`).
The phase order is rotated:

```
dispatch k (k = 0..48):  [ gate-up(E_{k-1}) -> glu*w -> down(E_{k-1}) -> x_k = h_{k-1} + ffn ]
                         [ input_norm(x_k) -> QKV(k) -> rope/attn(k) -> o(k) -> h_k = x_k + o ]
                         layerOut: h_k -> x BO (in place)
host: h_k -> rmsnorm(post_ln_k) -> router_k -> top-8 + renorm -> bind E_k, w_k -> dispatch k+1
```

Edge dispatches reuse the same program:

- **k = 0**: the FFN half runs with **all 8 routing weights = 0** (any real expert
  sub-buffers, e.g. layer 0's experts 0..7). GLU becomes `silu(g)*u*0 = 0`, so down is 0
  (including the `+min*sum(x)` term) and `x_0 = embedding`.
- **k = 48**: the attention half uses a **zero-attention slab** (QKV and o weights all
  q=0, scale=0, min=0) and a **zeroed scratch KV sub-buffer**. Then q, k and v are 0,
  qk-norm(0) = 0, softmax over zero keys gives o = 0, and o-proj(0) = 0, so the result
  is `h' = x_48`, the final hidden state. Keep the scratch KV zeroed: a NaN there would
  survive the multiply by the zero weights (0*NaN).
- Cost of the two edge dispatches: 23.6 MB + 11.8 MB of extra streaming, about 1 ms per
  token. Removing it is an M5 item.

**What changes on the device: almost only tables, plus two swapped refeed counts.**

1. Model entry with `DEST=["glu","rms","rope","rms"]`, `I2P=[12,2,5,2]`,
   `J2P=[4,12,4,8]`.
2. **Derived constants that assume the natural order**, each fixed so existing models
   produce byte-identical IR:
   - FD:732 `MIXER_NAME = DEST[0]`. Make it the first DEST whose base name is in
     {rope, conv, mix}.
   - FD:787-795: the "ph0 must be a mixer" check. Apply it to `MIXER_NAME` instead.
   - FD:1175 `DEMUX = list(dict.fromkeys(DEST_NAMES))`. Order it canonically,
     `sorted(..., key=["rope","rms","glu","mix",...].index)`, so rope stays ordinal 0.
     This keeps the hard-coded `indices=[0, 0]` QKV gets in `_rope_body` (FD:3178,
     FD:3181) valid. For every shipped model the order is unchanged.
   - FD:1178 `DOWN_PHASE = NPH - 1`. Use the first phase after `GLU_PHASE` whose name is
     "rms" (falling back to NPH-1). Then `OPROJ_PHASE` (FD:1387), `RMS_PHASES`, `FULL4`
     (FD:1426), `RMS_DEST`, `HOST_DRAIN` and `LAYER_RNDS` all derive correctly:
     GATEUP=0, DOWN=1, KV_PHASE=MIXER_PHASES[-1]=2, OPROJ=3.
   - FD:1244 `VOCAB_J2 = J2P[0]`. Use `K // (2*COL_BLOCK)`. The value is the same here,
     but the current form is wrong by accident.
   - FD:2986 `_xc_dec` multiplies `OPROJ_REFEED` by `K//512`, but the o-proj X is
     `DQ = 4096` wide. Use `DQ//512` for that term. qwen3-4b (DQ != K, FD:449) runs
     today, so the ring may not care, but match the count anyway.
3. **rms core** (FD:4802-4853): **swap the two refeed counts**. The step order, buffer
   sizes and get/put shapes are identical: the first landing is down (2048) instead of
   o (2048), and the second is o instead of down.
   ```python
   _RF1, _RF2 = ((REFEED[GATEUP_PHASE], XN_REFEED) if MOE_ROT
                 else (XN_REFEED, REFEED[GATEUP_PHASE]))
   # step1: rms_norm_aie(a_xn, a_x, a_w)          -> refeed(_RF1, xnorm.put)   # was _xn_refeed
   # step2: get(down) ; h = x + down ; rms_norm_aie(a_xn, a_h, a_w2) -> refeed(_RF2, ...)
   # step3: get(o) ; r2 = h + o ; layerOut(r2)     -> this is h_k for the router
   ```
   The host lays out the per-dispatch rms slab as `[post_ln(k-1) | in_ln(k)]`. rmsW and
   rmsW2 then sit at the same offsets they use today (FD:2510-2514). The shared
   buffer-free lock already starts at the larger of the two refeeds (FD:4642-4650 note),
   so it does not change.
4. **`@xnorm` time order** becomes rms(gate-up) -> down_buffer -> rms(QKV) -> omtb(o).
   That is the same rms / memtile / rms / memtile alternation as today, and each
   producer is still gated by data the previous phase made, so they stay time-disjoint.
5. **Shim stream** (`_uni_dec`, FD:2496-2813). The phase loop (FD:2731) sends phases 0
   and 1 to the new expert feed (§3). Phases 2 and 3 use `_feed_wcols` from W with
   `woff` counting attention phases only. `_emit_append`/`_emit_readback` already follow
   `p == KV_PHASE` (FD:2757), which is now phase 2. Dependency check: append waits on
   rope K/V, rope on QKV, QKV-X on rms, rms on the down output, and down on its weights,
   which were issued earlier. No cycle.
6. **GLU scaling** (§4) and the expert feed (§3).

Host changes: §7. Dispatch count: 49 layer dispatches + 1 LM = **50 per token**.

### Option B, two arms per layer

Add arm 3 = "moe" (norm, gate-up, glu*w, down, residual) beside arm 2 = "attn" (QKV,
attn, o, residual, emit h). One wave per dispatch, with the arm set by which instruction
stream the host submits.

- **Arm plumbing.** The arm has to reach every herd as a real per-dispatch RTP. Use the
  hybrid route: `_seg_arm_rt = air_api.rtp(...)` computed at launch scope (FD:2848-2865)
  and `_core_arm` for the herds (FD:2936-2941). Today that route only exists when
  `HYBRID_MIXER` is set, and that flag drags in the conv machinery
  (`assert not HYBRID_MIXER or CONV_MIXER`, FD:657). It would need decoupling.
- **proj cores** (FD:4261/4419): change `for ph in for_(0, nph_v)` to
  `for_(lo_v, hi_v)`, where `lo/hi = sel(arm)` gives [2,4) for moe, [0,2) for attn and
  [0,1) for vocab. Small change.
- **rms core**: three bodies (lm / attn / moe) under nested `ops.switch` (FD:4691). The
  attn body is norm(in_ln), refeed(10), get o, layerOut(h). The moe body is norm(post),
  refeed(24), get down, layerOut. **Each arm's channel ops get their own BDs on the rms
  tile's 2 S2MM + 2 MM2S.** The LM_HEAD comment (FD:4676-4681) already flags this
  doubling, and this is the main build risk.
- **Gating**: glu idle in attn (via `_arm_only`). rope, the attention CUs and the KV
  feeds are gated off in moe, the way vocab gates them today (FD:2476-2484). Memtiles
  need no change: a ring with no traffic waits.
- **Cost**: 97 layer dispatches + 1 LM. That is +48 dispatches over A (§6), and each
  arm's dispatch is short (11.8 MB or 23.6 MB), so fixed overheads weigh more.
- **Upside**: natural phase order, all weights in a dispatch belong to the same layer,
  and no zero-weight edge dispatches.

### Option C variants

- **C1 = A with the LM head fused into the last dispatch.** Use the full stream
  `[layer wave, 5 vocab waves]` for k=48. This saves one dispatch and is free once A
  works (the vocab arm already follows a decode wave inside one stream). Do it in M5.
- **C2 = A with patched expert offsets instead of 8 sub-buffer args.** Pass one
  per-layer expert parent (377 MB) as a single arg. The host patches the 64+256
  BD-offset words per dispatch, located by diffing two builds that bake different expert
  indices (the same calibration `decode_insts_gen.py:56-75` uses for L). This is the
  **fallback if the arg count (13) or per-dispatch sub-BO binding cost bites**. It costs
  rewriting about 320 words plus one insts sync per dispatch.
- **Rejected**:
  - Device-side top-k: BD addresses can't be data-dependent.
  - Speculative or pre-gated routing from h_{k-1}: inexact.
  - Host memcpy of the selected experts into a staging BO: 23.6 MB/layer, more than
    1 GB/token of memcpy.

**Recommendation: A.** It adds no arm and no new herd gating, keeps the rms-tile BD
count, and leaves memtile patterns unchanged. Its device diff is a phase table, a
handful of derived-constant fixes, two swapped refeed counts, the expert feed and a GLU
scale. B needs new arm plumbing through every herd plus a third rms body (BD budget),
and runs about twice the dispatches. Keep B as the fallback if the rotated `@xnorm`
order or the rms swap misbehaves on hardware.

---------------------------------------------------------------------------------------
## 3. Q2: feeding gate-up and down from 8 expert tensors

### 3.1 Arguments

Append 8 tensors after the base five (FD:1552). This is the same scheme `wg*` uses
(FD:1505-1512, WARG FD:2108-2110):

```python
EXP_GU = 384 * BLOCK_BF16          # 983,040 elems
EXP_DN = 192 * BLOCK_BF16          # 491,520 elems
EXP_LEN = EXP_GU + EXP_DN          # 1,474,560 elems = 2,949,120 B
_tensors = ([x_t, w_t, rms_t, y_t, kvc_t]
            + ([air_api.tensor([EXP_LEN], api_types.bf16, name=f"e{i}") for i in range(MOE_TOPK)]
               if MOE else [])
            + ...wg/wvoc (unused: DECODE_WGROUP=0)... + seqlen)
# launch_body: EB = [_la[4 + 5 + i] for i in range(MOE_TOPK)]
```

- **Arg count**: 13 BOs (x, w, rms, y, kvc, e0..e7).
- **Proven so far**: 9 (qwen3-8b with W_SPLIT: 5 + wg1..3 + wvoc, bound at RT:268).
- 13 must be checked in M1 (§8).
- If it fails, fall back to C2.
- Every stream from this xclbin (layer and LM) has the same signature, so the LM
  dispatch binds any valid BO to e0..e7.

`W_LAYER` (FD:1467) and `W_TOTAL_BLOCKS` (FD:1358) must sum **non-MoE phases only**.
`W_FAN_STEPS` (FD:1357) keeps summing **all** phases: the memtile fan is phase-agnostic
and count-free.

### 3.2 Per-expert packing (host) and why the concatenation is exact

For layer l and expert e (HF: `mlp.experts.{e}.{gate,up,down}_proj.weight`, router
`mlp.gate.weight [128, 2048]`):

```python
GLU_CHUNK = fd.GLU_CHUNK                     # 256 with GLU_PKTS=1
GU_e = _interleave(up_e, gate_e, GLU_CHUNK)   # 1536 x 2048: [U0:256|G0:256|U256:512|G256:512|U512:768|G512:768]
slab_e = concat(pack(GU_e, NCX=4, NCY=4, dual_chan=True, mins=..),   # 384 blocks, nbi_pc = 48/16 = 3
                pack(down_e, NCX=4, NCY=4, dual_chan=True, mins=..)) # 192 blocks, nbi_pc = 64/16 = 4, nbj = 3
```

`pack` is `pack_q4k_cascade_fast` (RT:108). Per-expert stream order is
`[cx][h][i'][j][cy']`.

- **Gate-up.** The dense matrix is `vstack(GU_0..GU_7)` (12288 rows). Iteration `i`
  owns dense rows `[512i, 512i+512)`, so **expert e owns exactly iterations
  `[3e, 3e+3)`** (1536/512 = 3). The dense row-block `gi = i*16+cx*4+cy` equals
  `48e + gi'` of expert e. Each (cx, h) channel stream is
  `for i: for j: for cy'`, which equals **concat over e of expert e's (cx,h) run**.
  The run is 3 i x 8 j x 2 blocks = 48 blocks = 122,880 elems, at offset
  `(cx*2+h)*48*BLOCK_BF16` inside the expert slab.
- **Down.** The dense matrix is `hstack(D_0..D_7)` (2048 x 6144). Expert e owns
  col-blocks `j in [3e, 3e+3)`. The dense (cx,h) stream is
  `for i in 0..3: for j in 0..23: for cy'`, which equals **for i: concat over e of
  expert e's (cx,h,i) run**. That run is 3 j x 2 blocks = 6 blocks = 15,360 elems, at
  `EXP_GU + ((cx*2+h)*4 + i)*6*BLOCK_BF16`.
- **The GLU output is already in down's K order.** Gate-up egress round r is dense rows
  `[512r, 512r+512)` = `[up 256 | gate 256]` of expert `r//3`, local chunk `r%3`. With
  GLU_PKTS=1 that is exactly one GLU slice. Its 256 outputs are intermediate indices
  `[256r, 256r+256)`, which is expert `r//3` local `[256(r%3), +256)`. That matches the
  `hstack` column order.

**Why GLU_PKTS=1** (FD:1443 derives 2 here, since (24//2)%2 == 0): with chunk 512, an
expert's 768 rows straddle the 512-row up/gate interleave. The expert then owns
different rows in different columns (rows 1024-1279 are expert 0 in cx=0,1 and expert 1
in cx=2,3), and the scale boundary falls mid-slice. With chunk 256, every expert is 3
whole iterations in every column and 3 whole GLU slices. Make it a model key:
`GLU_PKTS = MODEL.get("GLU_PKTS", <derived>)`. The 1-packet slice is the proven
qwen2.5-7b path (FD:1438-1442), and NGLU=24 stays even.

### 3.3 Feed code (replaces `_feed_wcols` for MoE phases, inside `_uni_dec`)

```python
RUN_GU = 3 * NBJ_PH[GATEUP_PHASE] * (NCY // 2) * blk   # 48 blocks
RUN_DN = (NBJ_PH[DOWN_PHASE] // MOE_TOPK) * (NCY // 2) * blk  # 3 j * 2 blocks = 6 blocks
NI_DN = I2P[DOWN_PHASE] * PAIR_ROWS                    # 4

def _feed_moe(p, EB):
    assert W_DUAL_CHAN
    if p == GATEUP_PHASE:
        rounds = [(e, lambda cx, h: (cx * 2 + h) * RUN_GU, RUN_GU) for e in range(MOE_TOPK)]
    else:  # DOWN_PHASE
        rounds = [(e, lambda cx, h, i=i: EXP_GU + ((cx * 2 + h) * NI_DN + i) * RUN_DN, RUN_DN)
                  for i in range(NI_DN) for e in range(MOE_TOPK)]
    for e, off, sz in rounds:              # ROUND-MAJOR: run r of every channel before r+1
        for cx in range(NCX):
            for h in range(2):
                ChannelPut(_wname(h, cx), EB[e], indices=[idx(0)],
                           offsets=[idx(off(cx, h))], sizes=[sz], strides=[1])

# phase loop FD:2731:
#   if p in MOE_PHASES: _feed_moe(p, EB); continue
#   else: _feed_wcols(_lo(_wbase, woff), _colspan, per_col // NCY); woff += ...
```

Every run is a whole number of fan steps (`(NCY/2)*blk` per step): 24 for gate-up, 3 for
down. The memtile fan get (FD:3038-3045) is unchanged.

**Put counts** per dispatch:

| phase | puts per channel | total over 8 channels |
|---|---|---|
| gate-up | 8 | 64 |
| down | 32 (4 i x 8 e) | 256 |
| QKV, o | 2 each, unchanged | |

Down needs 32 puts per channel because `i` (the core's row loop) is outside `j`, so
every row-block sweeps all 8 experts. Packing order cannot fix this: a run can only come
from one buffer. **Pack each expert's down as its own `[cx][h][i][j'][cy']` cascade and
iterate experts inside each row-block in the feed.** That needs no core change. Changing
the core loop to j-outer with 4 live accumulators would cut it to 8 puts and also cut X
re-feed traffic. But it rewrites the proj GEMV loop, the X refeed counts and the
down_buffer refeed into 8 sub-slices, so treat it as an M5 optimization only.

**Limits**:

- **BDs**: the shim reuses BD IDs between sequential tasks (issue_token=false). The
  existing per-token streams already issue far more than 16 tasks per channel (the
  vocab and KV comments at FD:2463-2468 and FD:2652-2659), so the 16-BD shim limit is
  not a hard cap. Each extra task costs instruction-stream words plus controller issue
  time.
- **Run size**: down runs are 30 KB. At roughly 5-7 GB/s per channel that is about
  5 us of data per task, so issue overhead may be visible. Measure it in M2.
- **Main risk: the coalescer phase barrier.** `_feed_wcol` relies on two *contiguous*
  puts being merged and tagged `air.coalesced_shim_feed`, so that AIRRtToNpu groups the
  2*NCX channels into one phase group (FD:2221-2264, 2275-2281). Puts on different BOs
  are not contiguous, so they will not merge and will not carry the tag. The compiler
  source is not in this tree (no AIRRtToNpuPass.cpp), so I could not confirm what
  happens to untagged multi-put feeds. In M1, inspect the lowered NPU sequence: the
  per-channel task order must be round-major across all 8 channels, with the depth-2
  awaits interleaved and not fenced per channel.
- **Fallback for the barrier**: emit each round as an explicit group of 8 puts (already
  the case above), and if needed mark the group with the same attribute the coalescer
  would add. Failing that, take C2 (one buffer per layer, patched offsets), which keeps
  each round's puts on one memref.

---------------------------------------------------------------------------------------
## 4. Q4: routing weights to the GLU core

**Transport: a new tiny channel `@moeRW`, shim -> glu tile S2MM1.** The glu tile's
S2MM0 carries `outY[0,GLU_DEST]`, so S2MM1 is free.

- FD:2056: `_chan("moeRW", size=[1])` (only if MOE).
- `_uni_dec`, before the phase loop:
  `_CH["moeRW"].put(rms_t[_moe_off : _moe_off + 16])`, where
  `_moe_off = UNI_DEC*RMS_LAYER + ROPE_W_LEN + K` (just after the final-norm slot; grow
  `rms_t`, FD:1519-1527, by 16). The host writes the 8 weights there before each
  dispatch (8 bf16 + 8 pad).
- **Shim MM2S budget is tight.** By my count the dual-channel attention build already
  uses 15 of 16 MM2S: 8 inW, rmsX+rmsW2 (packet-shared), rmsW, ropeLUT, 2 inKV_K,
  2 inKV_V. moeRW takes the last one. If allocation fails, declare it
  `channel_type="npu_dma_packet"` so it can share the rmsX/rmsW2 shim MM2S, as rmsW2
  already does (FD:1845-1851).

**glu herd** (FD:4085-4123):

```python
def _dec():
    rw = air_api.alloc([16], api_types.bf16, scope=_glu_h.private())
    _CH["moeRW"].get(rw[0:16])
    def _slice(sidx_i32):
        ... get gx ...
        glu_moe_aie(gh, gx, rw, sidx_i32, _arm)        # was glu_aie(gh, gx, _arm)
        ... put gh ...
    for _s in air_api.sequential(NGLU // 2):
        s0 = arith.muli(_s.materialize(), idx(2))       # as proj does with _j (FD:4349)
        _slice(arith.index_cast(i32, s0))
        _slice(arith.index_cast(i32, arith.addi(s0, idx(1))))
    air_api.dealloc(rw)
```

Declare `glu_moe_aie = _kernel("glu_moe_aie", [glu_hid_l1, glu_x_l1, rw16_l1, i32, i32],
"glu.o")` beside FD:1757. The tile gains one BD (rw).

**glu.cc** (minimal):

```cpp
#ifdef MOE_SLICES_PER_EXPERT   // = 768 / (GLU_SLICE/2) = 3, from models/qwen3-30b-a3b.h
template <int L>
void pseduo_glu_scaled(bf16 *y, const bf16 *x, bf16 w) {
  aie_round_nearest_even();
  const bf16 *hid = x, *gate = x + L / 2;
  aie::vector<bf16, 16> wv = aie::broadcast<bf16, 16>(w);
  for (int i = 0; i < L / 2; i += 16) {
    aie::vector<bf16, 16> g = getActivationBf16(aie::load_v<16>(gate + i));
    aie::vector<bf16, 16> t = aie::mul(g, aie::load_v<16>(hid + i));
    aie::store_v(y + i, aie::mul(t, wv).template to_vector<bf16>());
  }
}
extern "C" void glu_moe_aie(bf16 *restrict y, bf16 *restrict x, bf16 *restrict rw,
                            int slice, int _arm) {
  (void)_arm;
  pseduo_glu_scaled<GLU_SLICE>(y, x, rw[slice / MOE_SLICES_PER_EXPERT]);
}
#endif
```

Precision is identical to scaling the f32 product once. A routing weight of 0 makes the
slice exactly 0, which the k=0 edge dispatch relies on.

**Alternative transport (no new channel): 8 herd RTPs.** Add
`params=[rtp(arm), rtp(w0..w7)]` to `_glu_h`, select `w[slice//3]` in the herd body with
`_psw`, and pass the f32 bits as an i32 to the kernel. The RTP writes are instruction
words, so the host patches 8 words per dispatch in the shared insts BO (located like the
L words). This is cheap, but it couples the routing to the insts patcher. Use it only if
moeRW cannot be routed.

**Rejected**: folding w into the up rows' Q4 scale/min. That means about 1.5 MB of
header rewrites per layer, roughly 75 MB/token of host writes.

---------------------------------------------------------------------------------------
## 5. Q3 and Q5: KV cache and the LM head with per-layer dispatch

**KV.** Build with `UNI_DEC=1` (the layer count lives in a new `N_LAYERS_MODEL=48`
key). The wave index is 0, so `_kbase = 0` (FD:2350), and the append and readback
offsets are relative to the kvc arg.

- Pass **layer k's KV region as a sub-BO**:
  `kv_k = xrt.bo(kvc_parent, LREG*2, k*LREG*2)`, with `LREG = ATTN_MAXL*KVSZ_TOK =
  ATTN_MAXL*1024` elems (2 KB per token per layer). Add a 49th scratch region (zeroed)
  for the k=48 dispatch.
- **One insts stream serves all 48 layer dispatches.** Its L-dependent words (the CU
  RTP-L writes, plus the appendK/appendV slot `(L-1)*REGION_W` from `_loi_slot`,
  FD:2210-2218) are layer-independent. So `patch_insts(state, p+1)`
  (decode_staircase.py:115) runs **once per token**, exactly as today (RT:310).
- Calibration is unchanged: two builds `decode_layer_L{a}`, `decode_layer_L{b}` at the
  same ATTN_MAXL. `DecodeInstsGen` keys on the file prefix (decode_insts_gen.py:39-49),
  so add a `prefix=` argument.
- With a single window, `KVGeometry` re-spacing is not needed. For a staircase,
  pre-create sub-BOs per (window, layer).
- `DYNSEQ` is not needed.

**LM head as a separate final dispatch.** Make `UNI_WAVE_LO/HI` env-driven at FD:1311
(`DECODE_WAVE_LO`, `DECODE_WAVE_HI`) and build three streams from one xclbin:

| stream | waves | what it is |
|---|---|---|
| layer | [0, 1) | layer program, L-patched |
| lm | [1, 6) | 5 vocab chunks, no L words, static |
| full (M5, for C1) | [0, 6) | layer + LM |

The vocab arm is selected per wave by `iv >= UNI_DEC`, so it is correct in a range
build. Bindings for the lm stream:

- **x**: the final hidden state (left in the x BO by dispatch 48).
- **w**: a sub-BO whose offset `W_LAYER` lands on the vocab weights (`_vwb =
  UNI_DEC*W_LAYER + (iv-1)*VOCAB_W`, FD:2410-2416). Store the attention parent as
  `[attn_0 | ... | attn_47 | zero_attn | vocab]` and pass the sub-BO starting at
  `zero_attn`. That one slab doubles as the k=48 attention slab.
- **rms**: a slab carrying `final_norm` at `UNI_DEC*RMS_LAYER + ROPE_W_LEN`
  (FD:2441-2444).
- **y**: logits at `(HOST_ROUNDS+LAYER_RNDS)*PAYLOAD = (10+4)*512` (RT:248 formula
  unchanged).

**Must verify in M1**: the three range builds produce **byte-identical CDO/PDI** (diff
the device part of `air_project/*.mlir` or the xclbin PDI sections). Otherwise the
streams cannot share a hw_context. The FD:1307-1310 comment claims this holds; it has
not been re-checked on this tree.

---------------------------------------------------------------------------------------
## 6. Q6: per-token time model (estimate, to be replaced by M0/M4 measurements)

Bandwidth anchor: Llama-1B sustains **35.7 GB/s** on this builder (README:190).

**Bytes per token:**

| item | MB |
|---|---|
| 48 x (11.80 attention + 23.59 experts) | 1,698.7 |
| edge dispatches (zero FFN + zero attention) | 35.4 |
| LM head | 196.6 |
| KV readback: 48 x ATTN_MAXL x 2 KB | 24 at a 256 window; 192 at a 2048 window |
| **total** | **about 1.95 GB at short context** |

Streaming at 35.7 GB/s gives **about 55 ms**. Add +5% for down's fragmented 30 KB runs
(about 7.9 MB/layer at reduced efficiency) for **about 57 ms**.

**Per-dispatch overhead sources (x50 for A, x98 for B):**

1. Host submit plus wait: pyxrt `kern(...)` and `.wait()`, completion interrupt,
   Python. My estimate is **50-150 us**.
   - Measure it in M0.
   - Pre-build one `xrt.run` per layer and only `set_arg` the 8 expert BOs.
   - Move the loop to C++ if it dominates (`bench_decode.cpp` is a starting point).
2. Per-run instruction-stream preamble (BD/lock reprogramming of the layer program) and
   the herd RTP/lock arm: **about 20 us**, which is the measured wave-barrier cost
   (FD:1215-1218).
3. Pipeline fill/drain. A cold start streams the first weights while the cores are
   idle, and the tail (down -> rms -> layerOut -> attention o) runs with no next-layer
   prefetch. A fused multi-layer dispatch hides this; per-layer dispatch exposes it.
   Estimate **50-100 us**.
4. Host router per layer:
   - sync x FROM (4 KB)
   - numpy rmsnorm, 128x2048 GEMV, softmax, top-8, renorm: about 30-50 us
   - write the 8 routing weights and sync 32 B
   - **total about 60-80 us**

That is about 180-350 us per layer dispatch:

| | total ms | tok/s |
|---|---|---|
| Option A: 50 x 0.18-0.35 ms = **9-17 ms** overhead | **66-74** | **about 13.5-15** |
| Option B: 98 dispatches, **+8-15 ms** | 75-90 | about 11-13 |

At a 2k context, add about 5 ms of KV readback.

Main levers after correctness:

- C++ host loop: saves part of (1) and (4).
- C1 (fused tail): saves 1 dispatch.
- Dropping the k=0 zero-FFN with a dedicated "attention-only first layer" stream:
  saves 0.66 ms. That needs a different core phase count, so it is not free.
- j-outer down: recovers down efficiency.

---------------------------------------------------------------------------------------
## 7. Host driver (new `MoeDecoder`, alongside RT `NativeDecoder`)

**Weights** (`build_weights_moe`, streaming per layer like RT:103-168):

- `attn.W.bin`: per layer `pack(qkv)|pack(o)` with `DEST` order QKV then o, then
  `zero_attn`, then the vocab chunks (RT:154-163).
- `experts.L{l}.bin`: 128 slabs of 2,949,120 B (§3.2).
- `aux.npz`: in_ln, post_ln, q_norm, k_norm and the router `[48,128,2048]` (keep f32 or
  bf16 for the CPU router).
- **Requant cost.** RT's per-layer path is vectorized, and experts are 29 G weights, so
  expect about 30-60 min one-time. Parallelize per layer.

**BOs** (all host_only, created once):

| BO | contents | how it is bound |
|---|---|---|
| `x` | hidden state, 4 KB | arg 0 |
| `y` | logits | arg 3 |
| `insts` | layer stream; separate `lm` insts BO | |
| `attn_parent` | 49 attention slabs + vocab (about 0.8 GB) | sub-BOs `attn_k` |
| `rms_parent` | 50 slabs, each padded to 16 KiB | sub-BO `rms_k` |
| `kv_parent` | 49 x LREG | sub-BO `kv_k` |
| `exp_parent[l]` | 48 x 377 MB, one per layer | 6144 expert sub-BOs `E[l][e]`, pre-created |

Each rms slab holds `[post_ln(k-1) | in_ln(k) | cos/sin(128) | q_norm | k_norm |
final_norm | rw(16)]`.

**Per token:**

```python
patch_insts(st_layer, p + 1)                        # once per token, all layers share it
write cos/sin(p) into the 49 rms slabs; sync once
x <- embed[tok]
for k in range(49):
    if k == 0:  E, w = E[0][0:8], zeros(8)          # zero FFN
    else:       E, w = sel_E, sel_w                 # from router of layer k-1
    rms_k.rw <- w ; sync 32 B
    run_k.set_arg(5 + i, E[i]) for i in 0..7         # run_k pre-bound: x, attn_k|zero, rms_k, y, kv_k|scratch
    run_k.start(); run_k.wait()
    if k < 48:
        h = x.read(); hn = rmsnorm(h) * post_ln[k]
        p_ = softmax(Wr[k] @ hn); top = argsort(p_)[-8:]
        sel_w = p_[top] / p_[top].sum()             # norm_topk_prob=True
        sel_E = [Eb[k][e] for e in sorted(top)]     # keep w aligned with the sorted order
run_lm (x, attn sub-BO at zero_attn, rms_48, y) -> logits
```

Do NOT fold `post_ln` into the experts. The device norm uses rmsW (slab offset 0), which
already holds post_ln(k-1).

---------------------------------------------------------------------------------------
## 8. Milestones (each testable on hardware)

**M0: overhead baselines (no new build).**
- Using the existing qwen3-8b build, time a single `kern()` of a `DECODE_UNI_DEC=1`
  bisect build (FD:1279) against a full dispatch. That gives the per-dispatch fixed cost
  and the cold-start cost.
- Time `xrt.bo(parent, size, off)` creation and `run.set_arg` for 8 args.
- Allocate and sync 19 GB of host_only BOs on the target box (RAM check).

**M1: build-only lowering.**
- Add the model entry `qwen3-30b-a3b` plus `models/qwen3-30b-a3b.h`:
  - MODEL_DIM 2048, NUM_ATTN_HEADS 32, NUM_KV_HEADS 4, INTERMEDIATE_SIZE 6144,
    GLU_SLICE 512, DH 128, ATTN_IMPL_1x8x1, HAS_QK_NORM, MOE_SLICES_PER_EXPERT 3
  - register it in all_models.h; the Makefile's GLU_SLICE_EXPECTED cross-check
    (Makefile:56-62) will hold the 512.
- Land the §2.A derived-constant fixes, the rms refeed swap, the 8 expert tensors plus
  `_feed_moe`, moeRW, `glu_moe_aie`, and env-driven `UNI_WAVE_LO/HI`.
- Gates:
  - `FUSED_DECODE_EMIT_ONLY=1` IR is unchanged for llama-3.2-1b / qwen3-8b / lfm2
    (`~/air_tools/decode_ir_gate.py`).
  - aircc compiles all three wave ranges.
  - Identical CDO across the ranges.
  - The xclbin kernel has 13 BO args.
  - The lowered sequence shows the 8 inW channels interleaved round-major in both MoE
    phases (§3.3 risk).
  - Shim channel allocation succeeds (moeRW).
  - `insts_for(L=1)` vs `L=2` differ only in the RTP-L + append words.

**M2: one-dispatch dense-equivalence vs a CPU reference.**
- Inputs: real layer-0 attention plus experts 0..7 of layer 0, routing weights
  1/8 each, plus a non-uniform case.
- Reference (numpy, f32 math on the dequantized Q4NX weights, which isolates the device
  from quantization error): `x1 = x0 + sum_e w_e * D_e(silu(G_e h) * U_e h)` with
  `h = rmsnorm(x0)*post`, then attention at L=1..32 with a synthetic KV, then
  `h1 = x1 + o`. Compare the x BO.
- Sub-tests:
  - (a) w = 0 gives x1 == x0 bit-exact, and checks the k=0 edge.
  - (b) zero_attn plus zeroed scratch KV gives h1 == x1, and checks the k=48 edge.
  - (c) the expert permutation (bind experts in a shuffled order with w shuffled the
    same way) gives the same result within bf16 noise.
  - (d) a timing split of gate-up vs down, via a build with down's runs coalesced into a
    dense test slab, to measure the cost of the 32-put fragmentation.

**M3: routing over the full chain.**
- Run 49 dispatches with the host router and compare per-layer `h_k` and `x_k` against
  an HF-equivalent numpy reference at positions 0..15 (cosine > 0.99 per layer, same
  top-8 sets for at least 95% of layer-tokens).
- Then add the LM dispatch and check top-5 logits and a 64-token greedy match against
  the reference (bf16-tolerant).

**M4: integration and performance.**
- Hook `MoeDecoder` into `npu_cli`/`server`.
- Measure ms/token at ctx 256/2k, then bytes/time and dispatch overhead, against the §6
  model.

**M5: optimizations, one at a time, each re-gated by M3:**
- C1 fused tail.
- C++ host loop.
- `@moeRW` to RTP if the channel was packet-shared and slow.
- j-outer down feed.
- Staircase windows for KV.
- C2 offset patching if sub-BO binding cost shows up in M0/M4.
