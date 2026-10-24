// Copyright (C) 2026, Advanced Micro Devices, Inc. (dequant from MLIR-AIR's gemv_q4.cc, MIT)
// and the OpenNPU authors.
// SPDX-License-Identifier: MIT
//
// int4 GEMM for 8 tokens at once: Y[8][n] = sum_k X[8][k] W[n][k], on the same packed weights as
// gemv_any.cc (so a model needs one copy of its weights for decode and prompt). Each 8x8 (k, n)
// weight tile is dequantized once (exactly as in gemv_any.cc) and multiplied into all 8 tokens
// with one bf16 8x8x8 mmul, so a weight pass serves 8 prompt tokens.
//
// Packet (host packing, gemv_device.pack_block): q[GV_N/8 nb][GV_K/8 kb][8 kk][8 nn / 2],
// scale[GV_K/32][GV_N], min[GV_K/32][GV_N] (bf16); w = q * scale + min. -DGV_SYM (q4s): no mins,
// w = (q - 8) * scale.
// X (host layout): [K/8 kt][8 tok][8 kk] bf16, so one 8-token x 8-k tile is 64 contiguous values.
// acc: [GV_N/8 nb][8 tok][8 nn] f32. Output tile j: [nb][tok][nn] bf16.

#include <aie_api/aie.hpp>
#include <cstdint>

#ifndef GV_N
#define GV_N 64
#endif
#ifndef GV_K
#define GV_K 256
#endif
#define GV_GROUP 32
#define GM_T 8

extern "C" {

// Fused expert FFN, the down phase (kc >> 8 == 1): xblk holds the gate_up phase's output for 8 tokens
// in this kernel's input layout, [gate rows / 8][tok][8] then [up rows / 8][tok][8] with n = (kc & 0xff)
// * GV_K rows each; it becomes SiLU(gate) * up in place (the first n / 8 blocks). kc >> 8 == 0: nothing.
void gm_prep(bfloat16 *__restrict xblk, int32_t kc) {
  if (!(kc >> 8))
    return;
  const unsigned n = (unsigned)(kc & 0xff) * GV_K * GM_T;   // values per half
  const aie::vector<bfloat16, 16> half = aie::broadcast<bfloat16, 16>(0.5f);
  const aie::vector<bfloat16, 16> one = aie::broadcast<bfloat16, 16>(1.0f);
  for (unsigned k = 0; k < n; k += 16) {
    aie::vector<bfloat16, 16> g = aie::load_v<16>(xblk + k);
    aie::vector<bfloat16, 16> u = aie::load_v<16>(xblk + n + k);
    aie::vector<bfloat16, 16> gh = aie::mul(g, half);
    aie::accum<accfloat, 16> t;
    t.from_vector(gh);
    aie::vector<bfloat16, 16> th = aie::tanh<bfloat16>(t.to_vector<float>());
    aie::vector<bfloat16, 16> sg = aie::mul(half, aie::add(one, th));
    aie::vector<bfloat16, 16> silu = aie::mul(g, sg);
    aie::store_v(xblk + k, aie::vector<bfloat16, 16>(aie::mul(silu, u)));
  }
}

void gm_zero(float *acc) {
  for (unsigned i = 0; i < GV_N * GM_T; i += 64)
    aie::store_v(acc + i, aie::zeros<float, 64>());
}

// acc += X[:, kc*GV_K .. +GV_K) . dq(packet)^T
void gm_acc(uint8_t *__restrict packed, bfloat16 *__restrict xblk,
            float *__restrict acc, int32_t kc) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  constexpr unsigned LANES = 64;
  constexpr unsigned Q_BYTES = GV_N * GV_K / 2;
  constexpr unsigned NG = GV_K / GV_GROUP;
  constexpr unsigned TILES_PER_G = GV_GROUP / 8;
  using MM = aie::mmul<8, 8, 8, bfloat16, bfloat16>;
  const bfloat16 *sc = (const bfloat16 *)(packed + Q_BYTES);
#ifndef GV_SYM
  const bfloat16 *mn = sc + NG * GV_N;
#endif
  const aie::vector<uint8, LANES> bias = aie::broadcast<uint8, LANES>(0x43);
  const bfloat16 *xk = xblk + kc * (GV_K / 8) * LANES;
  for (unsigned nb = 0; nb < GV_N / 8; nb++) {
    const uint4 *q = (const uint4 *)packed + nb * (GV_K / 8) * LANES / 2;
    const bfloat16 *xt = xk;
    MM m(aie::accum<accfloat, 64>(aie::load_v<64>(acc + nb * LANES)));
    for (unsigned g = 0; g < NG; g++) {
      aie::vector<bfloat16, 8> s8 = aie::load_v<8>(sc + g * GV_N + nb * 8);
      aie::vector<bfloat16, 16> s16 = aie::concat(s8, s8);
      aie::vector<bfloat16, 32> s32 = aie::concat(s16, s16);
      aie::vector<bfloat16, LANES> sv = aie::concat(s32, s32);
#ifdef GV_SYM
      // symmetric (q4s): w = (q - 8) * scale = (128 + q) * scale - 136 * scale
      aie::accum<accfloat, LANES> base = aie::mul(sv, bfloat16(-136.0f));
#else
      aie::vector<bfloat16, 8> m8 = aie::load_v<8>(mn + g * GV_N + nb * 8);
      aie::vector<bfloat16, 16> m16 = aie::concat(m8, m8);
      aie::vector<bfloat16, 32> m32 = aie::concat(m16, m16);
      aie::vector<bfloat16, LANES> mv = aie::concat(m32, m32);
      aie::accum<accfloat, LANES> base;
      base.from_vector(mv);
      base = aie::msc(base, sv, bfloat16(128.0f));
#endif
      for (unsigned t = 0; t < TILES_PER_G; t++)
        chess_prepare_for_pipelining chess_loop_range(TILES_PER_G, TILES_PER_G) {
          aie::vector<uint4, LANES> qr = aie::load_v<LANES>(q);
          q += LANES / 2;
          aie::vector<uint8, LANES> qb = aie::unpack(qr);
          auto z = aie::interleave_zip(qb, bias, 1);
          aie::vector<bfloat16, LANES> w =
              aie::vector_cast<bfloat16>(aie::concat(z.first, z.second));
          // the dequantized tile, [8 kk][8 nn]: B of an (8 tok x 8 k) . (8 k x 8 n) product
          aie::vector<bfloat16, LANES> wd =
              aie::mac(base, w, sv).template to_vector<bfloat16>();
          m.mac(aie::load_v<64>(xt), wd);
          xt += LANES;
        }
    }
    aie::store_v(acc + nb * LANES, m.template to_vector<float>());
  }
}

// out[j * GV_N * GM_T + i] (bf16) = acc[i]   ([nb][tok][nn])
void gm_flush(float *__restrict acc, bfloat16 *__restrict out_base, int32_t j) {
  bfloat16 *out = out_base + j * GV_N * GM_T;
  for (unsigned i = 0; i < GV_N * GM_T; i += 64) {
    aie::accum<accfloat, 64> a;
    a.from_vector(aie::load_v<64>(acc + i));
    aie::store_v(out + i, a.template to_vector<bfloat16>());
  }
}

// A FLUSH pair of row tiles as one run of 2 * kc packets (kc may be odd: the pair's packet count stays
// even, so the ping-pong buffers stay in step). acc holds both tiles' accumulators.
void gm_zero2(float *acc) {
  for (unsigned i = 0; i < 2 * GV_N * GM_T; i += 64)
    aie::store_v(acc + i, aie::zeros<float, 64>());
}

// packet q of the pair (0 .. 2 kc): tile q / kc, K packet q % kc (kc & 0xff: the high bits are flags)
void gm_acc2(uint8_t *__restrict packed, bfloat16 *__restrict xblk, float *__restrict acc, int32_t q, int32_t kc) {
  const int32_t n = kc & 0xff;
  const int32_t j = q >= n ? 1 : 0;
  gm_acc(packed, xblk, acc + j * (GV_N * GM_T), q - j * n);
}

void gm_flush2(float *__restrict acc, bfloat16 *__restrict out_base) {
  gm_flush(acc, out_base, 0);
  gm_flush(acc + GV_N * GM_T, out_base, 1);
}

} // extern "C"
