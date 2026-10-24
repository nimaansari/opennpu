> **Status (2026-10-12): shipped differently.** Upstream MLIR-AIR merged a shared fused prefill
> (PR #2068, 2026-10-07): int4 GEMMs dequantizing on the cores and attention on the NPU, ~10x the
> option-B estimate below and with no bfp16 weight copy. OpenNPU uses it (`opennpu/native/prefill.py`):
> our own `_requant_q4k` weights are packed for it, its K/V are written into the decoder's regions.
> Measured: Qwen3-8B 238 tok 0.8 s, 1,900 tok 5.3 s (was 22 s / ~3 min); 12B/14B ~235 tok 1.1-1.3 s.
> Option B was also built and measured (238 tok in 5.1 s) and dropped. The MoE engine still prefills
> per token. The analysis below is kept for the record.

# Batched prompt prefill for OpenNPU's fused engines: design

Abbreviations: RT = `opennpu/native/runtime.py`, MOE = `opennpu/native/moe_runtime.py`, FD = `<mlir-air checkout>/programming_examples/fused_decode/fused_decode.py`, UP = `.../llms/qwen3_8b_q4nx/`. "est." marks an estimate. Nothing was run on the NPU.

## 0. Facts that drive the design

- **KV layout** (FD:1107-1181, append FD:2717-2735). The layout is region-major, not the `[tok][K|V]` that the stale comment at FD:1584 describes. Each layer slab is `LREG = ATTN_MAXL*KVSZ_TOK` bf16, laid out as `[K_g0 | K_g1 | V_g0 | V_g1]`. Each region is `ATTN_MAXL*REGION_W` long, and position p sits at `p*REGION_W`. The groups are the attention columns (3 and 4), and each holds `REGION_W/DH` consecutive KV heads in HF order.
  - Qwen3-8B: 8 KV heads, NGRP=2, REGION_W=512 (heads 0-3 | 4-7), KVSZ_TOK=2048. That is 4 KiB/token/layer, or 144 KiB/token over 36 layers.
  - Qwen3-30B-A3B: 4 KV heads, REGION_W=256, 2 KiB/token/layer, 96 KiB/token over 48 layers. The slabs sit in `kv_parent` at `k*LREG`, plus one zero scratch slab (MOE `kv_parent`).
- **What the device stores** (`kernels/rope.cc:144-200`). K is stored after qk-norm and RoPE (half-split/NEOX), rounded to bf16. V is the raw projection output in bf16. The 1/sqrt(DH) scale is applied later, at score time (`attn_qk.cc:217`).
- **The upstream driver already does this conversion.** `UP/qwen3_8b_q4nx_inference.py:333-354` (`seed_kv`) takes `fk/fv[L,P,kv_dim]` (roped K, raw V, natural head order) and slices `[:, gi*RW:(gi+1)*RW]` into the regions. That is the exact transform we need.
- **The fused decode is structurally M=1.**
  - `proj_qmm_acc` is a GEMV over a single resident `x_full` (`kernels/proj_qmm.cc:47-57`).
  - The rms core holds seven K-wide L1 buffers, which is already 63.5 of 64 KiB at K=4096 (FD:1038-1052, 1096).
  - There is one append slot, `APPEND_OFF=(ATTN_L-1)*KVSZ_TOK` (FD:1114), one RoPE position per dispatch (RT:297-302), and one runtime L for the causal mask.
- **Current cost.** RT:332-334 and MOE:298-300 feed every prompt token through `step()`. That is about 12 tok/s on 8B (≈43 s for 512 tokens) and 7.7 tok/s on 30B-A3B (≈67 s for 512 tokens).
- **CPU measured on this box** (8 threads, torch, while other load was running):
  - bf16 GEMM: 840-850 GFLOPS. fp32 GEMM: 144-351 GFLOPS.
  - Causal SDPA, 32 heads × 128: 11.5 ms/layer at P=512 and 141 ms/layer at P=2048 (fp32). bf16 SDPA is slower: 27.5 and 221 ms.
  - Host dequant of one 12288×4096 Q4 matrix to bf16 takes 94 ms, so ~13 s for the whole 8B model. **Dequantizing per prompt is not viable; prefill weights must stay resident.**
- **Python environments.** `import iron` fails in airenv, and `air` is missing from the IRON venv. `flm.GEMM` and `NativeDecoder` therefore cannot share a process today.

## 1. Options ranked

| | Speed-up 100-2000 tok (est.) | Effort | Extra RAM | Risk |
|---|---|---|---|---|
| **B. Hybrid: NPU `flm.GEMM` for the projections + CPU attention/norms, KV written into the fused layout** (dense) | 8B: 512 tok 43 s → ~2-4 s (10-20×); 2048 tok → ~10-17 s | ~1-2 weeks, host Python only | 8B: +7.8 GB of bfp16 weights (6.95 B params × 1.125 B) | Medium: env merge, second hw_context, bfp16 drift |
| **D1. iGPU llama.cpp Vulkan prefill + KV transplant** (MoE) | 30B-A3B: **measured pp512 = 261.6 tok/s** (own measurement), so 512 tok goes 67 s → ~2 s (~30×) | ~1 week (ctypes `llama_state_seq_get_data`, format parse) | +17.3 GiB of GGUF in GTT, which does not fit beside ~19 GB of MoE BOs in 46 GB unless mmap/host-ptr import avoids the copy | High: RAM, state format tied to the build (b11368), Q4_K vs Q4NX weights |
| **F. CPU bf16 prefill** (reference.py batched, HF bf16 memmap) | 8B: 512 tok ~9 s (5×); 2048 ~40 s | 2-3 days | 16 GB of page cache (8B) | Low; also the correctness scaffold for B |
| **A. Upstream MLIR-AIR prefill** | 8B only. Fixed CTX=2048 costs **7.72 s regardless of prompt length** (UP/README.md:41), which is slower than today below ~95 tokens. A CTX=512 build would take ~1.9 s (est., scaled from the qwen3_4b 1482/6047 ms) but needs a second resident bf16 weight set | Medium | ~28 GB RSS: bf16 per-layer static BOs ≈ 13.9 GB + 1.34 GB lm-head + load peak (UP prefill.py:184-214, `qwen3_4b_prefill.py:763`), plus 80 s load | High: OOM; FLM `model.q4nx` weights differ from our `_requant_q4k`; no 14B or MoE path |
| **C. Chunked M=4-16 fused decode** | ~4-6× at M=8 (est.: weights read once per chunk, but GEMV dequant/MAC efficiency is unknown) | Months; FD is 5,083 lines of deadlock-sensitive DMA choreography | ~0 | Very high |
| D2. CPU llama.cpp (ggml int4) for MoE | ~29 tok/s (third-party, 35B-A3B), ~4× | ~1 week | page cache only | Medium |

What C would need to change:
- x as `[M,K]` streamed col-blocks (the "reference-streaming" `proj_qmm` variant exists), with `y_acc` of 32×M.
- An rms core with M rows. It only fits in L1 with the lean/NORM_FOLD body.
- M RoPE LUT slabs per layer.
- An append of M·REGION_W per group, which is one contiguous BD, so this part is easy.
- Attention with per-query causal limits. Today the kernel masks on a single scalar L, and the extra queries would mean M× more Q heads in L1.

This is not worth doing before B.

**Recommendation:** B for dense models (8B/14B) and F as step 0, since F is B with CPU matmuls. For 30B-A3B, B does not work: the experts would need 32.6 GB of bfp16 weights (48×128×3×2048×768 params). Use D1 if the RAM can be made to fit, otherwise D2. Keep the NPU per-token path as the fallback.

## 2. Option B in detail (dense, e.g. Qwen3-8B)

### Per layer k, prompt tokens [p0, p0+n) on top of existing KV [0, p0)

1. **Input norm.** `h = rmsnorm(x)·n_in`, in fp32 on the CPU. Skip the `·n_in` when `fd.NORM_FOLD`: Qwen3-8B has NORM_FOLD=False because 4096·14+6144 = 63,488 ≤ 65,536 (FD:1049); 14B has it True.
2. **QKV projection.** `qkv = h @ Wqkv` on the NPU with `flm.GEMM`. Round the output to bf16, which is the dtype the device carries into rope.
3. **qk-norm, then RoPE, per head** (Q and K), to mirror `rope.cc`:
   - `bf16( (k·w_knorm) · rsqrt(mean(k²)+1e-6) )`.
   - Half-split rotation with cos/sin computed in float64 and **rounded to bf16** (RT:297-299), using `_inv_freq` with llama3/linear scaling (RT:173-187).
   - fp32 math, bf16 output.
   - For Qwen2.5, add the q/k/v bias before RoPE instead (rope.cc HAS_QKV_BIAS).
4. **Write K and V into the KV BO.** K is the bf16 roped key, V the bf16 raw value. They go straight into the mapped `self.kvc` BO (host_only, RT:279):
   ```
   base = k*LREG; RS = ATTN_MAXL*REGION_W; G = REGION_W//DH
   for gi in range(NGRP):
     kv[base+gi*RS+p0*RW : base+gi*RS+(p0+n)*RW]       = K[:, gi*G:(gi+1)*G, :].reshape(n, RW)
     kv[base+(NGRP+gi)*RS+p0*RW : ...+(p0+n)*RW]       = V[:, gi*G:(gi+1)*G, :].reshape(n, RW)
   ```
   Sync to device once at the end. The cost is 288 MiB at P=2048, ~50 ms (est.).
5. **Attention.** Causal GQA attention on the CPU (torch SDPA, fp32) over K/V for [0, p0+n). When p0>0, `kvc.sync(FROM)` first and read the earlier positions back from the regions.
6. **Rest of the layer.**
   - `x += attn @ Wo` (NPU).
   - `h2 = rmsnorm(x)·n_post`.
   - `g, u` (NPU), `silu(g)·u`, `x += · @ Wdown` (NPU).
   - Round the residual stream to bf16 after each add, because the device keeps x in bf16 (`x_bo`).
   - The last layer's FFN can be skipped.

### Plug points

- **`NativeDecoder.prefill(ids)`.** A new method: run the batched path for `ids` at `self.pos`, then `self.pos += len(ids)`.
- **`generate`** (RT:332-334). If `len(ids)-1 ≥ 32`, call `self.prefill(ids[:-1])`. Then always call `lg = self.step(ids[-1])`, so the logits come from the real decode and the prefill needs no LM head. `step()` re-patches the insts and RoPE for `pos` itself (RT:312-315), so no extra state is needed.
- **Server.** `KVSessions` wraps only `dec.step` to track `_fed` (`native/server.py:400-404`). Wrap `dec.prefill` the same way (`_fed.extend(ids)`). `_kv_prefix` already rewinds `dec.pos` (server.py:430-436), and the batched path must honour p0>0.
- **Weights.**
  - Rebuild exactly as `build_weights` does (RT:127-143): the same `_requant_q4k` of the same folded matrices, dequantized as `q*s+m`.
  - Pack once with `FLMEngine.weight` (`opennpu/engine.py:256-269`) and cache the result to disk next to `W.bin`.
  - Upload lazily on first use.
- **Environment.** Either install amd/iron + mlir_aie 1.4 into airenv (MLIR python packages could conflict), or run the GEMMs in a sidecar process over shared memory. The decoder already holds several hw_contexts per process (`decode_staircase.open_windows`), so a second context for the GEMM xclbin should be acceptable (est.).

### Estimates for Qwen3-8B

Projection FLOPs are 13.9 GFLOP/token (36 layers × 386 MFLOP). GEMM throughput is taken as 2.6 TFLOPS (measured at M=256) to 6.6 TFLOPS (what the upstream AIR GEMMs reach at M=2048: 4.3 s for 28.5 TFLOP, README:51).

| P | GEMM | CPU attention | Norms/RoPE/SwiGLU + glue | Total (est.) | Today |
|---|---|---|---|---|---|
| 512 | 1.1-2.7 s | 0.4 s | ~0.3 s | **~2-3.5 s** | ~43 s |
| 2048 | 4.3-11 s | 5.1 s | ~1 s | **~10-17 s** | >170 s |

- Peak extra RAM is ~0.5 GB of activations, plus 7.8 GB of resident bfp16 weights. Without them, host dequant would cost 13 s per prompt.
- At 2048 tokens, CPU attention is about a third of the total. The next lever is to run it on the NPU (upstream flash_attn ELF: 1.65 s per 36 layers at 2048) or to overlap it with the GEMMs across token chunks.

### Qwen3-30B-A3B under B

- The attention projections (0.9 B params, 1.0 GB bfp16) could be batched.
- The experts (29 B params) cannot: bfp16 copies would not fit, and CPU dequant runs at ~0.5 G params/s.
- So under B the MoE stays at ~130 ms/token. D1/D2 are the only cheap wins for it. With D1, convert the llama.cpp f16 K (post-k_norm, NEOX-roped) and V rows to bf16 and place them with the same region formula: REGION_W=256, slab k at `k*LREG` in `kv_parent`.
- KV size is 48 MiB at 512 tokens and 192 MiB at 2048.

## 3. Validation plan

1. **CPU only.** Extend `reference.forward` (`native/reference.py:31-91`) to return the per-layer roped K and raw V. Check that the batched emulation matches it with `quant=True` (cosine > 0.9999 with CPU matmuls).
2. **Device KV A/B.** For P in {17, 100, 511, 1500}:
   - Run the token-by-token path, then `kvc.sync(FROM)` and save the slab.
   - `reset()`, run `prefill(ids[:-1])`, and compare the slabs region by region over [0, P-1).
   - Pass: per-layer cosine ≥ 0.999 for layer 0, and ≥ 0.99 for the last layer.
   - Bytes beyond P must be untouched (zero).
   - First measure the noise floor: device versus `reference.forward(quant=True)`. Then require the batched path to stay within 2× that floor.
3. **Logits.** Compare `step(ids[-1])` between the two paths. Pass: cosine ≥ 0.995 and the same argmax on ≥ 95% of 50 prompts.
4. **Generation gate.** 32-token greedy generation with the upstream gate: top-5 inclusion until the first divergence.
5. **Continuation.** `prefill(A)` then `prefill(B)` must match `prefill(A+B)`. Also test with a session snapshot taken and restored between turns.

## 4. Pitfalls

- **bfp16 versus Q4NX drift.** `flm.GEMM` quantizes B to bfp16ebs8 (8 values share one exponent, 8-bit mantissa) and runs a bfp16 multiply on bf16 activations. That adds ~2⁻⁸ relative error per GEMM on top of Q4NX.
  - Source the prefill weights from the *dequantized Q4NX*, never from HF bf16, so both paths see the same model.
  - If the drift test fails, move the QKV GEMM (13% of FLOPs) to CPU bf16.
  - Option F (HF bf16) seeds "cleaner" KV than the decode would compute. It is usually harmless but must pass the same gate.
- **NORM_FOLD.** Q4(W·n) ≠ Q4(W)·n. Quantize the folded matrix and run a weightless RMSNorm, exactly as RT:135-141 does. The MoE build never folds.
- **qk-norm.** It is a per-head RMSNorm over DH, applied to both Q and K before RoPE. K is stored post-norm and post-RoPE, and the device rounds to bf16 in between (rope.cc:150-200).
- **RoPE.** Use a bf16 LUT, NEOX half-split, scaling from `_inv_freq`, and partial rotary (`ROPE_DIM`) for Phi-4. The per-layer RoPE payload in `r_bo` (RT:300-302; MOE `_write_rms`) is rewritten by `step()` for its own position, so the prefill must not depend on it or leave assumptions about it.
- **ATTN_MAXL and the staircase.**
  - The region stride is `ATTN_MAXL*REGION_W` of the *open* window. RT uses a single window today (`decode_staircase.resolve_windows`); multi-window builds would need a KV re-space.
  - Enforce `pos+P < ATTN_MAXL`.
  - Ignore the stale `[ATTN_MAXL][K|V]` comment at FD:1584.
- **Host BO coherency.** Do `sync(FROM)` before reading existing KV for p0>0, and a single `sync(TO)` after writing.
- **Upstream A specifics.** Its ELF cache and static BOs are per `seq_len` (UP prefill.py:105-112), so every extra CTX bucket duplicates ~15 GB. Its weights are FLM's own Q4NX, so seeding our decode with them is a cross-quantization mismatch.

## Revisions after review (2026-10-12)

1. **One environment, no sidecar.** The "IRON can't be imported in the engine env" blocker is stale.
   Since install.sh v0.3, `~/.local/share/opennpu/env` contains torch, IRON flm.GEMM, mlir_aie dev53,
   llvm-aie 0917 and mlir_air, and fused-decode output has been verified token-identical on those pins.
   flm.GEMM and NativeDecoder can therefore share one process; FLMEngine already compiles into
   `$OPENNPU_CACHE/iron`.
2. **Hardware-context switches.** flm.GEMM and the fused decode each occupy all 8 columns, and an
   earlier measurement put a switch between two whole-array xclbins at about 2.7 ms. So: batch per
   phase (all of a prompt's GEMMs, then decode) and never alternate per layer. **Measure the switch
   cost first**; that number decides whether any per-layer interleave is viable.
3. **NPU-first.** The project's constraint is that inference runs on the NPU, with the CPU and iGPU
   as comparison only.
   - CPU attention inside option B is a stepping stone and is documented as such.
   - Options D1 (iGPU llama.cpp) and D2 (CPU prefill) for the MoE are opt-in only, never the default.
   - The honest default for Qwen3-30B-A3B is per-token prefill (plus the per-session prompt cache)
     until a fused M>1 path exists.
4. **RAM.** The prefill weights (+7.8 GB bfp16 for 8B, about +13 GB for 14B) are:
   - **lazy**: built and loaded on first long prompt;
   - **budgeted**: accounted in the model pool's RAM budget and LRU;
   - **droppable**: under memory pressure they are released, and prefill falls back to per-token;
   - **never auto-enabled** for 14B without a RAM check.
5. **Validation gate.** In addition to §3, the existing `opennpu/native/verify.py` top-5 check at about
   250 tokens (the check the native engines were validated with) must pass with batched prefill on.
