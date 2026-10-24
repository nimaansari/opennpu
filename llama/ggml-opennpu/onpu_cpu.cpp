// libonpu CPU side: one fused C++ call for the per-token work the NPU doesn't do and PyTorch runs as
// dozens of tiny ops. Today: the Gated DeltaNet decode step (Qwen3.5 / Qwen3.6 linear-attention
// layers): causal conv update + SiLU, q/k L2 norm, gated delta-rule state update, gated RMSNorm.
// All fp32. A small persistent thread pool splits the work by value head.
// SPDX-License-Identifier: MIT
#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <functional>
#include <immintrin.h>
#include <mutex>
#include <thread>
#include <vector>

namespace {

// parallel_for over [0, n): the caller takes part; workers sleep on a condition variable (no
// spinning: busy CPU cores slow the NPU, which shares the power budget)
struct Pool {
    std::vector<std::thread> th;
    std::mutex m, callers;          // callers: one run at a time (the engine calls from two threads)
    std::condition_variable cv, done_cv;
    std::function<void(int)> fn;
    std::atomic<int> next{0};
    int n = 0, gen = 0, busy = 0;
    bool stop = false;

    explicit Pool(int workers) {
        for (int i = 0; i < workers; i++)
            th.emplace_back([this] {
                int seen = 0;
                for (;;) {
                    {
                        std::unique_lock<std::mutex> lk(m);
                        cv.wait(lk, [&] { return stop || gen != seen; });
                        if (stop) return;
                        seen = gen;
                    }
                    work();
                    std::lock_guard<std::mutex> lk(m);
                    if (--busy == 0) done_cv.notify_one();
                }
            });
    }
    ~Pool() {
        { std::lock_guard<std::mutex> lk(m); stop = true; }
        cv.notify_all();
        for (auto & t : th) t.join();
    }
    void work() {
        for (int i; (i = next.fetch_add(1)) < n;) fn(i);
    }
    void run(int count, std::function<void(int)> f) {
        std::lock_guard<std::mutex> one(callers);
        if (th.empty() || count == 1) {
            for (int i = 0; i < count; i++) f(i);
            return;
        }
        {
            std::lock_guard<std::mutex> lk(m);
            fn = std::move(f);
            n = count;
            next = 0;
            busy = (int) th.size();
            gen++;
        }
        cv.notify_all();
        work();
        std::unique_lock<std::mutex> lk(m);
        done_cv.wait(lk, [&] { return busy == 0; });
    }
};

Pool & pool() {
    static Pool p([] {
        const char * e = std::getenv("OPENNPU_CPU_THREADS");
        int t = e ? std::atoi(e) : 4;
        return std::max(0, t - 1);
    }());
    return p;
}

// a second pool for work on the engine's main thread while the first serves the experts thread
Pool & pool2() {
    static Pool p([] {
        const char * e = std::getenv("OPENNPU_CPU_THREADS2");
        return std::max(0, (e ? std::atoi(e) : 3) - 1);
    }());
    return p;
}

inline float silu(float x) { return x / (1.f + std::exp(-x)); }
inline float dot(const float * a, const float * b, int n) {
    float s = 0.f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

}  // namespace

extern "C" {

// One decode token of a Gated DeltaNet layer (transformers' Qwen3_5*GatedDeltaNet, seq_len 1).
// x [H]: the layer input (for the b / a projections, Wb / Wa [Hv][H]); qkv [C] = in_proj_qkv(x),
// C = 2*Hk*Dk + Hv*Dv; z [Hv*Dv] = in_proj_z(x). conv_w [C][Kc], conv_state [C][L] (updated);
// S [Hv][Dk][Dv] recurrent state (updated); out [Hv*Dv]: gated-RMSNorm output, before out_proj.
void onpu_gdn_step(const float * x, int H, const float * qkv, const float * z, const float * Wb,
                   const float * Wa, const float * conv_w, int Kc, float * conv_state, int L,
                   const float * A_log, const float * dt_bias, const float * norm_w, float eps, float * S,
                   int Hk, int Hv, int Dk, int Dv, float * out) {
    const int key_dim = Hk * Dk, C = 2 * key_dim + Hv * Dv, rep = Hv / Hk;
    std::vector<float> co(C);
    // causal conv over the last Kc inputs, then the state shifts by one
    pool().run(8, [&](int part) {
        const int c0 = (int) ((int64_t) C * part / 8), c1 = (int) ((int64_t) C * (part + 1) / 8);
        for (int c = c0; c < c1; c++) {
            float * st = conv_state + (int64_t) c * L;
            const float * w = conv_w + (int64_t) c * Kc;
            float s = w[Kc - 1] * qkv[c];
            for (int j = 0; j < Kc - 1; j++) s += w[j] * st[L - (Kc - 1) + j];
            co[c] = silu(s);
            for (int j = 0; j + 1 < L; j++) st[j] = st[j + 1];
            st[L - 1] = qkv[c];
        }
    });
    pool().run(Hv, [&](int h) {
        const int kh = h / rep;
        std::vector<float> q(co.begin() + kh * Dk, co.begin() + (kh + 1) * Dk);
        std::vector<float> k(co.begin() + key_dim + kh * Dk, co.begin() + key_dim + (kh + 1) * Dk);
        const float * v = co.data() + 2 * key_dim + (int64_t) h * Dv;
        const float qn = 1.f / std::sqrt(dot(q.data(), q.data(), Dk) + 1e-6f) / std::sqrt((float) Dk);
        const float kn = 1.f / std::sqrt(dot(k.data(), k.data(), Dk) + 1e-6f);
        for (int i = 0; i < Dk; i++) {
            q[i] *= qn;
            k[i] *= kn;
        }
        const float b = dot(Wb + (int64_t) h * H, x, H), a = dot(Wa + (int64_t) h * H, x, H);
        const float beta = 1.f / (1.f + std::exp(-b));
        const float ad = a + dt_bias[h];
        const float sp = ad > 20.f ? ad : std::log1p(std::exp(ad));
        const float decay = std::exp(-std::exp(A_log[h]) * sp);
        float * Sh = S + (int64_t) h * Dk * Dv;
        std::vector<float> kv(Dv, 0.f), o(Dv, 0.f), delta(Dv);
        for (int i = 0; i < Dk; i++) {
            float * r = Sh + (int64_t) i * Dv;
            const float ki = k[i];
            for (int j = 0; j < Dv; j++) {
                r[j] *= decay;
                kv[j] += r[j] * ki;
            }
        }
        for (int j = 0; j < Dv; j++) delta[j] = (v[j] - kv[j]) * beta;
        for (int i = 0; i < Dk; i++) {
            float * r = Sh + (int64_t) i * Dv;
            const float ki = k[i], qi = q[i];
            for (int j = 0; j < Dv; j++) {
                r[j] += ki * delta[j];
                o[j] += r[j] * qi;
            }
        }
        const float ms = dot(o.data(), o.data(), Dv) / Dv;
        const float rn = 1.f / std::sqrt(ms + eps);
        const float * zh = z + (int64_t) h * Dv;
        float * oh = out + (int64_t) h * Dv;
        for (int j = 0; j < Dv; j++) oh[j] = norm_w[j] * (o[j] * rn) * silu(zh[j]);
    });
}

// One decode token of softmax attention (any GQA model): q [Hq][D]; K, V: Hkv heads of S rows of D
// values (head stride hs, row stride rs, in elements); mask: S additive floats or null; sinks [Hq]
// or null (gpt-oss: an extra per-head logit in the softmax that takes probability mass but has
// no value). out [Hq][D]. The cache is fp32 (onpu_attn_step) or bf16 (onpu_attn_step_bf16: half
// the bytes to read at long context).
}  // extern "C"

namespace {
inline float kv_load(const float * p) { return *p; }
inline float kv_load(const uint16_t * p) { uint32_t u = (uint32_t) *p << 16; float f; std::memcpy(&f, &u, 4); return f; }

template <typename KV>
void attn_step(const float * q, int Hq, int D, const KV * K, const KV * V, int Hkv, int64_t S, int64_t hs, int64_t rs,
               const float * mask, const float * sinks, float scale, float * out) {
    const int rep = Hq / Hkv;
    pool().run(Hq, [&](int h) {
        const float * qh = q + (int64_t) h * D;
        const KV * Kh = K + (int64_t) (h / rep) * hs;
        const KV * Vh = V + (int64_t) (h / rep) * hs;
        std::vector<float> p(S);
        float mx = sinks ? sinks[h] : -INFINITY;
        for (int64_t j = 0; j < S; j++) {
            const KV * kj = Kh + j * rs;
            float s = 0.f;
#pragma omp simd reduction(+ : s)
            for (int d = 0; d < D; d++) s += qh[d] * kv_load(kj + d);
            s = s * scale + (mask ? mask[j] : 0.f);
            p[j] = s;
            mx = std::max(mx, s);
        }
        float sum = sinks ? std::exp(sinks[h] - mx) : 0.f;
        for (int64_t j = 0; j < S; j++) {
            p[j] = std::exp(p[j] - mx);
            sum += p[j];
        }
        float * o = out + (int64_t) h * D;
        std::fill(o, o + D, 0.f);
        for (int64_t j = 0; j < S; j++) {
            const float pj = p[j] / sum;
            const KV * vj = Vh + j * rs;
            for (int d = 0; d < D; d++) o[d] += pj * kv_load(vj + d);
        }
    });
}
}  // namespace

extern "C" {

void onpu_attn_step(const float * q, int Hq, int D, const float * K, const float * V, int Hkv, int64_t S,
                    int64_t hs, int64_t rs, const float * mask, const float * sinks, float scale, float * out) {
    attn_step(q, Hq, D, K, V, Hkv, S, hs, rs, mask, sinks, scale, out);
}

void onpu_attn_step_bf16(const float * q, int Hq, int D, const uint16_t * K, const uint16_t * V, int Hkv, int64_t S,
                         int64_t hs, int64_t rs, const float * mask, const float * sinks, float scale, float * out) {
    attn_step(q, Hq, D, K, V, Hkv, S, hs, rs, mask, sinks, scale, out);
}

// The expert activation for rows r of h [rows][Ng] (expert ids[r]; gu_bias [E][Ng] or null) -> a
// [rows][Ng/2], split over the pool. act as in onpu_moe_ffn.
void onpu_moe_act(const float * h, int64_t rows, int64_t Ng, const int32_t * ids, const float * gu_bias, int act,
                  float alpha, float limit, float * a) {
    const int64_t I = Ng / 2;
    const int parts = (int) std::min<int64_t>(rows, 32);
    pool().run(parts, [&](int part) {
        std::vector<float> tmp(Ng);
        for (int64_t r = rows * part / parts; r < rows * (part + 1) / parts; r++) {
            const float * hr = h + r * Ng;
            if (gu_bias) {
                const float * b = gu_bias + (int64_t) ids[r] * Ng;
                for (int64_t i = 0; i < Ng; i++) tmp[i] = hr[i] + b[i];
                hr = tmp.data();
            }
            float * ar = a + r * I;
            if (act == 1)
                for (int64_t i = 0; i < I; i++) {
                    const float g = std::min(hr[2 * i], limit), u = std::max(-limit, std::min(hr[2 * i + 1], limit));
                    ar[i] = (u + 1.f) * g / (1.f + std::exp(-alpha * g));
                }
            else
                for (int64_t i = 0; i < I; i++) ar[i] = hr[i] / (1.f + std::exp(-hr[i])) * hr[I + i];
        }
    });
}

// y [T][N] = sum_j w[t][j] * (o[t][j] (+ bias[ids[t][j]])), split over the pool
void onpu_moe_sum(const float * o, int64_t T, int k, int64_t N, const int32_t * ids, const float * w,
                  const float * bias, float * y) {
    const int parts = (int) std::min<int64_t>(T, 32);
    pool().run(parts, [&](int part) {
        for (int64_t t = T * part / parts; t < T * (part + 1) / parts; t++) {
            float * yt = y + t * N;
            std::fill(yt, yt + N, 0.f);
            for (int j = 0; j < k; j++) {
                const int64_t r = t * k + j;
                const float wr = w[r], * orow = o + r * N;
                const float * b = bias ? bias + (int64_t) ids[r] * N : nullptr;
                for (int64_t i = 0; i < N; i++) yt[i] += wr * (orow[i] + (b ? b[i] : 0.f));
            }
        }
    });
}

// y [N] = W . x for int8 W [N][K] with a scale per row (y[n] = s[n] * sum_k W[n][k] x[k]),
// rows split over pool2. For small matrices the CPU keeps (a MoE's shared expert): 4x fewer bytes than fp32.
void onpu_gemv_i8(const int8_t * W, const float * s, int64_t N, int64_t K, const float * x, float * y) {
    const int parts = (int) std::min<int64_t>(8, std::max<int64_t>(1, N / 128));
    pool2().run(parts, [&](int part) {   // pool2: the experts thread uses pool() meanwhile
        for (int64_t n = N * part / parts; n < N * (part + 1) / parts; n++) {
            const int8_t * w = W + n * K;
            float acc = 0.f;
            int64_t k = 0;
#if defined(__AVX512F__)
            __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
            for (; k + 32 <= K; k += 32) {
                const __m512 w0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *) (w + k))));
                const __m512 w1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *) (w + k + 16))));
                a0 = _mm512_fmadd_ps(w0, _mm512_loadu_ps(x + k), a0);
                a1 = _mm512_fmadd_ps(w1, _mm512_loadu_ps(x + k + 16), a1);
            }
            acc = _mm512_reduce_add_ps(_mm512_add_ps(a0, a1));
#endif
            for (; k < K; k++) acc += (float) w[k] * x[k];
            y[n] = acc * s[n];
        }
    });
}

}  // extern "C"
