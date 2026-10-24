// OpenNPU NPU core: geometry, int4 packing and the GEMV / 8-token devices (XRT 2.25 C++).
// Shared by the ggml backend (ggml-opennpu.cpp) and OpenNPU's own engines (libonpu, onpu_api.cpp).
// SPDX-License-Identifier: MIT
#pragma once

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <vector>

#ifndef ONPU_ASSERT
#define ONPU_ASSERT(c) do { if (!(c)) throw std::runtime_error("opennpu: " #c); } while (0)
#endif
#ifndef ONPU_WARN
#define ONPU_WARN(...) fprintf(stderr, __VA_ARGS__)
#endif

namespace onpu {

constexpr int NC = 8, NR = 4, GN = 64, GK = 256, KMAX = 2048, FLUSH = 2;
constexpr int PKT = GN * GK / 2 + 2 * 2 * (GK / 32) * GN;           // 10240 bytes
constexpr int PKT_S = GN * GK / 2 + 2 * (GK / 32) * GN;             // 9216: q4s (symmetric, no mins)
constexpr int BT = 8;                                               // tokens per pass, batched device
constexpr int64_t DELTA = 1 << 16;
constexpr int64_t MIN_PARAMS = 1 << 20;                             // smaller matrices: CPU

inline int out_tiles(int tiles) { return tiles + tiles % FLUSH; }   // an odd last tile: zero partner
size_t col_bytes(int tiles, int kc, bool sym = false) { return (size_t) tiles * kc * NR * (sym ? PKT_S : PKT); }

// ------------------------------------------------------------------ bf16
inline uint16_t f2bf(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffff) > 0x7f800000) return (uint16_t) ((u >> 16) | 64);   // NaN
    u += 0x7fff + ((u >> 16) & 1);
    return (uint16_t) (u >> 16);
}
inline float bf2f(uint16_t h) {
    uint32_t u = (uint32_t) h << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// ------------------------------------------------------------------ geometry
struct Mat {
    int64_t N = 0, K = 0;
    int ncols = 0, R = 0, S = 0, tiles = 0, kc = 0;
    int64_t ks = 0, rows_col = 0;
    size_t cb = 0, nbytes = 0;
    bool ok = false, sym = false;   // sym: the q4s weight format

    Mat() = default;
    Mat(int64_t n, int64_t k, int nc, bool sy = false) : N(n), K(k), ncols(nc), sym(sy) {
        double best = -1;
        for (int s = 1; s <= nc; s++) {
            if (nc % s) continue;
            int kcs = (int) ((K + (int64_t) s * GK - 1) / ((int64_t) s * GK));   // odd allowed
            if (kcs > KMAX / GK) continue;
            int r = nc / s;
            int tl = (int) ((N + (int64_t) r * NR * GN - 1) / ((int64_t) r * NR * GN));
            if ((tl * kcs) % 2) {                     // tiles * kc even keeps the ping-pong in step
                if (kcs + 1 <= KMAX / GK && (int64_t) tl * (kcs + 1) < (int64_t) (tl + 1) * kcs) kcs++;
                else tl++;
            }
            double cost = (double) r * s * tl * kcs * (1 + 0.03 * (s - 1));
            if (best < 0 || cost < best) {
                best = cost; R = r; S = s; tiles = tl; kc = kcs;
            }
        }
        if (best < 0) return;
        ks = (int64_t) kc * GK;
        rows_col = (int64_t) NR * tiles * GN;
        cb = col_bytes(tiles, kc, sym);
        nbytes = (size_t) R * S * cb;
        ok = true;
    }
};

// ---- 4-bit quantizers, per 32 values (zero-padded past the matrix)
// q4nx: w ~= q * s + m, q in 0..15. The grid is stretched 21 ways around min..max and, for each, (s, m)
// refit by least squares (as llama.cpp's K-quants); the pair with the least squared error wins.
// s and m are rounded to bf16 (as stored) before the final codes are chosen.
inline void quant_asym32(const float * x, uint8_t * q, uint16_t & s_bf, uint16_t & m_bf) {
    float lo = x[0], hi = x[0];
    for (int i = 1; i < 32; i++) { lo = std::min(lo, x[i]); hi = std::max(hi, x[i]); }
    const float range = hi - lo;
    if (!(range > 0)) {
        s_bf = f2bf(1.0f); m_bf = f2bf(lo);
        std::memset(q, 0, 32);
        return;
    }
    float best_e = INFINITY, best_s = range / 15, best_m = lo;
    for (int st = 0; st <= 20; st++) {
        const float isc = (-1.0f + 0.1f * st + 15.0f) / range;
        float sl = 0, sll = 0, sx = 0, slx = 0;
        uint8_t L[32];
        for (int i = 0; i < 32; i++) {
            const float l = std::min(15.0f, std::max(0.0f, std::nearbyint(isc * (x[i] - lo))));
            L[i] = (uint8_t) l; sl += l; sll += l * l; sx += x[i]; slx += l * x[i];
        }
        const float D = 32 * sll - sl * sl;
        float sc = range / 15, mn = lo;
        if (D > 0) { sc = (32 * slx - sl * sx) / D; mn = (sll * sx - sl * slx) / D; }
        sc = bf2f(f2bf(sc)); mn = bf2f(f2bf(mn));
        if (!(sc > 0)) continue;
        float e = 0;
        for (int i = 0; i < 32; i++) {
            const float l = std::min(15.0f, std::max(0.0f, std::nearbyint((x[i] - mn) / sc)));
            const float d = l * sc + mn - x[i];
            e += d * d;
        }
        if (e < best_e) { best_e = e; best_s = sc; best_m = mn; }
    }
    best_s = bf2f(f2bf(best_s)); best_m = bf2f(f2bf(best_m));
    for (int i = 0; i < 32; i++)
        q[i] = (uint8_t) std::min(15.0f, std::max(0.0f, std::nearbyint((x[i] - best_m) / best_s)));
    s_bf = f2bf(best_s); m_bf = f2bf(best_m);
}

// q4s: w ~= (q - 8) * d. 16 scales around absmax / 8 (both signs), each refit by least squares
// (d = sum(qx) / sum(qq)) and rounded to bf16; the least squared error wins.
inline void quant_sym32(const float * x, uint8_t * q, uint16_t & d_bf) {
    float mx = 0;
    for (int i = 0; i < 32; i++) if (std::fabs(x[i]) > std::fabs(mx)) mx = x[i];
    if (mx == 0) { d_bf = f2bf(1.0f); std::memset(q, 8, 32); return; }
    float best_e = INFINITY, best_d = mx / -8.0f;
    for (int c = 0; c < 16; c++) {
        const float dv = c < 8 ? -9.0f + 2.5f * c / 7 : 6.5f + 1.5f * (c - 8) / 7;
        const float d0 = mx / dv;
        float sqq = 0, sqx = 0;
        for (int i = 0; i < 32; i++) {
            const float v = std::min(7.0f, std::max(-8.0f, std::nearbyint(x[i] / d0)));
            sqq += v * v; sqx += v * x[i];
        }
        float d = sqq > 0 ? sqx / sqq : d0;
        d = bf2f(f2bf(d));
        if (d == 0) continue;
        float e = 0;
        for (int i = 0; i < 32; i++) {
            const float v = std::min(7.0f, std::max(-8.0f, std::nearbyint(x[i] / d)));
            const float r = v * d - x[i];
            e += r * r;
        }
        if (e < best_e) { best_e = e; best_d = d; }
    }
    for (int i = 0; i < 32; i++)
        q[i] = (uint8_t) (std::min(7.0f, std::max(-8.0f, std::nearbyint(x[i] / best_d))) + 8);
    d_bf = f2bf(best_d);
}

// codes for the 32 weights at row gr, columns [gc, gc + 32) (zero past column k1): q (32 codes), the
// group's bf16 scale and (q4nx) min. pack_block calls it per group; the default quantizes floats,
// the ggml backend also transcodes GGUF blocks with it.
using QuantFn = std::function<void(int64_t gr, int64_t gc, int64_t k1, uint8_t * q, uint16_t & s, uint16_t & m)>;

// Pack rows [r0, r1) x cols [k0, k1) into one column block (NR*tiles*GN rows x kc*GK cols,
// zero-padded), stream order [tile][kc][core row][packet]. sym: q4s; else q4nx.
void pack_block_fn(int64_t r0, int64_t r1, int64_t k0, int64_t k1, int tiles, int kc, uint8_t * out, bool sym,
                   const QuantFn & quant) {
    const int64_t R = (int64_t) NR * tiles * GN, Kp = (int64_t) kc * GK;
    std::vector<uint8_t> q((size_t) R * Kp, sym ? 8 : 0);
    std::vector<uint16_t> sc((size_t) R * (Kp / 32), 0), mn((size_t) R * (Kp / 32), 0);
    for (int64_t r = 0; r < R; r++) {
        const int64_t gr = r0 + r;
        if (gr >= r1) break;
        for (int64_t g = 0; g < Kp / 32; g++) {
            const int64_t gc = k0 + g * 32;
            if (gc >= k1) break;
            quant(gr, gc, k1, &q[r * Kp + g * 32], sc[r * (Kp / 32) + g], mn[r * (Kp / 32) + g]);
        }
    }
    // stream order
    size_t o = 0;
    for (int t = 0; t < tiles; t++)
        for (int k = 0; k < kc; k++)
            for (int cr = 0; cr < NR; cr++) {
                const int64_t row0 = (int64_t) cr * tiles * GN + (int64_t) t * GN, col0 = (int64_t) k * GK;
                // nibbles [GN/8 nb][GK/8 kb][8 kk][8 nn / 2]
                for (int nb = 0; nb < GN / 8; nb++)
                    for (int kb = 0; kb < GK / 8; kb++)
                        for (int kk = 0; kk < 8; kk++)
                            for (int nn = 0; nn < 8; nn += 2) {
                                const int64_t c = col0 + kb * 8 + kk;
                                uint8_t a = q[(row0 + nb * 8 + nn) * Kp + c];
                                uint8_t b = q[(row0 + nb * 8 + nn + 1) * Kp + c];
                                out[o++] = (uint8_t) (a | (b << 4));
                            }
                // scale, min [GK/32 g][GN n] bf16 (q4s: scale only)
                for (int which = 0; which < (sym ? 1 : 2); which++) {
                    const std::vector<uint16_t> & a = which ? mn : sc;
                    for (int g = 0; g < GK / 32; g++)
                        for (int n = 0; n < GN; n++) {
                            uint16_t v = a[(row0 + n) * (Kp / 32) + (col0 / 32 + g)];
                            out[o++] = (uint8_t) (v & 0xff);
                            out[o++] = (uint8_t) (v >> 8);
                        }
                }
            }
}

// float rows of W (row stride ldw), quantized with quant_sym32 / quant_asym32
void pack_block(const float * W, int64_t ldw, int64_t r0, int64_t r1, int64_t k0, int64_t k1,
                int tiles, int kc, uint8_t * out, bool sym = false) {
    pack_block_fn(r0, r1, k0, k1, tiles, kc, out, sym,
                  [&](int64_t gr, int64_t gc, int64_t kend, uint8_t * qg, uint16_t & sg, uint16_t & mg) {
                      float xg[32];
                      for (int i = 0; i < 32; i++) xg[i] = gc + i < kend ? W[gr * ldw + gc + i] : 0.0f;
                      if (sym) quant_sym32(xg, qg, sg);
                      else quant_asym32(xg, qg, sg, mg);
                  });
}

// all blocks of one matrix (rows of W: N, cols: K, row stride ldw) -> m.nbytes at out
// threads: column blocks packed in parallel (the quantizer's search makes packing CPU-heavy).
// quant: the codes source (default: quantize W)
void pack_mat(const Mat & m, const float * W, int64_t ldw, uint8_t * out, int threads = 1,
              const QuantFn * quant = nullptr) {
    auto one = [&](int c) {
        const int r = c / m.S, s = c % m.S;
        const int64_t r0 = r * m.rows_col, r1 = std::min(m.N, (r + 1) * m.rows_col);
        const int64_t k0 = s * m.ks, k1 = std::min(m.K, (s + 1) * m.ks);
        uint8_t * dst = out + (size_t) c * m.cb;
        if (!(r1 > r0 && k1 > k0)) std::memset(dst, 0, m.cb);
        else if (quant) pack_block_fn(r0, r1, k0, k1, m.tiles, m.kc, dst, m.sym, *quant);
        else pack_block(W, ldw, r0, r1, k0, k1, m.tiles, m.kc, dst, m.sym);
    };
    const int nb = m.R * m.S;
    if (threads <= 1 || nb == 1) { for (int c = 0; c < nb; c++) one(c); return; }
    std::vector<std::thread> th;
    std::atomic<int> next{0};
    for (int t = 0; t < std::min(threads, nb); t++)
        th.emplace_back([&] { for (int c; (c = next.fetch_add(1)) < nb;) one(c); });
    for (auto & t : th) t.join();
}

// ------------------------------------------------------------------ device
std::string gemv_dir() {
    const char * e = std::getenv("OPENNPU_GEMV_DIR");
    if (e && *e) return e;
    const char * x = std::getenv("XDG_CACHE_HOME");
    std::string base = x && *x ? std::string(x) : std::string(std::getenv("HOME") ? std::getenv("HOME") : "") + "/.cache";
    const char * c = std::getenv("OPENNPU_CACHE");
    return (c && *c ? std::string(c) : base + "/opennpu") + "/gemv";
}

std::string gemm8_dir() {
    const char * e = std::getenv("OPENNPU_GEMM8_DIR");
    if (e && *e) return e;
    std::string g = gemv_dir();
    return g.substr(0, g.rfind('/')) + "/gemm8";
}

bool file_exists(const std::string & p) { struct stat st; return stat(p.c_str(), &st) == 0; }

std::vector<uint32_t> read_words(const std::string & p) {
    std::ifstream f(p, std::ios::binary);
    std::vector<char> b((std::istreambuf_iterator<char>(f)), {});
    std::vector<uint32_t> w(b.size() / 4);
    std::memcpy(w.data(), b.data(), w.size() * 4);
    return w;
}

// one of the in-flight dispatches of a shape (prompts): its own input, output and insts buffers
constexpr int NSLOT = 4;
struct Slot {
    xrt::bo x, y, ib;
    uint16_t * xm = nullptr;
    std::vector<uint32_t> words;
    xrt::run run;
};

struct Shape {
    std::unique_ptr<Slot> sl[4];   // NSLOT in-flight dispatches (the driver takes 4 per context)
    std::vector<int64_t> base;
    std::vector<size_t> idx;
    std::vector<int64_t> d;
    std::vector<int> col;
    xrt::bo y, ib;                 // output, dynamic insts (experts)
    std::vector<uint32_t> words;   // last dynamic insts
    std::vector<int64_t> last;     // offsets the dynamic insts hold
    int tiles = 0, kc = 0;

    void at(const int64_t * offs, uint32_t * out) const {
        for (size_t i = 0; i < idx.size(); i++) {
            int64_t num = d[i] * offs[col[i]];
            out[idx[i]] = (uint32_t) (base[idx[i]] + num / DELTA);
        }
    }
};

struct Prepared {                  // a fixed matrix: its own insts + run
    xrt::bo ib;
    xrt::run run;
    Shape * sh = nullptr;
};

struct Device {
    std::mutex mu;
    std::string dir, prefix = "gemv";
    int T = 1;                     // tokens per pass: 1 (GEMV, decode) or BT (prompts)
    bool ok = false;
    xrt::device dev;
    xrt::xclbin xb;
    xrt::hw_context ctx;
    xrt::kernel kern;
    xrt::bo x;
    uint16_t * xm = nullptr;
    std::map<std::pair<int, int>, std::unique_ptr<Shape>> shapes;
    int64_t calls = 0;
    double wait_s = 0;

    bool init(int t, bool sym = false) {
        T = t;
        prefix = T == 1 ? "gemv" : "gemm8";
        dir = (T == 1 ? gemv_dir() : gemm8_dir()) + (sym ? "-s" : "");
        if (!file_exists(dir + "/" + prefix + ".xclbin")) {
            if (T == 1) ONPU_WARN("opennpu: no GEMV device in %s (run OpenNPU once to build it)\n", dir.c_str());
            return false;
        }
        try {
            dev = xrt::device(0);
            std::string p = dir + "/" + prefix + ".xclbin";
            xb = xrt::xclbin{p};
            dev.register_xclbin(xb);
            ctx = xrt::hw_context(dev, xb.get_uuid());
            kern = xrt::kernel(ctx, "MLIR_AIE");
            x = bo((size_t) NC * KMAX * T * 2, 1);
            xm = x.map<uint16_t *>();
        } catch (const std::exception & e) {
            ONPU_WARN("opennpu: NPU unavailable: %s\n", e.what());
            return false;
        }
        ok = true;
        return true;
    }

    xrt::bo bo(size_t n, int arg) { return xrt::bo(dev, std::max<size_t>(n, 4096), xrt::bo::flags::host_only, kern.group_id(3 + arg)); }

    std::map<std::pair<int, int>, bool> have;   // has_shape answers (a stat per dispatch otherwise)
    std::mutex have_mu;
    bool has_shape(int tiles, int kc) {
        if (!ok) return false;
        std::lock_guard<std::mutex> lk(have_mu);
        auto key = std::make_pair(tiles, kc);
        auto it = have.find(key);
        if (it != have.end() && it->second) return true;   // a missing shape may be built later: re-check
        const bool h = file_exists(dir + "/" + prefix + "_" + std::to_string(tiles) + "_" + std::to_string(kc) + "_c7.insts.bin");
        have[key] = h;
        return h;
    }

    Shape * shape(int tiles, int kc) {
        auto key = std::make_pair(tiles, kc);
        auto it = shapes.find(key);
        if (it != shapes.end()) return it->second.get();
        const std::string n = dir + "/" + prefix + "_" + std::to_string(tiles) + "_" + std::to_string(kc);
        auto s = std::make_unique<Shape>();
        s->tiles = tiles; s->kc = kc;
        auto b = read_words(n + ".insts.bin");
        s->base.assign(b.begin(), b.end());
        for (int c = 0; c < NC; c++) {
            auto v = read_words(n + "_c" + std::to_string(c) + ".insts.bin");
            ONPU_ASSERT(v.size() == b.size());
            for (size_t i = 0; i < v.size(); i++)
                if (v[i] != b[i]) { s->idx.push_back(i); s->d.push_back((int64_t) v[i] - (int64_t) b[i]); s->col.push_back(c); }
        }
        s->y = bo((size_t) NC * out_tiles(tiles) * NR * GN * T * 2, 2);
        s->words.assign(b.begin(), b.end());
        s->ib = xrt::bo(dev, s->words.size() * 4, xrt::bo::flags::cacheable, kern.group_id(1));
        s->ib.write(s->words.data());
        s->ib.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        Shape * r = s.get();
        shapes[key] = std::move(s);
        return r;
    }

    Prepared prepare(xrt::bo & w, const int64_t * offs, int tiles, int kc) {
        Shape * s = shape(tiles, kc);
        Prepared p;
        p.sh = s;
        std::vector<uint32_t> words(s->base.size());
        for (size_t i = 0; i < words.size(); i++) words[i] = (uint32_t) s->base[i];
        s->at(offs, words.data());
        p.ib = xrt::bo(dev, words.size() * 4, xrt::bo::flags::cacheable, kern.group_id(1));
        p.ib.write(words.data());
        p.ib.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        p.run = xrt::run(kern);
        p.run.set_arg(0, 3);
        p.run.set_arg(1, p.ib);
        p.run.set_arg(2, (uint32_t) words.size());
        p.run.set_arg(3, w);
        p.run.set_arg(4, x);
        p.run.set_arg(5, s->y);
        return p;
    }

    // xs: [NC][ks * T] f32 column inputs (T = BT: x_block layout); returns y bf16 [NC][tiles*NR*GN*T]
    const uint16_t * exec(xrt::run & run, Shape * s, const float * xs, int64_t ks) {
        const int64_t xl = ks * T, xmax = (int64_t) KMAX * T;
        for (int c = 0; c < NC; c++) {
            uint16_t * d = xm + (size_t) c * xmax;
            for (int64_t k = 0; k < xl; k++) d[k] = f2bf(xs[c * xl + k]);
            for (int64_t k = xl; k < xmax; k++) d[k] = 0;
        }
        x.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto t0 = std::chrono::steady_clock::now();
        run.start();
        auto st = run.wait(std::chrono::milliseconds(30000));
        ONPU_ASSERT(st == ERT_CMD_STATE_COMPLETED && "opennpu: GEMV dispatch failed");
        wait_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        calls++;
        s->y.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        return s->y.map<const uint16_t *>();
    }

    // Pipelined dispatches: start(.., k) queues a run on slot k (0/1) and returns at once; finish(s, k)
    // waits for it. Two slots let the host prepare the next dispatch and unpack the previous one
    // while the NPU runs.
    // nt: token slots holding real inputs (8-token device: the others are left as they are)
    void start(xrt::bo & w, const int64_t * offs, Shape * s, const float * xs, int64_t ks, int k, int nt = BT) {
        if (!s->sl[k]) {
            auto q = std::make_unique<Slot>();
            q->x = bo((size_t) NC * KMAX * T * 2, 1);
            q->xm = q->x.map<uint16_t *>();
            std::memset(q->xm, 0, (size_t) NC * KMAX * T * 2);
            q->y = bo((size_t) NC * out_tiles(s->tiles) * NR * GN * T * 2, 2);
            q->words.assign(s->base.begin(), s->base.end());
            q->ib = xrt::bo(dev, q->words.size() * 4, xrt::bo::flags::cacheable, kern.group_id(1));
            s->sl[k] = std::move(q);
        }
        Slot & q = *s->sl[k];
        const int64_t xl = ks * T, xmax = (int64_t) KMAX * T;
        for (int c = 0; c < NC; c++) {
            uint16_t * d = q.xm + (size_t) c * xmax;
            const float * src = xs + c * xl;
            if (T == 1 || nt >= T)
                for (int64_t i = 0; i < xl; i++) d[i] = f2bf(src[i]);
            else                                       // [ks/8][T][8]: the first nt token slots of each group
                for (int64_t i = 0; i < xl; i += (int64_t) T * 8)
                    for (int64_t j = i; j < i + (int64_t) nt * 8; j++) d[j] = f2bf(src[j]);
        }
        q.x.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        s->at(offs, q.words.data());
        q.ib.write(q.words.data());
        q.ib.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        q.run = xrt::run(kern);
        q.run.set_arg(0, 3);
        q.run.set_arg(1, q.ib);
        q.run.set_arg(2, (uint32_t) q.words.size());
        q.run.set_arg(3, w);
        q.run.set_arg(4, q.x);
        q.run.set_arg(5, q.y);
        q.run.start();
        calls++;
    }

    const uint16_t * finish(Shape * s, int k) {
        Slot & q = *s->sl[k];
        auto t0 = std::chrono::steady_clock::now();
        auto st = q.run.wait(std::chrono::milliseconds(30000));
        ONPU_ASSERT(st == ERT_CMD_STATE_COMPLETED && "opennpu: GEMV dispatch failed");
        wait_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        q.y.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        return q.y.map<const uint16_t *>();
    }

    const uint16_t * dyn(xrt::bo & w, const int64_t * offs, Shape * s, const float * xs, int64_t ks) {
        bool same = s->last.size() == NC;
        for (int c = 0; same && c < NC; c++) same = s->last[c] == offs[c];
        if (!same) {
            s->at(offs, s->words.data());
            s->ib.write(s->words.data());
            s->ib.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            s->last.assign(offs, offs + NC);
        }
        xrt::run run(kern);
        run.set_arg(0, 3);
        run.set_arg(1, s->ib);
        run.set_arg(2, (uint32_t) s->words.size());
        run.set_arg(3, w);
        run.set_arg(4, x);
        run.set_arg(5, s->y);
        return exec(run, s, xs, ks);
    }
};

// sym: the devices for q4s weights (their own builds, gemv-s / gemm8-s)
Device & device(bool sym = false) {
    static Device d[2];
    static bool tried[2] = {false, false};
    if (!tried[sym]) { tried[sym] = true; d[sym].init(1, sym); }
    return d[sym];
}

// the 8-token device (prompts); OPENNPU_NPU4_BATCHED=0 turns it off
Device & device8(bool sym = false) {
    static Device d[2];
    static bool tried[2] = {false, false};
    if (!tried[sym]) {
        tried[sym] = true;
        const char * e = std::getenv("OPENNPU_NPU4_BATCHED");
        if (!(e && std::string(e) == "0")) d[sym].init(BT, sym);
    }
    return d[sym];
}

// rows[tok] (n <= BT rows, cols [k0, k0+ks) of each, zero past K) -> x_block layout [ks/8][BT][8].
// all_slots false: only the n token slots are written (the others' outputs are unused, so whatever
// they hold doesn't matter)
void x_block(const float * const * rows, int n, int64_t K, int64_t k0, int64_t ks, float * out, bool all_slots = true) {
    const int nt = all_slots ? BT : n;
    for (int64_t kt = 0; kt < ks / 8; kt++)
        for (int t = 0; t < nt; t++)
            for (int kk = 0; kk < 8; kk++) {
                const int64_t g = k0 + kt * 8 + kk;
                out[(kt * BT + t) * 8 + kk] = (t < n && g < K) ? rows[t][g] : 0.0f;
            }
}

// gather: device output of n copies of m (n*R*S columns from c0) -> out [n][N] (+=)
void gather(const Mat & m, const uint16_t * y, int c0, int n, float * out) {
    const int to = out_tiles(m.tiles);
    const int64_t per_col = (int64_t) to * NR * GN, t2 = to / FLUSH;
    for (int e = 0; e < n; e++) {
        float * o = out + (int64_t) e * m.N;
        std::memset(o, 0, m.N * sizeof(float));
        for (int r = 0; r < m.R; r++)
            for (int s = 0; s < m.S; s++) {
                const uint16_t * yc = y + (int64_t) (c0 + e * m.R * m.S + r * m.S + s) * per_col;
                for (int64_t tt = 0; tt < t2; tt++)
                    for (int cr = 0; cr < NR; cr++) {
                        // rows of this flush pair (an odd last tile's zero partner is skipped)
                        const int64_t nv = std::min<int64_t>(FLUSH * GN, ((int64_t) m.tiles - tt * FLUSH) * GN);
                        const int64_t row0 = r * m.rows_col + (int64_t) cr * m.tiles * GN + tt * FLUSH * GN;
                        const int64_t n = std::min<int64_t>(nv, m.N - row0);
                        const uint16_t * src = yc + (tt * NR + cr) * FLUSH * GN;
                        for (int64_t i = 0; i < n; i++) o[row0 + i] += bf2f(src[i]);
                    }
            }
    }
}

// one input row -> the column inputs of one copy of m, written at xs[c0..]
void x_cols(const Mat & m, const float * x, float * xs, int c0) {
    for (int r = 0; r < m.R; r++)
        for (int s = 0; s < m.S; s++) {
            float * d = xs + (int64_t) (c0 + r * m.S + s) * m.ks;
            for (int64_t k = 0; k < m.ks; k++) {
                const int64_t g = s * m.ks + k;
                d[k] = g < m.K ? x[g] : 0.0f;
            }
        }
}

// 8-token device output for copy e (its R*S columns from c0 + e*R*S) -> out [n][N] (=): the first
// n <= BT tokens only
void gather8(const Mat & m, const uint16_t * y, int c0, float * out, int n = BT) {
    const int to = out_tiles(m.tiles);
    const int64_t per_col = (int64_t) to * NR * GN * BT, t2n = to / FLUSH;
    std::memset(out, 0, (size_t) n * m.N * sizeof(float));
    for (int r = 0; r < m.R; r++)
        for (int s = 0; s < m.S; s++) {
            const uint16_t * yc = y + (int64_t) (c0 + r * m.S + s) * per_col;
            // [t2][NR][FLUSH][GN/8 nb][BT tok][8 nn]: runs of 8 consecutive rows per token
            for (int64_t t2 = 0; t2 < t2n; t2++)
                for (int cr = 0; cr < NR; cr++)
                    for (int j = 0; j < FLUSH; j++) {
                        if (t2 * FLUSH + j >= m.tiles) continue;              // a zero partner tile
                        const int64_t row0 = r * m.rows_col + (int64_t) cr * m.tiles * GN + (t2 * FLUSH + j) * GN;
                        if (row0 >= m.N) continue;
                        const uint16_t * src = yc + ((t2 * NR + cr) * FLUSH + j) * (int64_t) (GN / 8) * BT * 8;
                        for (int nb = 0; nb < GN / 8; nb++) {
                            const int64_t row = row0 + nb * 8;
                            const int cnt = (int) std::min<int64_t>(8, m.N - row);
                            if (cnt <= 0) break;
                            for (int t = 0; t < n; t++) {
                                const uint16_t * a = src + ((int64_t) nb * BT + t) * 8;
                                float * o = out + (int64_t) t * m.N + row;
                                for (int nn = 0; nn < cnt; nn++) o[nn] += bf2f(a[nn]);
                            }
                        }
                    }
        }
}

// n <= BT rows -> the 8-token device inputs of one copy of m, at xs[c0..] ([cols][ks*BT])
void x_cols8(const Mat & m, const float * const * rows, int n, float * xs, int c0, bool all_slots = true) {
    for (int r = 0; r < m.R; r++)
        for (int s = 0; s < m.S; s++)
            x_block(rows, n, m.K, s * m.ks, m.ks, xs + (int64_t) (c0 + r * m.S + s) * m.ks * BT, all_slots);
}


}  // namespace onpu
