// OpenNPU ggml backend: llama.cpp's matrix products on the AMD Ryzen AI (XDNA2) NPU.
//
// MUL_MAT and MUL_MAT_ID (mixture-of-experts) with a weight matrix as src0 run on OpenNPU's
// shape-generic int4 GEMV device (opennpu/native/npu4/gemv_device.py, built per (tiles, kc) shape
// into $OPENNPU_GEMV_DIR, default ~/.cache/opennpu/gemv). Everything else stays on the CPU.
//
//  * weights: any ggml type with a to_float (Q4_0/Q4_K/Q5_K/Q6_K/Q8_0/MXFP4/F16/BF16/F32 ...) is
//    dequantized once and re-quantized to Q4NX (per 32 weights a bf16 scale and min, 4-bit q:
//    the same codec as OpenNPU's other engines), packed in the device's stream order into NPU
//    buffers. Dense matrices pack on first use; expert banks reserve the whole bank and pack each
//    expert the first time the router picks it.
//  * geometry (must match opennpu/native/npu4/model.py Mat): a matrix spans the 8 columns as
//    R row blocks x S K-slices of one (tiles, kc) shape, so it is one dispatch; a token's experts
//    share one dispatch (8 / n_used columns each).
//  * insts: the base build plus one build per column with that column's weight offset moved by
//    DELTA; any offsets are a linear combination (OpenNPU's LinearInsts).
//
// SPDX-License-Identifier: MIT

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-opennpu.h"
#include "ggml-backend-impl.h"

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <functional>
#include <set>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <array>
#include <atomic>
#include <chrono>
#include <sys/stat.h>
#include <thread>
#include <vector>

#define ONPU_ASSERT(c) GGML_ASSERT(c)
#define ONPU_WARN(...) GGML_LOG_WARN(__VA_ARGS__)
#include "opennpu_core.hpp"

namespace {
using namespace onpu;

// ------------------------------------------------------------------ packed weights
// A "group" is one weight matrix or several that read the same input (q/k/v, gate/up, ...):
// their rows are packed as one matrix, run as one dispatch and the output split back.
using Group = std::vector<const ggml_tensor *>;

int64_t group_rows(const Group & g) { int64_t n = 0; for (auto * w : g) n += w->ne[1]; return n; }

struct Dense {
    Mat m;
    xrt::bo bo;
    Prepared prep, prep8;
    bool has8 = false;
};

struct Bank {                      // expert bank [E] of [N, K] (N: the group's rows per expert)
    Mat m;
    int64_t E = 0;
    xrt::bo bo;
    uint8_t * map = nullptr;
    std::vector<char> packed;
};

void to_float_rows(const ggml_tensor * t, int64_t i2, int64_t r0, int64_t r1, float * out) {
    const auto * tt = ggml_get_type_traits(t->type);
    const int64_t K = t->ne[0];
    for (int64_t r = r0; r < r1; r++) {
        const char * src = (const char *) t->data + i2 * t->nb[2] + r * t->nb[1];
        float * dst = out + (r - r0) * K;
        if (t->type == GGML_TYPE_F32) std::memcpy(dst, src, K * sizeof(float));
        else tt->to_float(src, dst, K);
    }
}

// the group's matrices (expert i2 of each) stacked by rows -> W [rows][K]
void dequant_group(const Group & g, int64_t i2, std::vector<float> & W) {
    const int64_t K = g[0]->ne[0];
    W.resize((size_t) K * group_rows(g));
    int64_t o = 0;
    for (auto * w : g) {
        to_float_rows(w, i2, 0, w->ne[1], W.data() + o * K);
        o += w->ne[1];
    }
}

// GGUF Q4_K transcodes straight into q4nx: each 32-weight sub-block of a Q4_K super-block is
// d * sc * q - dmin * m with 4-bit q, our format exactly; only the scale and min become bf16 (the
// pair among neighbouring bf16 values with the least error against the exact weights is kept).
bool group_q4k(const Group & g) {
    for (auto * w : g) if (w->type != GGML_TYPE_Q4_K) return false;
    return true;
}

uint16_t bf_step(uint16_t h, int dir) {                 // the neighbouring bf16 value
    const float f = bf2f(h);
    if (f == 0) return dir > 0 ? (uint16_t) 0x0001 : (uint16_t) 0x8001;
    return (uint16_t) (((f > 0) == (dir > 0)) ? h + 1 : h - 1);
}

QuantFn q4k_codes(const Group & g, int64_t i2) {
    return [&g, i2](int64_t gr, int64_t gc, int64_t, uint8_t * q, uint16_t & s_out, uint16_t & m_out) {
        const ggml_tensor * t = g[0];
        int64_t r = gr;
        for (auto * w : g) {
            if (r < w->ne[1]) { t = w; break; }
            r -= w->ne[1];
        }
        const uint8_t * b = (const uint8_t *) t->data + i2 * t->nb[2] + r * t->nb[1] + (gc / 256) * 144;
        const float d = ggml_fp16_to_fp32(*(const ggml_fp16_t *) b), dmin = ggml_fp16_to_fp32(*(const ggml_fp16_t *) (b + 2));
        const uint8_t * sc = b + 4, * qs = b + 16;
        const int sub = (int) ((gc % 256) / 32);
        uint8_t sv, mv;
        if (sub < 4) { sv = sc[sub] & 63; mv = sc[sub + 4] & 63; }
        else { sv = (sc[sub + 4] & 0xF) | ((sc[sub - 4] >> 6) << 4); mv = (sc[sub + 4] >> 4) | ((sc[sub] >> 6) << 4); }
        const uint8_t * qq = qs + (sub / 2) * 32;
        for (int l = 0; l < 32; l++) q[l] = (sub & 1) ? (qq[l] >> 4) : (qq[l] & 0xF);
        const float S = d * sv, M = -dmin * mv;
        const uint16_t s0 = f2bf(S), m0 = f2bf(M);
        float best = INFINITY;
        for (int ds = -1; ds <= 1; ds++)
            for (int dm = -1; dm <= 1; dm++) {
                const uint16_t sc_ = ds ? bf_step(s0, ds) : s0, mn_ = dm ? bf_step(m0, dm) : m0;
                const float sf = bf2f(sc_), mf = bf2f(mn_);
                float e = 0;
                for (int l = 0; l < 32; l++) { const float x = q[l] * sf + mf - (q[l] * S + M); e += x * x; }
                if (e < best) { best = e; s_out = sc_; m_out = mn_; }
            }
    };
}

struct Store {
    std::map<Group, std::unique_ptr<Dense>> dense;
    std::map<Group, std::unique_ptr<Bank>> banks;
};
Store & store() { static Store s; return s; }

struct Context {
    // OPENNPU_PROFILE=1: every ~3 s, where the time went (stderr)
    bool prof = std::getenv("OPENNPU_PROFILE") != nullptr;
    int64_t graphs = 0, nodes[2] = {0, 0};
    double op_s[2] = {0, 0};
    std::chrono::steady_clock::time_point t_last = std::chrono::steady_clock::now();
    // packed weights are shared by every backend instance (llama.cpp may open several contexts)
    std::map<Group, std::unique_ptr<Dense>> & dense = store().dense;
    std::map<Group, std::unique_ptr<Bank>> & banks = store().banks;
    std::vector<float> W, xs, ys;
    // learned fusion: weights that read the same input within one token run as one dispatch
    std::map<const ggml_tensor *, Group> group_of;                          // first member -> its group
    std::set<const ggml_tensor *> member;                                   // non-first members
    bool grouped = false;
    int64_t hits = 0, misses = 0;
    struct Pending {                    // a sibling's output, computed with the group's first member
        const ggml_tensor * src1 = nullptr;
        std::vector<float> x;           // the input it was computed for (checked before use)
        std::vector<int32_t> ids;
        std::vector<float> out;         // [rows or (t, j)] x N
    };
    std::map<const ggml_tensor *, Pending> pend;
};

// columns per expert: enough that a token's experts fill the 8 columns, more when K needs
// more slices (then fewer experts per dispatch)
int expert_cols(int n_used, int64_t N, int64_t K) {
    int c = (n_used <= NC && NC % n_used == 0) ? NC / n_used : 1;
    while (c < NC && !Mat(N, K, c).ok) c *= 2;
    return c;
}

Dense * get_dense(Context * ctx, const Group & g) {
    auto it = ctx->dense.find(g);
    if (it != ctx->dense.end()) return it->second.get();
    if (std::getenv("OPENNPU_DEBUG") && g.size() > 1) fprintf(stderr, "opennpu: pack dense %s (+%zu)\n", g[0]->name, g.size() - 1);
    Device & D = device();
    auto d = std::make_unique<Dense>();
    d->m = Mat(group_rows(g), g[0]->ne[0], NC);
    d->bo = D.bo(d->m.nbytes, 0);
    std::vector<uint8_t> blob(d->m.nbytes);
    if (group_q4k(g)) {
        const QuantFn f = q4k_codes(g, 0);
        pack_mat(d->m, nullptr, 0, blob.data(), 8, &f);
    } else {
        dequant_group(g, 0, ctx->W);
        pack_mat(d->m, ctx->W.data(), g[0]->ne[0], blob.data(), 8);
    }
    d->bo.write(blob.data());
    d->bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    int64_t offs[NC];
    for (int c = 0; c < NC; c++) offs[c] = (int64_t) c * d->m.cb;
    d->prep = D.prepare(d->bo, offs, d->m.tiles, d->m.kc);
    Dense * r = d.get();
    ctx->dense[g] = std::move(d);
    return r;
}

int pack_threads() {
    const char * e = std::getenv("OPENNPU_PACK_THREADS");
    int n = e ? std::atoi(e) : (int) std::thread::hardware_concurrency();
    return std::max(1, n);
}

Bank * get_bank(Context * ctx, const Group & g, int n_used) {
    auto it = ctx->banks.find(g);
    if (it != ctx->banks.end()) return it->second.get();
    if (std::getenv("OPENNPU_DEBUG")) fprintf(stderr, "opennpu: pack bank %s (+%zu)\n", g[0]->name, g.size() - 1);
    Device & D = device();
    auto b = std::make_unique<Bank>();
    const int64_t N = group_rows(g), K = g[0]->ne[0];
    b->m = Mat(N, K, expert_cols(n_used, N, K));
    b->E = g[0]->ne[2];
    b->bo = D.bo(b->m.nbytes * b->E, 0);
    b->map = b->bo.map<uint8_t *>();
    b->packed.assign(b->E, 0);
    // pack the whole bank now, in parallel (OPENNPU_LAZY_EXPERTS=1: pack each expert on first use)
    if (!std::getenv("OPENNPU_LAZY_EXPERTS")) {
        std::atomic<int64_t> next{0};
        std::vector<std::thread> th;
        Bank * bp = b.get();
        for (int i = 0; i < pack_threads(); i++)
            th.emplace_back([&, bp]() {
                std::vector<float> W;
                for (int64_t e; (e = next++) < bp->E;) {
                    if (group_q4k(g)) {
                        const QuantFn f = q4k_codes(g, e);
                        pack_mat(bp->m, nullptr, 0, bp->map + (size_t) e * bp->m.nbytes, 1, &f);
                    } else {
                        dequant_group(g, e, W);
                        pack_mat(bp->m, W.data(), K, bp->map + (size_t) e * bp->m.nbytes);
                    }
                    bp->packed[e] = 1;
                }
            });
        for (auto & t : th) t.join();
        b->bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
    Bank * r = b.get();
    ctx->banks[g] = std::move(b);
    return r;
}

void ensure_expert(Context * ctx, Bank * b, const Group & g, int64_t e) {
    if (b->packed[e]) return;
    if (group_q4k(g)) {
        const QuantFn f = q4k_codes(g, e);
        pack_mat(b->m, nullptr, 0, b->map + (size_t) e * b->m.nbytes, 1, &f);
    } else {
        dequant_group(g, e, ctx->W);
        pack_mat(b->m, ctx->W.data(), g[0]->ne[0], b->map + (size_t) e * b->m.nbytes);
    }
    b->bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, b->m.nbytes, (size_t) e * b->m.nbytes);
    b->packed[e] = 1;
}

using Sink = std::function<void(int64_t row, int64_t slot, const float * y)>;   // y: the group's N outputs

// x rows of a MUL_MAT input in (i1, i2, i3) order
std::vector<const float *> input_rows(const ggml_tensor * x) {
    std::vector<const float *> xr;
    for (int64_t i3 = 0; i3 < x->ne[3]; i3++)
        for (int64_t i2 = 0; i2 < x->ne[2]; i2++)
            for (int64_t i1 = 0; i1 < x->ne[1]; i1++)
                xr.push_back((const float *) ((const char *) x->data + i1 * x->nb[1] + i2 * x->nb[2] + i3 * x->nb[3]));
    return xr;
}

// ------------------------------------------------------------------ ops
// the group's product for every input row: sink(row, -1, y)
void run_dense(Context * ctx, const Group & g, const ggml_tensor * x, const Sink & sink) {
    Device & D = device();
    std::lock_guard<std::mutex> lk(D.mu);
    Dense * d = get_dense(ctx, g);
    const Mat & m = d->m;
    std::vector<const float *> xr = input_rows(x);
    Device & D8 = device8();
    if (xr.size() > 1 && D8.has_shape(m.tiles, m.kc)) {     // prompt: 8 rows per weight pass
        if (!d->has8) {
            int64_t offs[NC];
            for (int c = 0; c < NC; c++) offs[c] = (int64_t) c * m.cb;
            d->prep8 = D8.prepare(d->bo, offs, m.tiles, m.kc);
            d->has8 = true;
        }
        ctx->xs.resize((size_t) NC * m.ks * BT);
        ctx->ys.resize((size_t) BT * m.N);
        for (size_t r0 = 0; r0 < xr.size(); r0 += BT) {
            const int n = (int) std::min<size_t>(BT, xr.size() - r0);
            x_cols8(m, xr.data() + r0, n, ctx->xs.data(), 0);
            const uint16_t * y = D8.exec(d->prep8.run, d->prep8.sh, ctx->xs.data(), m.ks);
            gather8(m, y, 0, ctx->ys.data());
            for (int t = 0; t < n; t++) sink((int64_t) (r0 + t), -1, ctx->ys.data() + (int64_t) t * m.N);
        }
        return;
    }
    ctx->xs.resize((size_t) NC * m.ks);
    ctx->ys.resize((size_t) m.N);
    for (size_t r = 0; r < xr.size(); r++) {
        x_cols(m, xr[r], ctx->xs.data(), 0);
        const uint16_t * y = D.exec(d->prep.run, d->prep.sh, ctx->xs.data(), m.ks);
        gather(m, y, 0, 1, ctx->ys.data());
        sink((int64_t) r, -1, ctx->ys.data());
    }
}

// the group's expert products: sink(token, slot, y)
void run_experts(Context * ctx, const Group & g, const ggml_tensor * x, const ggml_tensor * ids, const Sink & sink) {
    const int n_used = (int) ids->ne[0];
    Device & D = device();
    std::lock_guard<std::mutex> lk(D.mu);
    Bank * b = get_bank(ctx, g, n_used);
    const Mat & m = b->m;
    const int per = NC / m.ncols;
    Device & D8 = device8();
    if (ids->ne[1] > 1 && D8.has_shape(m.tiles, m.kc)) {    // prompt: tokens grouped per expert, 8 per pass
        struct Item { int64_t e; std::vector<std::pair<int64_t, int>> p; };   // (token, slot)
        std::map<int64_t, std::vector<std::pair<int64_t, int>>> by;
        for (int64_t t = 0; t < ids->ne[1]; t++) {
            const int32_t * id = (const int32_t *) ((const char *) ids->data + t * ids->nb[1]);
            for (int j = 0; j < n_used; j++) by[id[j]].push_back({t, j});
        }
        std::vector<Item> items;
        for (auto & kv : by)
            for (size_t r0 = 0; r0 < kv.second.size(); r0 += BT)
                items.push_back({kv.first, std::vector<std::pair<int64_t, int>>(
                    kv.second.begin() + r0, kv.second.begin() + std::min(kv.second.size(), r0 + BT))});
        Shape * sh8 = D8.shape(m.tiles, m.kc);
        ctx->xs.resize((size_t) NC * m.ks * BT);
        ctx->ys.resize((size_t) BT * m.N);
        std::vector<const float *> rows(BT);
        for (size_t i0 = 0; i0 < items.size(); i0 += per) {
            const int k = (int) std::min<size_t>(per, items.size() - i0);
            int64_t offs[NC];
            for (int i = 0; i < per; i++) {
                const Item & it = items[i0 + std::min(i, k - 1)];      // spare columns repeat the last item
                ensure_expert(ctx, b, g, it.e);
                for (size_t r = 0; r < it.p.size(); r++) {
                    const int64_t xi = x->ne[1] == 1 ? 0 : it.p[r].second;
                    rows[r] = (const float *) ((const char *) x->data + xi * x->nb[1] + it.p[r].first * x->nb[2]);
                }
                x_cols8(m, rows.data(), (int) it.p.size(), ctx->xs.data(), i * m.ncols);
                for (int c = 0; c < m.ncols; c++) offs[i * m.ncols + c] = it.e * (int64_t) m.nbytes + (int64_t) c * m.cb;
            }
            const uint16_t * y = D8.dyn(b->bo, offs, sh8, ctx->xs.data(), m.ks);
            for (int i = 0; i < k; i++) {
                const Item & it = items[i0 + i];
                gather8(m, y, i * m.ncols, ctx->ys.data());
                for (size_t r = 0; r < it.p.size(); r++) sink(it.p[r].first, it.p[r].second, ctx->ys.data() + (int64_t) r * m.N);
            }
        }
        return;
    }
    Shape * sh = D.shape(m.tiles, m.kc);
    ctx->xs.resize((size_t) NC * m.ks);
    ctx->ys.resize((size_t) per * m.N);
    for (int64_t t = 0; t < ids->ne[1]; t++) {
        const int32_t * id = (const int32_t *) ((const char *) ids->data + t * ids->nb[1]);
        for (int e0 = 0; e0 < n_used; e0 += per) {
            const int k = std::min(per, n_used - e0);
            int64_t offs[NC];
            for (int i = 0; i < per; i++) {
                const int j = std::min(i, k - 1);            // spare columns repeat the last expert
                const int64_t e = id[e0 + j];
                ensure_expert(ctx, b, g, e);
                const int64_t xi = x->ne[1] == 1 ? 0 : e0 + j;
                const float * xr = (const float *) ((const char *) x->data + xi * x->nb[1] + t * x->nb[2]);
                x_cols(m, xr, ctx->xs.data(), i * m.ncols);
                for (int c = 0; c < m.ncols; c++) offs[i * m.ncols + c] = (int64_t) e * m.nbytes + (int64_t) c * m.cb;
            }
            const uint16_t * y = D.dyn(b->bo, offs, sh, ctx->xs.data(), m.ks);
            gather(m, y, 0, k, ctx->ys.data());
            for (int i = 0; i < k; i++) sink(t, e0 + i, ctx->ys.data() + (int64_t) i * m.N);
        }
    }
}

// ---- learned fusion
// Which weights read the same input, recorded while llama.cpp reserves its graphs (supports_op
// sees every node before anything runs), so groups are known before any weight is packed.
struct Siblings {
    std::mutex mu;
    std::map<const ggml_tensor *, std::vector<const ggml_tensor *>> by_input;   // src1 -> weights
    std::map<const ggml_tensor *, int> n_used;
} g_sib;

std::string blk_of(const char * name) {               // "blk.12." of "blk.12.attn_q.weight"
    std::string n(name);
    size_t a = n.find('.'), b = a == std::string::npos ? a : n.find('.', a + 1);
    return b == std::string::npos ? n : n.substr(0, b + 1);
}

void note_sibling(const ggml_tensor * op) {
    static const bool off = std::getenv("OPENNPU_FUSE") && std::string(std::getenv("OPENNPU_FUSE")) == "0";
    if (off) return;
    std::lock_guard<std::mutex> lk(g_sib.mu);
    auto & v = g_sib.by_input[op->src[1]];
    if (std::find(v.begin(), v.end(), op->src[0]) == v.end()) v.push_back(op->src[0]);
    if (op->op == GGML_OP_MUL_MAT_ID) g_sib.n_used[op->src[0]] = (int) op->src[2]->ne[0];
}
std::vector<float> input_copy(const ggml_tensor * x) {
    std::vector<float> v;
    const int64_t K = x->ne[0];
    for (const float * r : input_rows(x)) v.insert(v.end(), r, r + K);
    return v;
}

std::vector<int32_t> ids_copy(const ggml_tensor * ids) {
    std::vector<int32_t> v;
    if (!ids) return v;
    for (int64_t t = 0; t < ids->ne[1]; t++) {
        const int32_t * id = (const int32_t *) ((const char *) ids->data + t * ids->nb[1]);
        v.insert(v.end(), id, id + ids->ne[0]);
    }
    return v;
}

int64_t node_rows(const ggml_tensor * node) {
    return node->op == GGML_OP_MUL_MAT_ID ? node->src[2]->ne[1] * node->src[2]->ne[0]
                                         : node->src[1]->ne[1] * node->src[1]->ne[2] * node->src[1]->ne[3];
}

// groups from the registry: same block, same K, same kind; the largest prefix with a built shape
void finalize_groups(Context * ctx) {
    std::lock_guard<std::mutex> lk(g_sib.mu);
    for (auto & kv : g_sib.by_input) {
        const auto & v = kv.second;
        if (v.size() < 2) continue;
        Group g{v[0]};
        for (size_t i = 1; i < v.size() && g.size() < 4; i++) {
            const ggml_tensor * w = v[i];
            if (w->ne[0] == g[0]->ne[0] && w->ne[2] == g[0]->ne[2] && blk_of(w->name) == blk_of(g[0]->name) &&
                (w->ne[2] > 1) == (g[0]->ne[2] > 1))
                g.push_back(w);
        }
        while (g.size() > 1) {
            const int64_t N = group_rows(g), K = g[0]->ne[0];
            const bool ex = g[0]->ne[2] > 1;
            const int nc = ex ? expert_cols(g_sib.n_used.count(g[0]) ? g_sib.n_used[g[0]] : 8, N, K) : NC;
            Mat m(N, K, nc);
            if (m.ok && device().has_shape(m.tiles, m.kc)) break;
            g.pop_back();
        }
        if (g.size() < 2 || ctx->group_of.count(g[0]) || ctx->member.count(g[0])) continue;
        bool clash = false;
        for (size_t i = 1; i < g.size(); i++) clash |= ctx->group_of.count(g[i]) || ctx->member.count(g[i]);
        if (clash) continue;
        ctx->group_of[g[0]] = g;
        for (size_t i = 1; i < g.size(); i++) ctx->member.insert(g[i]);
        if (std::getenv("OPENNPU_DEBUG")) {
            fprintf(stderr, "opennpu: fuse");
            for (auto * w : g) fprintf(stderr, " %s", w->name);
            fprintf(stderr, "\n");
        }
    }
}

void op_node(Context * ctx, ggml_tensor * node) {
    const ggml_tensor * w = node->src[0], * x = node->src[1];
    const bool ex = node->op == GGML_OP_MUL_MAT_ID;
    const ggml_tensor * ids = ex ? node->src[2] : nullptr;
    if (!ctx->grouped) { finalize_groups(ctx); ctx->grouped = true; }
    // a sibling whose output was computed with its group's first member
    auto pit = ctx->pend.find(w);
    if (pit != ctx->pend.end()) {
        Context::Pending & p = pit->second;
        bool ok = p.src1 == x && p.x == input_copy(x) && p.ids == ids_copy(ids);
        if (ok) {
            const int64_t N = w->ne[1];
            if (!ex) {
                std::vector<const float *> xr = input_rows(x);
                for (size_t r = 0; r < xr.size(); r++) {
                    const int64_t i1 = r % x->ne[1], i2 = (r / x->ne[1]) % x->ne[2], i3 = r / (x->ne[1] * x->ne[2]);
                    std::memcpy((char *) node->data + i1 * node->nb[1] + i2 * node->nb[2] + i3 * node->nb[3],
                                p.out.data() + r * N, N * sizeof(float));
                }
            } else {
                for (int64_t t = 0; t < ids->ne[1]; t++)
                    for (int64_t j = 0; j < ids->ne[0]; j++)
                        std::memcpy((char *) node->data + j * node->nb[1] + t * node->nb[2],
                                    p.out.data() + (t * ids->ne[0] + j) * N, N * sizeof(float));
            }
            ctx->pend.erase(pit);
            ctx->hits++;
            return;
        }
        ctx->misses++;
        if (std::getenv("OPENNPU_DEBUG") && ctx->misses < 20)
            fprintf(stderr, "opennpu: pending miss %s (src1 %s, x %s, ids %s)\n", w->name, p.src1 == x ? "same" : "other",
                    p.x == input_copy(x) ? "same" : "differs", p.ids == ids_copy(ids) ? "same" : "differs");
        ctx->pend.erase(pit);
    }
    Group g{w};
    auto git = ctx->group_of.find(w);
    if (git != ctx->group_of.end()) g = git->second;
    // where each member's slice of the group output goes: [0] = this node, others -> pending
    std::vector<Context::Pending *> pp(g.size(), nullptr);
    if (g.size() > 1) {
        const std::vector<float> xc = input_copy(x);
        const std::vector<int32_t> ic = ids_copy(ids);
        for (size_t i = 1; i < g.size(); i++) {
            Context::Pending & p = ctx->pend[g[i]];
            p.src1 = x; p.x = xc; p.ids = ic;
            p.out.assign((size_t) node_rows(node) * g[i]->ne[1], 0.0f);
            pp[i] = &p;
        }
    }
    auto put = [&](int64_t row, int64_t slot, const float * y) {
        int64_t o = 0;
        for (size_t i = 0; i < g.size(); i++) {
            const int64_t n = g[i]->ne[1];
            if (i == 0) {
                char * dst;
                if (!ex) {
                    const int64_t i1 = row % x->ne[1], i2 = (row / x->ne[1]) % x->ne[2], i3 = row / (x->ne[1] * x->ne[2]);
                    dst = (char *) node->data + i1 * node->nb[1] + i2 * node->nb[2] + i3 * node->nb[3];
                } else {
                    dst = (char *) node->data + slot * node->nb[1] + row * node->nb[2];
                }
                std::memcpy(dst, y + o, n * sizeof(float));
            } else {
                const int64_t r = ex ? row * ids->ne[0] + slot : row;
                std::memcpy(pp[i]->out.data() + r * n, y + o, n * sizeof(float));
            }
            o += n;
        }
    };
    if (ex) run_experts(ctx, g, x, ids, put);
    else run_dense(ctx, g, x, put);
}

ggml_backend_buffer_type_t opennpu_buft();

bool weight_ok(const ggml_tensor * w) {
    if (!w || !w->buffer) return false;
    // llama.cpp's weight placement probe: no data yet, an empty buffer of our buffer type
    const bool probe = w->data == nullptr && ggml_backend_buffer_get_type(w->buffer) == opennpu_buft();
    if (!probe && ggml_backend_buffer_get_usage(w->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) return false;
    if (!ggml_is_contiguous(w) || w->ne[0] % 32 != 0) return false;
    if (w->type != GGML_TYPE_F32 && ggml_get_type_traits(w->type)->to_float == nullptr) return false;
    return true;
}

bool shape_ok(int64_t N, int64_t K, int ncols) {
    Mat m(N, K, ncols);
    return m.ok && device().ok && device().has_shape(m.tiles, m.kc);
}

}  // namespace

// ------------------------------------------------------------------ backend interface
static const char * opennpu_get_name(ggml_backend_t) { return "OpenNPU"; }

static void opennpu_free(ggml_backend_t backend) {
    delete (Context *) backend->context;
    delete backend;
}

static ggml_status opennpu_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    auto * ctx = (Context *) backend->context;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) continue;
        auto t0 = std::chrono::steady_clock::now();
        const int kind = node->op == GGML_OP_MUL_MAT_ID ? 1 : 0;
        switch (node->op) {
            case GGML_OP_MUL_MAT: case GGML_OP_MUL_MAT_ID: op_node(ctx, node); break;
            case GGML_OP_NONE: case GGML_OP_RESHAPE: case GGML_OP_VIEW:
            case GGML_OP_PERMUTE: case GGML_OP_TRANSPOSE: break;
            default: GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
        }
        if (ctx->prof && (node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID)) {
            ctx->op_s[kind] += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            ctx->nodes[kind]++;
        }
    }
    if (ctx->prof) {
        ctx->graphs++;
        auto now = std::chrono::steady_clock::now();
        const double wall = std::chrono::duration<double>(now - ctx->t_last).count();
        if (wall > 3.0) {                       // every ~3 s: shares of wall time
            Device & D = device();
            fprintf(stderr, "opennpu profile: %.0f%% in NPU ops (mul_mat %.0f%%, %.0f/s; mul_mat_id %.0f%%, %.0f/s), "
                            "device wait %.0f%%, %lld NPU splits/s, %.0f%% outside (CPU ops)\n",
                    100 * (ctx->op_s[0] + ctx->op_s[1]) / wall, 100 * ctx->op_s[0] / wall, ctx->nodes[0] / wall,
                    100 * ctx->op_s[1] / wall, ctx->nodes[1] / wall, 100 * D.wait_s / wall,
                    (long long) (ctx->graphs / wall), 100 * (wall - ctx->op_s[0] - ctx->op_s[1]) / wall);
            ctx->op_s[0] = ctx->op_s[1] = 0; ctx->nodes[0] = ctx->nodes[1] = 0; D.wait_s = 0;
            ctx->graphs = 0; ctx->t_last = now;
        }
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_backend_i opennpu_backend_i = {
    /* .get_name                = */ opennpu_get_name,
    /* .free                    = */ opennpu_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ opennpu_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t opennpu_guid(void) {
    static ggml_guid guid = { 0x4f, 0x70, 0x65, 0x6e, 0x4e, 0x50, 0x55, 0x21, 0x78, 0x64, 0x6e, 0x61, 0x32, 0x67, 0x65, 0x6d };
    return &guid;
}

ggml_backend_t ggml_backend_opennpu_init(void) {
    return new ggml_backend {
        /* .guid    = */ opennpu_guid(),
        /* .iface   = */ opennpu_backend_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_opennpu_reg(), 0),
        /* .context = */ new Context,
    };
}

bool ggml_backend_is_opennpu(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, opennpu_guid());
}

static const char * opennpu_dev_get_name(ggml_backend_dev_t) { return "OpenNPU"; }
static const char * opennpu_dev_get_description(ggml_backend_dev_t) { return "AMD Ryzen AI NPU (XDNA2), int4 GEMV"; }
static void opennpu_dev_get_memory(ggml_backend_dev_t, size_t * free, size_t * total) { *free = 0; *total = 0; }
static enum ggml_backend_dev_type opennpu_dev_get_type(ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_ACCEL; }

static void opennpu_dev_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name = opennpu_dev_get_name(dev);
    props->description = opennpu_dev_get_description(dev);
    props->type = opennpu_dev_get_type(dev);
    opennpu_dev_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = { /* async */ false, /* host_buffer */ false, /* buffer_from_host_ptr */ true,
                    /* events */ false, /* mmap_support */ true };
}

static ggml_backend_t opennpu_dev_init_backend(ggml_backend_dev_t, const char *) { return ggml_backend_opennpu_init(); }
// plain host memory under its own name: llama.cpp skips an ACCEL device whose buffer type is the CPU's, and
// then the CPU's repacked (interleaved) weight layouts claim the weights first. With its own type, each weight
// is offered to the NPU first (normal layout, stays mmapped); the ones it declines (small matrices) fall
// through to the CPU's faster repacked layouts.
static ggml_backend_dev_t opennpu_device_ptr();
static const char * opennpu_buft_name(ggml_backend_buffer_type_t) { return "OpenNPU_Host"; }
static ggml_backend_buffer_type_t opennpu_buft_impl() {
    static ggml_backend_buffer_type buft = [] {
        ggml_backend_buffer_type b = *ggml_backend_cpu_buffer_type();
        b.iface.get_name = opennpu_buft_name;
        b.device = opennpu_device_ptr();
        b.context = nullptr;
        return b;
    }();
    return &buft;
}
namespace { ggml_backend_buffer_type_t opennpu_buft() { return opennpu_buft_impl(); } }
static ggml_backend_buffer_type_t opennpu_dev_get_buffer_type(ggml_backend_dev_t) { return opennpu_buft_impl(); }
static ggml_backend_buffer_t opennpu_dev_buffer_from_host_ptr(ggml_backend_dev_t, void * ptr, size_t size, size_t) {
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);
}

static bool opennpu_supports(const ggml_tensor * op);

// OPENNPU_DEBUG=1: say once per weight which matrix products run on the NPU and which stay on the CPU
static bool opennpu_dev_supports_op(ggml_backend_dev_t, const ggml_tensor * op) {
    const bool r = opennpu_supports(op);
    if (r && (op->op == GGML_OP_MUL_MAT || op->op == GGML_OP_MUL_MAT_ID) && op->src[0]->data) note_sibling(op);
    static const bool dbg = std::getenv("OPENNPU_DEBUG") != nullptr;
    if (dbg && (op->op == GGML_OP_MUL_MAT || op->op == GGML_OP_MUL_MAT_ID) && op->src[0]) {
        static std::mutex m;
        static std::map<std::string, bool> seen;
        std::lock_guard<std::mutex> lk(m);
        const std::string n = op->src[0]->name;
        if (!seen.count(n)) {
            seen[n] = r;
            const ggml_tensor * w = op->src[0];
            GGML_LOG_INFO("opennpu: %-28s %s [%lld x %lld x %lld] %s %s\n", n.c_str(), ggml_op_name(op->op),
                          (long long) w->ne[0], (long long) w->ne[1], (long long) w->ne[2], ggml_type_name(w->type),
                          r ? "-> NPU" : "-> CPU");
        }
    }
    return r;
}

// OPENNPU_CPU_TENSORS="output.weight,..." : weights (name prefixes) that stay on the CPU, e.g. to keep
// a 6-bit output layer at full precision
static bool keep_on_cpu(const ggml_tensor * w) {
    static const std::vector<std::string> names = [] {
        std::vector<std::string> v;
        const char * e = std::getenv("OPENNPU_CPU_TENSORS");
        std::string cur;
        for (const char * c = e ? e : ""; ; c++) {
            if (*c == ',' || *c == 0) { if (!cur.empty()) v.push_back(cur); cur.clear(); if (!*c) break; }
            else cur += *c;
        }
        return v;
    }();
    for (const auto & n : names)
        if (std::strncmp(w->name, n.c_str(), n.size()) == 0) return true;
    return false;
}

// dense matrices below this many weights run faster on the CPU than one NPU dispatch (fixed ~75 us);
// OPENNPU_MIN_PARAMS overrides the default
static int64_t min_params() {
    static const int64_t v = [] {
        const char * e = std::getenv("OPENNPU_MIN_PARAMS");
        return e ? std::atoll(e) : MIN_PARAMS;
    }();
    return v;
}

static bool opennpu_supports(const ggml_tensor * op) {
    if ((op->op == GGML_OP_MUL_MAT || op->op == GGML_OP_MUL_MAT_ID) && op->src[0] && keep_on_cpu(op->src[0])) return false;
    switch (op->op) {
        case GGML_OP_NONE: case GGML_OP_RESHAPE: case GGML_OP_VIEW:
        case GGML_OP_PERMUTE: case GGML_OP_TRANSPOSE:
            return true;
        case GGML_OP_MUL_MAT: {
            const ggml_tensor * w = op->src[0], * x = op->src[1];
            if (ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD) return false;
            if (!weight_ok(w) || w->ne[2] != 1 || w->ne[3] != 1 || x->type != GGML_TYPE_F32) return false;
            if (x->nb[0] != sizeof(float) || op->nb[0] != sizeof(float)) return false;
            if (w->ne[0] * w->ne[1] < min_params()) return false;
            return shape_ok(w->ne[1], w->ne[0], NC);
        }
        case GGML_OP_MUL_MAT_ID: {
            const ggml_tensor * w = op->src[0], * x = op->src[1], * ids = op->src[2];
            if (!weight_ok(w) || x->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32) return false;
            if (x->nb[0] != sizeof(float) || op->nb[0] != sizeof(float)) return false;
            if (w->ne[0] * w->ne[1] < MIN_PARAMS / 8) return false;
            return shape_ok(w->ne[1], w->ne[0], expert_cols((int) ids->ne[0], w->ne[1], w->ne[0]));
        }
        default:
            return false;
    }
}

static bool opennpu_dev_supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft);
}

static const ggml_backend_device_i opennpu_device_i = {
    /* .get_name             = */ opennpu_dev_get_name,
    /* .get_description      = */ opennpu_dev_get_description,
    /* .get_memory           = */ opennpu_dev_get_memory,
    /* .get_type             = */ opennpu_dev_get_type,
    /* .get_props            = */ opennpu_dev_get_props,
    /* .init_backend         = */ opennpu_dev_init_backend,
    /* .get_buffer_type      = */ opennpu_dev_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ opennpu_dev_buffer_from_host_ptr,
    /* .supports_op          = */ opennpu_dev_supports_op,
    /* .supports_buft        = */ opennpu_dev_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

static const char * opennpu_reg_get_name(ggml_backend_reg_t) { return "OpenNPU"; }
static size_t opennpu_reg_get_device_count(ggml_backend_reg_t) { return device().ok ? 1 : 0; }

static ggml_backend_dev_t opennpu_reg_get_device(ggml_backend_reg_t, size_t index) {
    GGML_ASSERT(index == 0);
    return opennpu_device_ptr();
}

static void * opennpu_get_proc_address(ggml_backend_reg_t, const char *) { return NULL; }

static const ggml_backend_reg_i opennpu_reg_i = {
    /* .get_name         = */ opennpu_reg_get_name,
    /* .get_device_count = */ opennpu_reg_get_device_count,
    /* .get_device       = */ opennpu_reg_get_device,
    /* .get_proc_address = */ opennpu_get_proc_address,
};

static ggml_backend_dev_t opennpu_device_ptr() {
    static ggml_backend_device dev = { /* .iface = */ opennpu_device_i, /* .reg = */ ggml_backend_opennpu_reg(), /* .context = */ nullptr };
    return &dev;
}

ggml_backend_reg_t ggml_backend_opennpu_reg(void) {
    static ggml_backend_reg reg = { /* .api_version = */ GGML_BACKEND_API_VERSION, /* .iface = */ opennpu_reg_i, /* .context = */ NULL };
    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_opennpu_reg)
