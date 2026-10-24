// libonpu: a C API over OpenNPU's NPU core (opennpu_core.hpp) for OpenNPU's own engines.
// The 4-bit engine (opennpu/native/npu4) packs weights in Python (the same geometry, checked with
// onpu_geometry) straight into these buffers and then calls one C function per matrix product, so
// the per-dispatch host work (input conversion, run, output unpacking) is C++.
// SPDX-License-Identifier: MIT
#include "opennpu_core.hpp"

#include <algorithm>
#include <cmath>

using namespace onpu;

namespace {

struct DenseH {
    Mat m;
    xrt::bo bo;
    uint8_t * map = nullptr;
    Prepared prep, prep8;
    bool has8 = false;
    std::vector<float> xs, ys;
};

struct BankH {
    Mat m;
    int64_t E = 0;
    int64_t base = 0;              // byte offset of expert 0 in bo (a fused FFN keeps both banks in one bo)
    xrt::bo bo;
    uint8_t * map = nullptr;
    std::vector<int32_t> perm;     // fused gate_up: packed row r holds original row perm[r]
    std::vector<float> xs, ys;
};

// A routed-expert FFN whose gate_up and down run in ONE NPU dispatch (decode): the gate_up phase
// writes [gate | up] into the input buffer, the down phase reads it back and applies SiLU(gate) * up
// while expanding its input (gv_expand, kc >> 8). Needs one column per expert and 8 experts.
struct FfnInsts {                  // for one device (1-token or 8-token)
    std::vector<uint32_t> base;    // the two phases' insts, joined
    std::vector<size_t> idx;       // W-offset words: position, delta, column, phase
    std::vector<int64_t> d;
    std::vector<int> col, phase;
    xrt::bo ib;
    std::vector<uint32_t> words;
};

struct FfnH {
    BankH * gu = nullptr, * dn = nullptr;
    bool fused = false;
    FfnInsts one, eight;
    std::vector<float> xs, ys;
};

std::mutex g_mu;

}  // namespace

extern "C" {

int onpu_ok(void) { return device().ok ? 1 : 0; }
int onpu_batched_ok(void) { return device8().ok ? 1 : 0; }

// R, S, tiles, kc, ks, rows_col, cb, nbytes of an [N, K] matrix on ncols columns (0s if no geometry);
// sym: the q4s weight format
void onpu_geometry2(int64_t N, int64_t K, int ncols, int sym, int64_t * out) {
    Mat m(N, K, ncols, sym != 0);
    int64_t v[8] = {m.R, m.S, m.tiles, m.kc, m.ks, m.rows_col, (int64_t) m.cb, (int64_t) m.nbytes};
    for (int i = 0; i < 8; i++) out[i] = m.ok ? v[i] : 0;
}
void onpu_geometry(int64_t N, int64_t K, int ncols, int64_t * out) { onpu_geometry2(N, K, ncols, 0, out); }

// dispatches so far and seconds spent waiting for them (all devices); for profiling
void onpu_stats(int64_t * calls, double * wait_s) {
    *calls = 0;
    *wait_s = 0;
    for (int sym = 0; sym < 2; sym++)
        for (Device * d : {&device(sym), &device8(sym)}) {
            *calls += d->calls;
            *wait_s += d->wait_s;
        }
}

int onpu_has_shape(int tiles, int kc, int batched) {
    return (batched ? device8() : device()).has_shape(tiles, kc) ? 1 : 0;
}

// ---- dense
void * onpu_dense_new2(int64_t N, int64_t K, int sym) {
    auto * d = new DenseH;
    d->m = Mat(N, K, NC, sym != 0);
    if (!d->m.ok || !device(d->m.sym).has_shape(d->m.tiles, d->m.kc)) {   // shape not built
        if (std::getenv("ONPU_DEBUG"))
            fprintf(stderr, "onpu_dense_new2(%ld, %ld, sym %d): m.ok %d device ok %d has_shape(%d,%d) %d dir %s\n", (long) N,
                    (long) K, sym, d->m.ok, device(d->m.sym).ok, d->m.tiles, d->m.kc,
                    device(d->m.sym).has_shape(d->m.tiles, d->m.kc), device(d->m.sym).dir.c_str());
        delete d;
        return nullptr;
    }
    d->bo = device(d->m.sym).bo(d->m.nbytes, 0);
    d->map = d->bo.map<uint8_t *>();
    return d;
}
void * onpu_dense_new(int64_t N, int64_t K) { return onpu_dense_new2(N, K, 0); }
uint8_t * onpu_dense_buf(void * h) { return ((DenseH *) h)->map; }
int64_t onpu_dense_bytes(void * h) { return (int64_t) ((DenseH *) h)->m.nbytes; }

void onpu_dense_sync(void * h) {
    auto * d = (DenseH *) h;
    Device & D = device(d->m.sym);
    std::lock_guard<std::mutex> lk(g_mu);
    d->bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    int64_t offs[NC];
    for (int c = 0; c < NC; c++) offs[c] = (int64_t) c * d->m.cb;
    d->prep = D.prepare(d->bo, offs, d->m.tiles, d->m.kc);
}

void onpu_dense_free(void * h) { std::lock_guard<std::mutex> lk(g_mu); delete (DenseH *) h; }

// y [rows][N] = x [rows][K] . W^T (rows > 1: the 8-token device when it has the shape)
void onpu_dense_mv(void * h, const float * x, int64_t rows, float * y) {
    auto * d = (DenseH *) h;
    const Mat & m = d->m;
    std::lock_guard<std::mutex> lk(g_mu);
    Device & D8 = device8(m.sym);
    if (rows > 1 && D8.has_shape(m.tiles, m.kc)) {         // prompt: 8 rows per pass, two passes in flight
        Shape * sh8 = D8.shape(m.tiles, m.kc);
        int64_t offs[NC];
        for (int c = 0; c < NC; c++) offs[c] = (int64_t) c * m.cb;
        const int64_t nb = (rows + BT - 1) / BT;
        std::vector<float> xbuf[NSLOT];
        for (auto & b : xbuf) b.resize((size_t) NC * m.ks * BT);
        d->ys.resize((size_t) BT * m.N);
        auto start = [&](int64_t b) {
            const float * xr[BT];
            const int n = (int) std::min<int64_t>(BT, rows - b * BT);
            for (int t = 0; t < n; t++) xr[t] = x + (b * BT + t) * m.K;
            x_cols8(m, xr, n, xbuf[b % NSLOT].data(), 0, false);
            D8.start(d->bo, offs, sh8, xbuf[b % NSLOT].data(), m.ks, (int) (b % NSLOT), n);
        };
        for (int64_t b = 0; b < std::min<int64_t>(NSLOT - 1, nb); b++) start(b);
        for (int64_t b = 0; b < nb; b++) {
            if (b + NSLOT - 1 < nb) start(b + NSLOT - 1);       // keep NSLOT dispatches queued
            const uint16_t * yy = D8.finish(sh8, (int) (b % NSLOT));
            const int n = (int) std::min<int64_t>(BT, rows - b * BT);
            gather8(m, yy, 0, d->ys.data(), n);
            std::memcpy(y + b * BT * m.N, d->ys.data(), (size_t) n * m.N * sizeof(float));
        }
        return;
    }
    Device & D = device(m.sym);
    d->xs.resize((size_t) NC * m.ks);
    for (int64_t r = 0; r < rows; r++) {
        x_cols(m, x + r * m.K, d->xs.data(), 0);
        const uint16_t * yy = D.exec(d->prep.run, d->prep.sh, d->xs.data(), m.ks);
        gather(m, yy, 0, 1, y + r * m.N);
    }
}

// ---- expert banks: E experts of [N, K], each packed in m.nbytes at e * m.nbytes
void * onpu_bank_new2(int64_t N, int64_t K, int64_t E, int ncols, int sym) {
    auto * b = new BankH;
    b->m = Mat(N, K, ncols, sym != 0);
    if (!b->m.ok || !device(b->m.sym).has_shape(b->m.tiles, b->m.kc)) { delete b; return nullptr; }
    b->E = E;
    b->bo = device(b->m.sym).bo(b->m.nbytes * E, 0);
    b->map = b->bo.map<uint8_t *>();
    return b;
}
void * onpu_bank_new(int64_t N, int64_t K, int64_t E, int ncols) { return onpu_bank_new2(N, K, E, ncols, 0); }
uint8_t * onpu_bank_buf(void * h) { return ((BankH *) h)->map; }
int64_t onpu_bank_expert_bytes(void * h) { return (int64_t) ((BankH *) h)->m.nbytes; }
void onpu_bank_sync(void * h) {
    std::lock_guard<std::mutex> lk(g_mu);
    ((BankH *) h)->bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
}
void onpu_bank_free(void * h) { std::lock_guard<std::mutex> lk(g_mu); delete (BankH *) h; }

// y [T][k][N]: token t, slot j -> expert ids[t*k + j] applied to its input:
// x [T][K] when x_per_slot == 0 (one input per token), else [T][k][K]
void onpu_bank_run(void * h, const float * x, int x_per_slot, int64_t T, const int32_t * ids, int k, float * y) {
    auto * b = (BankH *) h;
    const Mat & m = b->m;
    const int per = NC / m.ncols;
    std::lock_guard<std::mutex> lk(g_mu);
    auto xin = [&](int64_t t, int j) { return x + (x_per_slot ? (t * k + j) : t) * m.K; };
    Device & D8 = device8(m.sym);
    if (T > 1 && D8.has_shape(m.tiles, m.kc)) {             // prompt: tokens grouped per expert, 8 per pass
        struct Item { int64_t e; std::vector<std::pair<int64_t, int>> p; };
        std::map<int64_t, std::vector<std::pair<int64_t, int>>> by;
        for (int64_t t = 0; t < T; t++)
            for (int j = 0; j < k; j++) by[ids[t * k + j]].push_back({t, j});
        std::vector<Item> items;
        for (auto & kv : by)
            for (size_t r0 = 0; r0 < kv.second.size(); r0 += BT)
                items.push_back({kv.first, std::vector<std::pair<int64_t, int>>(
                    kv.second.begin() + r0, kv.second.begin() + std::min(kv.second.size(), r0 + BT))});
        Shape * sh8 = D8.shape(m.tiles, m.kc);
        const int64_t ng = ((int64_t) items.size() + per - 1) / per;
        std::vector<float> xbuf[NSLOT];
        for (auto & v : xbuf) v.resize((size_t) NC * m.ks * BT);
        b->ys.resize((size_t) BT * m.N);
        auto start = [&](int64_t g) {               // one dispatch: `per` items, each on its own columns
            const int64_t i0 = g * per;
            const int kk = (int) std::min<int64_t>(per, (int64_t) items.size() - i0);
            int64_t offs[NC];
            const float * rows[BT];
            int nmax = 0;
            for (int i = 0; i < kk; i++) nmax = std::max(nmax, (int) items[i0 + i].p.size());
            for (int i = 0; i < per; i++) {
                const Item & it = items[i0 + std::min(i, kk - 1)];
                for (size_t r = 0; r < it.p.size(); r++) rows[r] = xin(it.p[r].first, it.p[r].second);
                x_cols8(m, rows, (int) it.p.size(), xbuf[g % NSLOT].data(), i * m.ncols, false);
                for (int c = 0; c < m.ncols; c++) offs[i * m.ncols + c] = b->base + it.e * (int64_t) m.nbytes + (int64_t) c * m.cb;
            }
            D8.start(b->bo, offs, sh8, xbuf[g % NSLOT].data(), m.ks, (int) (g % NSLOT), nmax);
        };
        for (int64_t g = 0; g < std::min<int64_t>(NSLOT - 1, ng); g++) start(g);
        for (int64_t g = 0; g < ng; g++) {
            if (g + NSLOT - 1 < ng) start(g + NSLOT - 1);
            const uint16_t * yy = D8.finish(sh8, (int) (g % NSLOT));
            const int64_t i0 = g * per;
            const int kk = (int) std::min<int64_t>(per, (int64_t) items.size() - i0);
            for (int i = 0; i < kk; i++) {
                const Item & it = items[i0 + i];
                gather8(m, yy, i * m.ncols, b->ys.data(), (int) it.p.size());
                for (size_t r = 0; r < it.p.size(); r++)
                    std::memcpy(y + (it.p[r].first * k + it.p[r].second) * m.N, b->ys.data() + (int64_t) r * m.N,
                                m.N * sizeof(float));
            }
        }
        return;
    }
    Device & D = device(m.sym);
    Shape * sh = D.shape(m.tiles, m.kc);
    b->xs.resize((size_t) NC * m.ks);
    b->ys.resize((size_t) per * m.N);
    for (int64_t t = 0; t < T; t++)
        for (int e0 = 0; e0 < k; e0 += per) {
            const int kk = std::min(per, k - e0);
            int64_t offs[NC];
            for (int i = 0; i < per; i++) {
                const int j = std::min(i, kk - 1);
                const int64_t e = ids[t * k + e0 + j];
                x_cols(m, xin(t, e0 + j), b->xs.data(), i * m.ncols);
                for (int c = 0; c < m.ncols; c++) offs[i * m.ncols + c] = b->base + e * (int64_t) m.nbytes + (int64_t) c * m.cb;
            }
            const uint16_t * yy = D.dyn(b->bo, offs, sh, b->xs.data(), m.ks);
            gather(m, yy, 0, kk, b->ys.data());
            std::memcpy(y + (t * k + e0) * m.N, b->ys.data(), (size_t) kk * m.N * sizeof(float));
        }
}

void onpu_moe_act(const float * h, int64_t rows, int64_t Ng, const int32_t * ids, const float * gu_bias, int act,
                  float alpha, float limit, float * a);   // onpu_cpu.cpp
void onpu_moe_sum(const float * o, int64_t T, int k, int64_t N, const int32_t * ids, const float * w,
                  const float * bias, float * y);

// A routed-expert FFN in one call (decode and prompts): for token t and slot j (expert e = ids[t*k+j]),
// h = gu_e . x_t (+ gu_bias[e]); a = act(h); y_t += w[t*k+j] * (dn_e . a (+ dn_bias[e])).
// act 0: silu(h[:I]) * h[I:] (Qwen, Llama-style experts); act 1: gpt-oss, interleaved gate/up,
// clamped at limit, gate * sigmoid(alpha * gate) * (up + 1). y [T][N of dn] is overwritten.
void onpu_moe_ffn(void * gu, void * dn, const float * x, int64_t T, const int32_t * ids, int k, const float * w,
                  int act, float alpha, float limit, const float * gu_bias, const float * dn_bias, float * y) {
    const int64_t Ng = ((BankH *) gu)->m.N, Nd = ((BankH *) dn)->m.N, I = Ng / 2;
    std::vector<float> h((size_t) T * k * Ng), a((size_t) T * k * I), o((size_t) T * k * Nd);
    onpu_bank_run(gu, x, 0, T, ids, k, h.data());
    const std::vector<int32_t> & perm = ((BankH *) gu)->perm;
    if (!perm.empty()) {                       // fused layout: back to [gate | up] order
        std::vector<float> row(Ng);
        for (int64_t r = 0; r < T * k; r++) {
            float * hr = h.data() + r * Ng;
            for (int64_t i = 0; i < Ng; i++) row[perm[i]] = hr[i];
            std::memcpy(hr, row.data(), Ng * sizeof(float));
        }
    }
    onpu_moe_act(h.data(), T * k, Ng, ids, gu_bias, act, alpha, limit, a.data());
    onpu_bank_run(dn, a.data(), 1, T, ids, k, o.data());
    onpu_moe_sum(o.data(), T, k, Nd, ids, w, dn_bias, y);
}

// ---- fused expert FFN
namespace {
int64_t txn_op_words(const uint32_t * w) {
    const uint32_t op = w[0] & 0xff;
    if (op == 0) return w[5] / 4;
    if (op == 1) return w[3] / 4;
    if (op == 3 || op == 4) return w[6] / 4;
    if (op >= 128) return w[1] / 4;
    return -1;
}
}  // namespace

void * onpu_ffn_new(int64_t Ng, int64_t Kg, int64_t Nd, int64_t Kd, int64_t E, int sym) {
    Mat mg(Ng, Kg, 1, sym != 0), md(Nd, Kd, 1, sym != 0);
    Device & D = device(sym != 0);
    if (!mg.ok || !md.ok || !D.has_shape(mg.tiles, mg.kc) || !D.has_shape(md.tiles, md.kc)) return nullptr;
    auto * f = new FfnH;
    f->gu = new BankH;
    f->dn = new BankH;
    f->gu->m = mg;
    f->dn->m = md;
    f->gu->E = f->dn->E = E;
    f->gu->bo = D.bo((size_t) (mg.nbytes + md.nbytes) * E, 0);
    f->dn->bo = f->gu->bo;
    f->gu->map = f->gu->bo.map<uint8_t *>();
    f->dn->base = (int64_t) mg.nbytes * E;
    f->dn->map = f->gu->map + f->dn->base;
    // the device writes column outputs in [t2][core][FLUSH*GN] order: pack gate_up rows so that order
    // is the original one ([gate | up]); prompts (two dispatches) undo it on the host
    const int64_t I = Ng / 2;
    f->fused = mg.R == 1 && mg.S == 1 && md.R == 1 && md.S == 1 && Ng % 2 == 0 && Kd == I
               && I % GK == 0 && (int64_t) mg.rows_col == Ng && Kg <= KMAX && mg.tiles % FLUSH == 0;
    f->gu->perm.assign(Ng, 0);
    for (int64_t r = 0; r < Ng; r++) f->gu->perm[r] = (int32_t) r;
    if (f->fused) {
        const int64_t t2n = mg.tiles / FLUSH;
        for (int64_t t2 = 0; t2 < t2n; t2++)
            for (int cr = 0; cr < NR; cr++)
                for (int i = 0; i < FLUSH * GN; i++) {
                    const int64_t p = (t2 * NR + cr) * FLUSH * GN + i;          // output position
                    const int64_t r = (int64_t) cr * mg.tiles * GN + t2 * FLUSH * GN + i;   // packed row
                    f->gu->perm[r] = (int32_t) p;
                }
    }
    return f;
}
void * onpu_ffn_gu(void * h) { return ((FfnH *) h)->gu; }
void * onpu_ffn_dn(void * h) { return ((FfnH *) h)->dn; }
int onpu_ffn_fused(void * h) { return ((FfnH *) h)->fused ? 1 : 0; }
void onpu_ffn_perm(void * h, int32_t * out) {
    const auto & p = ((FfnH *) h)->gu->perm;
    std::memcpy(out, p.data(), p.size() * sizeof(int32_t));
}
void onpu_ffn_free(void * h) {
    std::lock_guard<std::mutex> lk(g_mu);
    auto * f = (FfnH *) h;
    delete f->gu;
    delete f->dn;
    delete f;
}

namespace {
// both phases' insts joined into one stream: phase 1 (gate_up) writes its output into the input
// buffer (X, column c at c * KMAX * 2 bytes); phase 2 (down) runs the act expansion (kc | 0x100)
void ffn_build(FfnH * f, Device & D, FfnInsts & fi) {
    Shape * s1 = D.shape(f->gu->m.tiles, f->gu->m.kc);
    Shape * s2 = D.shape(f->dn->m.tiles, f->dn->m.kc);
    std::vector<uint32_t> w1(s1->base.begin(), s1->base.end()), w2(s2->base.begin(), s2->base.end());
    for (size_t i = 4; i < w1.size();) {             // retarget the output BDs (arg 2 = Y) to X
        const int64_t n = txn_op_words(&w1[i]);
        ONPU_ASSERT(n > 0);
        if ((w1[i] & 0xff) == 129 && w1[i + 8] == 2) {
            const uint32_t col = w1[i + 6] >> 25;
            w1[i + 8] = 1;
            w1[i + 10] = col * KMAX * D.T * 2;
        }
        i += n;
    }
    // down phase: the activation flag on the kc loop count. The insts start with the loop counts'
    // writes (go, tiles/2, kc/2, kc per core); their address depends on the build (1-token: 0xF100..,
    // 8-token: 0x3800..), so take the 4th register written
    std::vector<uint32_t> regs;
    for (size_t i = 4; i < w2.size() && regs.size() < 4;) {
        if ((w2[i] & 0xff) == 0) {
            const uint32_t r = w2[i + 2] & 0xfffff;
            if (std::find(regs.begin(), regs.end(), r) == regs.end()) regs.push_back(r);
        }
        i += txn_op_words(&w2[i]);
    }
    ONPU_ASSERT(regs.size() == 4);
    int flagged = 0;
    for (size_t i = 4; i < w2.size();) {
        const int64_t n = txn_op_words(&w2[i]);
        ONPU_ASSERT(n > 0);
        if ((w2[i] & 0xff) == 0 && (w2[i + 2] & 0xfffff) == regs[3]) {
            ONPU_ASSERT(w2[i + 4] == (uint32_t) f->dn->m.kc);
            w2[i + 4] |= 0x100;
            flagged++;
        }
        i += n;
    }
    ONPU_ASSERT(flagged == NC * NR);
    fi.base = w1;
    fi.base.insert(fi.base.end(), w2.begin() + 4, w2.end());
    fi.base[2] = w1[2] + w2[2];
    fi.base[3] = (uint32_t) (fi.base.size() * 4);
    const size_t shift = w1.size() - 4;
    for (size_t i = 0; i < s1->idx.size(); i++) {
        fi.idx.push_back(s1->idx[i]); fi.d.push_back(s1->d[i]); fi.col.push_back(s1->col[i]); fi.phase.push_back(0);
    }
    for (size_t i = 0; i < s2->idx.size(); i++) {
        fi.idx.push_back(s2->idx[i] + shift); fi.d.push_back(s2->d[i]); fi.col.push_back(s2->col[i]); fi.phase.push_back(1);
    }
    fi.words = fi.base;
    fi.ib = xrt::bo(D.dev, fi.words.size() * 4, xrt::bo::flags::cacheable, D.kern.group_id(1));
}

// the joined insts for these experts (one per column) -> fi.ib
void ffn_patch(FfnH * f, FfnInsts & fi, const int64_t * experts) {
    int64_t off[2][NC];
    for (int c = 0; c < NC; c++) {
        off[0][c] = f->gu->base + experts[c] * (int64_t) f->gu->m.nbytes;
        off[1][c] = f->dn->base + experts[c] * (int64_t) f->dn->m.nbytes;
    }
    for (size_t i = 0; i < fi.idx.size(); i++)
        fi.words[fi.idx[i]] = (uint32_t) ((int64_t) fi.base[fi.idx[i]] + fi.d[i] * off[fi.phase[i]][fi.col[i]] / DELTA);
    fi.ib.write(fi.words.data());
    fi.ib.sync(XCL_BO_SYNC_BO_TO_DEVICE);
}

xrt::run ffn_run_obj(FfnH * f, Device & D, FfnInsts & fi, Shape * s2) {
    xrt::run run(D.kern);
    run.set_arg(0, 3);
    run.set_arg(1, fi.ib);
    run.set_arg(2, (uint32_t) fi.words.size());
    run.set_arg(3, f->gu->bo);
    run.set_arg(4, D.x);
    run.set_arg(5, s2->y);
    return run;
}
}  // namespace

// one token: y [N of down] = sum_j w[j] * FFN_{ids[j]}(x), the 8 experts in one dispatch
int onpu_ffn_run(void * h, const float * x, const int32_t * ids, int k, const float * w, float * y) {
    auto * f = (FfnH *) h;
    if (!f->fused || k != NC) return 0;
    const Mat & mg = f->gu->m, & md = f->dn->m;
    Device & D = device(mg.sym);
    std::lock_guard<std::mutex> lk(g_mu);
    if (f->one.base.empty()) ffn_build(f, D, f->one);
    int64_t ex[NC];
    for (int c = 0; c < NC; c++) ex[c] = ids[c];
    ffn_patch(f, f->one, ex);
    f->xs.resize((size_t) NC * mg.ks);
    for (int c = 0; c < NC; c++) x_cols(mg, x, f->xs.data(), c);
    Shape * s2 = D.shape(md.tiles, md.kc);
    xrt::run run = ffn_run_obj(f, D, f->one, s2);
    const uint16_t * yy = D.exec(run, s2, f->xs.data(), mg.ks);
    f->ys.resize((size_t) NC * md.N);
    gather(md, yy, 0, NC, f->ys.data());
    std::fill(y, y + md.N, 0.f);
    for (int c = 0; c < NC; c++) {
        const float * o = f->ys.data() + (int64_t) c * md.N;
        for (int64_t i = 0; i < md.N; i++) y[i] += w[c] * o[i];
    }
    return 1;
}

// several tokens (T > 1, the 8-token device): tokens grouped per expert (<= 8 per group), 8 groups
// (one per column) per dispatch, both phases in it. y [T][N] = sum_j w[t][j] * FFN_{ids[t][j]}(x_t).
// 0 if this FFN can't be fused or the 8-token device lacks a shape (caller falls back).
int onpu_ffn_run_many(void * h, const float * x, int64_t T, const int32_t * ids, int k, const float * w, float * y) {
    auto * f = (FfnH *) h;
    const Mat & mg = f->gu->m, & md = f->dn->m;
    Device & D8 = device8(mg.sym);
    if (!f->fused || !D8.has_shape(mg.tiles, mg.kc) || !D8.has_shape(md.tiles, md.kc)) return 0;
    std::lock_guard<std::mutex> lk(g_mu);
    if (f->eight.base.empty()) ffn_build(f, D8, f->eight);
    std::map<int64_t, std::vector<std::pair<int64_t, int>>> by;     // expert -> (token, slot)
    for (int64_t t = 0; t < T; t++)
        for (int j = 0; j < k; j++) by[ids[t * k + j]].push_back({t, j});
    struct Item { int64_t e; std::vector<std::pair<int64_t, int>> p; };
    std::vector<Item> items;
    for (auto & kv : by)
        for (size_t r0 = 0; r0 < kv.second.size(); r0 += BT)
            items.push_back({kv.first, std::vector<std::pair<int64_t, int>>(
                kv.second.begin() + r0, kv.second.begin() + std::min(kv.second.size(), r0 + BT))});
    std::fill(y, y + T * md.N, 0.f);
    Shape * s2 = D8.shape(md.tiles, md.kc);
    f->xs.assign((size_t) NC * mg.ks * BT, 0.f);
    f->ys.resize((size_t) BT * md.N);
    for (size_t i0 = 0; i0 < items.size(); i0 += NC) {
        const int kk = (int) std::min<size_t>(NC, items.size() - i0);
        int64_t ex[NC];
        for (int c = 0; c < NC; c++) {
            const Item & it = items[i0 + std::min(c, kk - 1)];
            ex[c] = it.e;
            const float * rows[BT];
            for (size_t r = 0; r < it.p.size(); r++) rows[r] = x + it.p[r].first * mg.K;
            x_cols8(mg, rows, (int) it.p.size(), f->xs.data(), c);
        }
        ffn_patch(f, f->eight, ex);
        xrt::run run = ffn_run_obj(f, D8, f->eight, s2);
        const uint16_t * yy = D8.exec(run, s2, f->xs.data(), mg.ks);
        for (int c = 0; c < kk; c++) {
            const Item & it = items[i0 + c];
            gather8(md, yy, c, f->ys.data(), (int) it.p.size());
            for (size_t r = 0; r < it.p.size(); r++) {
                const int64_t t = it.p[r].first;
                const float wr = w[t * k + it.p[r].second], * o = f->ys.data() + (int64_t) r * md.N;
                float * yt = y + t * md.N;
                for (int64_t n = 0; n < md.N; n++) yt[n] += wr * o[n];
            }
        }
    }
    return 1;
}

// pack an [N, K] float matrix (row stride K) for ncols columns into out (m.nbytes); sym: q4s
int64_t onpu_pack(const float * W, int64_t N, int64_t K, int ncols, int sym, uint8_t * out, int threads) {
    Mat m(N, K, ncols, sym != 0);
    if (!m.ok) return 0;
    pack_mat(m, W, K, out, threads);
    return (int64_t) m.nbytes;
}

// what the NPU will see for W [N, K] (row stride K): each 32-weight group quantized as pack would,
// dequantized into out (for a residual pass: W - out). K % 32 == 0.
void onpu_quant_dequant(const float * W, int64_t N, int64_t K, int sym, float * out, int threads) {
    auto rows = [&](int64_t r0, int64_t r1) {
        uint8_t q[32];
        for (int64_t r = r0; r < r1; r++)
            for (int64_t g = 0; g < K; g += 32) {
                const float * x = W + r * K + g;
                float * o = out + r * K + g;
                uint16_t s = 0, m = 0;
                if (sym) {
                    quant_sym32(x, q, s);
                    for (int i = 0; i < 32; i++) o[i] = ((float) q[i] - 8.0f) * bf2f(s);
                } else {
                    quant_asym32(x, q, s, m);
                    for (int i = 0; i < 32; i++) o[i] = (float) q[i] * bf2f(s) + bf2f(m);
                }
            }
    };
    const int nt = std::max(1, std::min<int>(threads, (int) (N / 64) + 1));
    std::vector<std::thread> th;
    for (int t = 0; t < nt; t++) th.emplace_back(rows, N * t / nt, N * (t + 1) / nt);
    for (auto & t : th) t.join();
}

}  // extern "C"
