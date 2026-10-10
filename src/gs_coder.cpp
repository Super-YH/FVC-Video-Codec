#include "gs_coder.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <memory>
#include <stdexcept>

#include "block_coder.hpp"
#include "fvc/quant.hpp"

namespace fvc {

namespace {

constexpr int kQgMin = -40, kQgMax = 140;
constexpr int kNumPn = 6;

// 構文と再構成の両方が参照する状態 (利得予測・コピー元の有無)
struct GsState {
    int qg1 = 0, qg2 = 0, ngain = 0;
    int nvec = 0;  // 非ゼロの再構成パーティション数 (コピー元の数)
    bool has_vec = false;
    int prev_mode = 0;
    int band_qg[16] = {};
    bool band_set[16] = {};
    const GsMem* mem = nullptr;
};

// 前変換 (type 0 の pn): y = sc·f(x)
double pn_fwd(int pn, double x) {
    switch (pn) {
    case 1: return pre_forward(PreNonlin::Power, x, 0.75);
    case 2: return pre_forward(PreNonlin::Power, x, 0.6);
    case 3: return 2.0 * pre_forward(PreNonlin::Asinh, x, 2.0);
    case 4: return 1.5 * pre_forward(PreNonlin::SignedLog, x, 1.0);
    case 5: return pre_forward(PreNonlin::Pwl, x, 0.0);
    default: return x;
    }
}
double pn_inv(int pn, double y) {
    switch (pn) {
    case 1: return pre_inverse(PreNonlin::Power, y, 0.75);
    case 2: return pre_inverse(PreNonlin::Power, y, 0.6);
    case 3: return pre_inverse(PreNonlin::Asinh, y / 2.0, 2.0);
    case 4: return pre_inverse(PreNonlin::SignedLog, y / 1.5, 1.0);
    case 5: return pre_inverse(PreNonlin::Pwl, y, 0.0);
    default: return y;
    }
}
// pn 1 は非線形スカラ量子化器 (§5.5(c), べき乗コンパンディング) そのもの
const NonlinearSQ kNlsq{0.75, 1.0, 2.0 / 3.0, 0.0};
int32_t pn_quant(int pn, double x) {
    if (pn == 1) return kNlsq.quantize(x);
    const double y = pn_fwd(pn, x);
    const int32_t q = static_cast<int32_t>(std::abs(y) + 1.0 / 3.0);
    return y < 0 ? -q : q;
}
double pn_dequant(int pn, int32_t q) { return pn == 1 ? kNlsq.dequantize(q) : pn_inv(pn, static_cast<double>(q)); }

const RandomProjection& rproj(int lp) {
    thread_local std::map<int, std::unique_ptr<RandomProjection>> cache;
    auto& p = cache[lp];
    if (!p) p.reset(new RandomProjection((lp + 1) / 2, lp, 0x5EEDull * 1000003ull + static_cast<uint64_t>(lp)));
    return *p;
}

int sym_count(int type, int lp) { return type == 0 ? lp : type == 1 ? (lp + 7) / 8 * 8 : (lp + 1) / 2; }

void normalize_to(std::vector<double>& v, double g) {
    double e = 0;
    for (double a : v) e += a * a;
    if (e <= 0) { std::fill(v.begin(), v.end(), 0.0); return; }
    const double k = g / std::sqrt(e);
    for (double& a : v) a *= k;
}

// 1 パーティションの構文
void unit_syntax(SymIO& io, CMModel& m, GsUnit& u, int band, int lp, uint32_t pc, GsState& st) {
    const uint32_t bc = static_cast<uint32_t>(std::min(band, 7));
    const int nz = io.bit(m, 1, bc, pc * 8 + static_cast<uint32_t>(std::min(st.prev_mode, 5)), u.mode != 0);
    int mode = 0;
    if (nz) {
        const int v = u.mode - 1;
        uint32_t tree = 1;
        int val = 0;
        for (int b = 2; b >= 0; --b) {
            const int bt = io.bit(m, 2 + tree, bc, pc, (v >> b) & 1);
            tree = (tree << 1) | static_cast<uint32_t>(bt);
            val = (val << 1) | bt;
        }
        if (val > 4) throw std::runtime_error("corrupt stream: gs mode");
        mode = val + 1;
    }
    if (mode == 4 && !st.has_vec) throw std::runtime_error("corrupt stream: gs copy");
    const int bi = std::min(band, 15);
    u.mode = static_cast<uint8_t>(mode);
    if (mode == 1) u.gon = static_cast<uint8_t>(io.bit(m, 1, 66, pc * 8 + bc, u.gon));
    const bool has_gain = (mode >= 2 && mode <= 4) || (mode == 1 && u.gon);
    if (has_gain) {
        // 利得予測: 同じバンドの直前 > 前ブロックの同じバンド > 直前の利得 > 既定
        int pred, srcc;
        if (st.band_set[bi]) { pred = st.band_qg[bi]; srcc = 0; }
        else if (st.mem && st.mem->v[bi]) { pred = st.mem->qg[bi]; srcc = 1; }
        else if (st.ngain) { pred = st.qg1; srcc = 2; }
        else { pred = 8; srcc = 3; }
        u.qg = pred + io.sint(m, 32 + bc, pc + 2 * static_cast<uint32_t>(srcc), u.qg - pred);
        if (u.qg < kQgMin || u.qg > kQgMax) throw std::runtime_error("corrupt stream: gs gain");
    } else if (mode == 5) {
        u.qg = st.ngain >= 2 ? std::clamp(2 * st.qg1 - st.qg2, kQgMin, st.qg1) : st.ngain ? std::max(kQgMin, st.qg1 - 4) : 0;
    }
    if (has_gain || mode == 5) { st.qg2 = st.qg1; st.qg1 = u.qg; ++st.ngain; st.band_qg[bi] = u.qg; st.band_set[bi] = true; }
    if (mode >= 1) { st.has_vec = true; ++st.nvec; }
    if (mode == 1) {
        const int t1 = io.bit(m, 1, 64, pc, u.type != 0);
        u.type = static_cast<uint8_t>(t1 ? 1 + io.bit(m, 2, 64, pc, u.type == 2) : 0);
        if (u.type == 0) {
            uint32_t tree = 1;
            int val = 0;
            for (int b = 2; b >= 0; --b) {
                const int bt = io.bit(m, 10 + tree, 64, pc, (u.pn >> b) & 1);
                tree = (tree << 1) | static_cast<uint32_t>(bt);
                val = (val << 1) | bt;
            }
            if (val >= kNumPn) throw std::runtime_error("corrupt stream: gs prenl");
            u.pn = static_cast<uint8_t>(val);
        }
        const int ns = sym_count(u.type, lp);
        if (!io.enc) u.sym.assign(ns, 0);
        int prev = 0;
        for (int k = 0; k < ns; ++k) {
            // 文脈: 種類・位置・バンド・直前シンボルの大きさ
            u.sym[k] = io.sint(m, 128 + u.type * 16u + static_cast<uint32_t>(std::min(k, 15)),
                               pc * 8 + bc + 16u * static_cast<uint32_t>(std::min(prev, 3)), u.sym[k]);
            prev = std::abs(u.sym[k]);
            if (std::abs(u.sym[k]) > (1 << 16)) throw std::runtime_error("corrupt stream: gs symbol");
        }
    } else if (mode == 3) {
        if (!io.enc) u.sym.assign(lp, 0);
        for (int k = 0; k < lp; ++k) u.sym[k] = io.bit(m, 1, 96, pc, u.sym[k]);
    } else if (mode == 4) {
        const int hi = io.bit(m, 1, 65, pc, u.perm >> 1);
        const int lo = io.bit(m, 2 + static_cast<uint32_t>(hi), 65, pc, u.perm & 1);
        u.perm = static_cast<uint8_t>(hi * 2 + lo);
        if (st.nvec > 1) {
            const int s1 = io.bit(m, 1, 67, pc, u.src != 0);
            int sv = 0;
            if (s1) { sv = 1; while (sv < std::min(st.nvec, 4) - 1 && io.bit(m, 1 + static_cast<uint32_t>(sv), 68, pc, u.src > sv)) ++sv; }
            u.src = static_cast<uint8_t>(sv);
        } else {
            u.src = 0;
        }
    }
    st.prev_mode = mode;
}

// 1 パーティションの再構成 (ステップ単位)。last はコピー元 (非ゼロの直前パーティション) を保持
using GsHist = std::vector<std::vector<double>>;  // mode>=1 のパーティション再構成 (新しい順に最大 4)
void unit_recon(const GsUnit& u, const std::vector<double>& rho, uint64_t seed, GsHist& hist, std::vector<double>& out) {
    const int lp = static_cast<int>(rho.size());
    out.assign(lp, 0.0);
    if (u.mode == 0) return;
    const double g = std::exp2(u.qg / 4.0);
    switch (u.mode) {
    case 1:
        if (u.type == 0) {
            for (int k = 0; k < lp; ++k) out[k] = pn_dequant(u.pn, u.sym[k]);
        } else if (u.type == 1) {
            for (int k = 0; k < lp; ++k) out[k] = u.sym[k] * 0.5;
        } else {
            const RandomProjection& P = rproj(lp);
            std::vector<double> y(P.m());
            for (int i = 0; i < P.m(); ++i) y[i] = u.sym[i];
            P.reconstruct(y.data(), out.data());
        }
        break;
    case 2:
    case 5: {
        SplitMix64 rng(seed);
        for (int k = 0; k < lp; ++k) out[k] = rng.uniform() - 0.5;
        break;
    }
    case 3:
        for (int k = 0; k < lp; ++k) out[k] = (u.sym[k] ? -1.0 : 1.0) / std::max(1.0, rho[k]);
        break;
    case 4: {
        if (static_cast<int>(hist.size()) <= u.src) break;
        const std::vector<double>& last = hist[u.src];
        const int n = static_cast<int>(last.size());
        if (!n) break;
        for (int k = 0; k < lp; ++k) {
            int j = k;
            if (u.perm == 1) j = lp - 1 - k;
            else if (u.perm == 2) j = (k + lp / 2) % lp;
            else if (u.perm == 3) j = (k ^ 1) < lp ? (k ^ 1) : k;
            out[k] = last[static_cast<size_t>(j) * n / lp];
        }
        break;
    }
    default: break;
    }
    if (!(u.mode == 1 && !u.gon)) normalize_to(out, g);
    hist.insert(hist.begin(), out);
    if (hist.size() > 4) hist.pop_back();
}

uint64_t unit_seed(uint64_t seed, size_t i) { return seed * 0x9E3779B97F4A7C15ull + i * 0xD1B54A32D192ED03ull + 1; }

int qg_of(double g) { return std::clamp(static_cast<int>(std::lround(4.0 * std::log2(std::max(g, 1e-9)))), kQgMin, kQgMax); }

}  // namespace

const GsLayout& gs_layout(int l) {
    static const std::array<GsLayout, 11> cache = [] {
        std::array<GsLayout, 11> c;
        for (int ll = 0; ll <= 10; ++ll) {
            const int s = 1 << ll;
            const auto& scan = diag_scan(ll);
            std::map<int, std::vector<std::pair<int, double>>> bands;  // band -> (走査順 idx, ρ)
            for (int i = 1; i < s * s; ++i) {
                const int p = scan[i], u = p & (s - 1), v = p >> ll;
                const double r = std::sqrt(static_cast<double>(u * u + v * v));
                const int b = 1 + static_cast<int>(std::floor(std::log2(r) + 1e-9));
                bands[b].push_back({i, r});
            }
            for (auto& [b, list] : bands) {
                for (size_t k = 0; k < list.size(); k += kGsPart) {
                    std::vector<int> id;
                    std::vector<double> rh;
                    for (size_t j = k; j < std::min(list.size(), k + kGsPart); ++j) { id.push_back(list[j].first); rh.push_back(list[j].second); }
                    c[ll].idx.push_back(id);
                    c[ll].rho.push_back(rh);
                    c[ll].band.push_back(b);
                }
            }
        }
        return c;
    }();
    return cache[l];
}

bool gs_nonzero(const GsBlock& g) {
    if (g.dc) return true;
    for (const GsUnit& u : g.u) if (u.mode) return true;
    return false;
}

void gs_syntax(SymIO& io, CMModel& m, GsBlock& g, int l, uint32_t pc, GsMem& mem) {
    const GsLayout& L = gs_layout(l);
    const int nu = static_cast<int>(L.idx.size());
    g.dc = io.sint(m, 300, pc, g.dc);
    if (std::abs(g.dc) > (1 << 20)) throw std::runtime_error("corrupt stream: gs dc");
    int used = 0;
    if (io.enc) for (int i = 0; i < static_cast<int>(g.u.size()); ++i) if (g.u[i].mode) used = i + 1;
    used = static_cast<int>(io.uintc(m, 301, pc * 16 + static_cast<uint32_t>(l), static_cast<uint32_t>(used)));
    if (used > nu) throw std::runtime_error("corrupt stream: gs units");
    if (!io.enc) g.u.assign(nu, GsUnit{});
    else g.u.resize(nu);
    GsState st;
    st.mem = &mem;
    for (int i = 0; i < used; ++i) unit_syntax(io, m, g.u[i], L.band[i], static_cast<int>(L.idx[i].size()), pc, st);
    for (int i = used; i < nu; ++i) g.u[i] = GsUnit{};
    if (!io.cost)  // 実際の符号化/復号のときだけ次ブロック用の利得を更新 (レート推定では変えない)
        for (int b = 0; b < 16; ++b) if (st.band_set[b]) { mem.qg[b] = st.band_qg[b]; mem.v[b] = true; }
}

void gs_reconstruct(const GsBlock& g, int l, uint64_t seed, double* xh) {
    const GsLayout& L = gs_layout(l);
    const int n = 1 << (2 * l);
    std::fill(xh, xh + n, 0.0);
    xh[0] = g.dc;
    GsHist last;
    std::vector<double> v;
    for (size_t i = 0; i < L.idx.size() && i < g.u.size(); ++i) {
        unit_recon(g.u[i], L.rho[i], unit_seed(seed, i), last, v);
        for (size_t k = 0; k < v.size(); ++k) xh[L.idx[i][k]] = v[k];
    }
}

void gs_encode(const double* x, int l, double lam, CMModel& m, uint32_t pc, uint64_t seed, const GsMem& mem, GsBlock& out) {
    const GsLayout& L = gs_layout(l);
    const int nu = static_cast<int>(L.idx.size());
    out.u.assign(nu, GsUnit{});
    out.dc = static_cast<int32_t>(std::abs(x[0]) + 0.5);
    if (x[0] < 0) out.dc = -out.dc;
    GsState st;
    st.mem = &mem;
    GsHist last;
    std::vector<double> v, xu;
    auto rate = [&](GsUnit& u, int i, GsState s2) {
        double c = 0;
        SymIO io;
        io.cost = &c;
        io.enc = true;
        unit_syntax(io, m, u, L.band[i], static_cast<int>(L.idx[i].size()), pc, s2);
        return c;
    };
    for (int i = 0; i < nu; ++i) {
        const int lp = static_cast<int>(L.idx[i].size());
        xu.resize(lp);
        double ex = 0, amax = 0;
        for (int k = 0; k < lp; ++k) { xu[k] = x[L.idx[i][k]]; ex += xu[k] * xu[k]; amax = std::max(amax, std::abs(xu[k])); }
        const uint64_t us = unit_seed(seed, static_cast<size_t>(i));
        GsUnit best;
        double bj = ex + lam * rate(best, i, st);
        GsHist best_last = last;
        auto consider = [&](GsUnit u) {
            if (u.mode == 5) rate(u, i, st);  // 包絡補間の利得は構文側で導出される
            GsHist lt = last;
            unit_recon(u, L.rho[i], us, lt, v);
            double d = 0;
            for (int k = 0; k < lp; ++k) { const double e = xu[k] - v[k]; d += e * e; }
            if (d >= bj) return;
            const double j = d + lam * rate(u, i, st);
            if (j < bj) { bj = j; best = u; best_last = lt; }
        };
        // 利得: 形状 ŝ に対する MSE 最適 ĝ = <x, ŝ/‖ŝ‖> を対数量子化
        auto with_gain = [&](GsUnit u) {
            if (u.mode == 1) { GsUnit z = u; z.gon = 0; consider(z); }
            GsHist lt = last;
            u.qg = 0;
            unit_recon(u, L.rho[i], us, lt, v);  // 利得 1 の形状
            double ip = 0;
            for (int k = 0; k < lp; ++k) ip += xu[k] * v[k];
            if (ip <= 0) return;
            u.qg = qg_of(ip);
            consider(u);
            if (u.qg > kQgMin) { --u.qg; consider(u); }
        };
        if (amax >= 0.45) {
            for (int pn = 0; pn < kNumPn; ++pn) {
                GsUnit u; u.mode = 1; u.type = 0; u.pn = static_cast<uint8_t>(pn);
                u.sym.resize(lp);
                bool nzs = false;
                for (int k = 0; k < lp; ++k) { u.sym[k] = pn_quant(pn, xu[k]); nzs |= u.sym[k] != 0; }
                if (nzs) with_gain(u);
            }
            {   // E8 格子 VQ (§5.5(a))
                GsUnit u; u.mode = 1; u.type = 1;
                const int ns = sym_count(1, lp);
                u.sym.assign(ns, 0);
                bool nzs = false;
                for (int c = 0; c < ns; c += 8) {
                    double a[8], y[8];
                    for (int k = 0; k < 8; ++k) a[k] = c + k < lp ? xu[c + k] : 0.0;
                    nearest_e8(a, y);
                    for (int k = 0; k < 8; ++k) { u.sym[c + k] = static_cast<int32_t>(std::lround(2.0 * y[k])); nzs |= u.sym[c + k] != 0; }
                }
                if (nzs) with_gain(u);
            }
            if (lp >= 4) {  // ランダム投影 (§5.5(b))
                const RandomProjection& P = rproj(lp);
                std::vector<double> y(P.m());
                P.project(xu.data(), y.data());
                GsUnit u; u.mode = 1; u.type = 2;
                u.sym.resize(P.m());
                bool nzs = false;
                for (int k = 0; k < P.m(); ++k) {
                    const int32_t q = static_cast<int32_t>(std::abs(y[k]) + 0.5);
                    u.sym[k] = y[k] < 0 ? -q : q;
                    nzs |= q != 0;
                }
                if (nzs) with_gain(u);
            }
            {   // 符号のみ (§5.6-4)
                GsUnit u; u.mode = 3;
                u.sym.resize(lp);
                for (int k = 0; k < lp; ++k) u.sym[k] = xu[k] < 0;
                with_gain(u);
            }
            if (st.has_vec && !last.empty())  // 係数コピー (コピー元 4 候補 = 周波数領域コピー) + 並び替え (§5.6-2, §7.6-3/4)
                for (int sc = 0; sc < std::min<int>(4, static_cast<int>(last.size())); ++sc)
                    for (int pm = 0; pm < 4; ++pm) {
                        GsUnit u; u.mode = 4; u.perm = static_cast<uint8_t>(pm); u.src = static_cast<uint8_t>(sc);
                        with_gain(u);
                    }
        }
        if (ex > 0) {  // ノイズ置換 (利得は RMS) と包絡補間 (§5.6-1, -3)
            GsUnit u; u.mode = 2; u.qg = qg_of(std::sqrt(ex) * 0.5);
            consider(u);
            GsUnit w; w.mode = 5;
            consider(w);
        }
        {
            double c = 0;
            SymIO io; io.cost = &c; io.enc = true;
            unit_syntax(io, m, best, L.band[i], lp, pc, st);
        }
        last = best_last;
        out.u[i] = best;
    }
}

}  // namespace fvc
