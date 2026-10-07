#include "block_coder.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "fvc/quant.hpp"

namespace fvc {

// ---------------- 共通ユーティリティ ----------------
const std::vector<int>& diag_scan(int log2s) {
    static std::array<std::vector<int>, 11> cache = [] {
        std::array<std::vector<int>, 11> c;
        for (int l = 0; l <= 10; ++l) {
            const int s = 1 << l;
            for (int d = 0; d <= 2 * (s - 1); ++d)
                for (int v = std::min(d, s - 1); v >= 0; --v) {
                    const int u = d - v;
                    if (u < s) c[l].push_back(v * s + u);
                }
        }
        return c;
    }();
    return cache[log2s];
}

double qp_step(int qp, int bit_depth) { return std::pow(2.0, (qp - 4) / 6.0) * (1 << (bit_depth - 8)); }

int32_t quant_dz(double c, double step, double rnd) {
    const int32_t q = static_cast<int32_t>(std::abs(c) / step + rnd);
    return c < 0 ? -q : q;
}

namespace {
// HEVC 互換の角度表 (モード 2..34)
constexpr int kAngle[35] = {0, 0, 32, 26, 21, 17, 13, 9, 5, 2, 0, -2, -5, -9, -13, -17, -21, -26,
                            -32, -26, -21, -17, -13, -9, -5, -2, 0, 2, 5, 9, 13, 17, 21, 26, 32};
int inv_angle(int a) { return (256 * 32 + (a < 0 ? -a : a) / 2) / a; }  // a<0 のみ使用 (負値)

int ilog2(int s) { int l = 0; while ((1 << l) < s) ++l; return l; }
}  // namespace

BlockCoder::BlockCoder(Plane* rec, const Plane* org, const Plane* luma, int plane, int32_t lo, int32_t hi, double step,
                       double lambda, int min_log2, int max_log2, const Tools& tools, const Search& search)
    : rec_(rec), org_(org), luma_(luma), plane_(plane), lo_(lo), hi_(hi), mid_((lo + hi + 1) / 2), step_(step),
      lambda_(lambda), min_log2_(min_log2), max_log2_(max_log2), tools_(tools), search_(search) {
    if (!luma_) tools_.cfl = false;
    modes4_.assign(static_cast<size_t>(rec_->w / 4) * (rec_->h / 4), static_cast<int8_t>(kModePlanar));
    split_map_.resize(max_log2_ + 1);
    leaf_map_.resize(max_log2_ + 1);
    if (org_)
        for (int l = min_log2_; l <= max_log2_; ++l) {
            split_map_[l].assign(static_cast<size_t>(rec_->w >> l) * (rec_->h >> l), 0);
            leaf_map_[l].resize(static_cast<size_t>(rec_->w >> l) * (rec_->h >> l));
        }
}

// ---------------- 予測 ----------------
void BlockCoder::mpm(int x0, int y0, int& m0, int& m1) const {
    const int gw = rec_->w / 4;
    const int left = x0 > 0 ? modes4_[(y0 / 4) * gw + (x0 / 4 - 1)] : kModePlanar;
    const int above = y0 > 0 ? modes4_[(y0 / 4 - 1) * gw + (x0 / 4)] : kModePlanar;
    m0 = left;
    m1 = above != left ? above : (left != kModePlanar ? kModePlanar : kModeDC);
}

void BlockCoder::set_modes4(int x0, int y0, int s, int mode) {
    const int gw = rec_->w / 4;
    const int m = (mode == kModeCfl || mode < 0) ? kModePlanar : mode;
    for (int y = y0 / 4; y < (y0 + s) / 4; ++y)
        for (int x = x0 / 4; x < (x0 + s) / 4; ++x) modes4_[y * gw + x] = static_cast<int8_t>(m);
}

void BlockCoder::intra_angular(int x0, int y0, int s, int mode, int32_t* pred) const {
    const Plane& r = *rec_;
    const bool ht = y0 > 0, hl = x0 > 0;
    // 参照: corner, top[0..2s), left[0..2s)。右上/左下は端値で延長
    std::vector<int32_t> top(2 * s), left(2 * s);
    for (int i = 0; i < 2 * s; ++i) {
        const int ii = std::min(i, s - 1);
        top[i] = ht ? r.at(x0 + ii, y0 - 1) : (hl ? r.at(x0 - 1, y0) : mid_);
        left[i] = hl ? r.at(x0 - 1, y0 + ii) : (ht ? r.at(x0, y0 - 1) : mid_);
    }
    const int32_t corner = (ht && hl) ? r.at(x0 - 1, y0 - 1) : (ht ? top[0] : left[0]);
    if (mode == kModeDC) {
        int64_t sum = 0;
        for (int i = 0; i < s; ++i) sum += top[i] + left[i];
        const int32_t dc = static_cast<int32_t>((sum + s) / (2 * s));
        for (int i = 0; i < s * s; ++i) pred[i] = dc;
        return;
    }
    if (mode == kModePlanar) {
        const int sh = ilog2(s);
        for (int y = 0; y < s; ++y)
            for (int x = 0; x < s; ++x) {
                const int64_t h = static_cast<int64_t>(s - 1 - x) * left[y] + static_cast<int64_t>(x + 1) * top[s];
                const int64_t v = static_cast<int64_t>(s - 1 - y) * top[x] + static_cast<int64_t>(y + 1) * left[s];
                pred[y * s + x] = static_cast<int32_t>((h + v + s) >> (sh + 1));
            }
        return;
    }
    const bool vert = mode >= 18;
    const int ang = kAngle[mode];
    const std::vector<int32_t>& main = vert ? top : left;
    const std::vector<int32_t>& side = vert ? left : top;
    // ref[k + s] , k ∈ [-s, 2s]
    std::vector<int32_t> ref(3 * s + 1);
    ref[s] = corner;
    for (int i = 0; i < 2 * s; ++i) ref[s + 1 + i] = main[i];
    if (ang < 0) {
        const int ia = inv_angle(ang);
        const int kmin = (s * ang) >> 5;
        for (int k = kmin; k < 0; ++k) {
            const int j = (k * ia + 128) >> 8;  // side 上の位置 (1 始まり、0=corner)
            ref[s + k] = j <= 0 ? corner : side[std::min(j - 1, 2 * s - 1)];
        }
    }
    for (int y = 0; y < s; ++y) {
        const int pos = (y + 1) * ang;
        const int idx = pos >> 5, fact = pos & 31;
        for (int x = 0; x < s; ++x) {
            const int i0 = std::clamp(s + x + idx + 1, 0, 3 * s), i1 = std::clamp(s + x + idx + 2, 0, 3 * s);
            const int32_t v = ((32 - fact) * ref[i0] + fact * ref[i1] + 16) >> 5;
            if (vert) pred[y * s + x] = v;
            else pred[x * s + y] = v;
        }
    }
}

int BlockCoder::fit_cfl_alpha(int x0, int y0, int s) const {
    double ml = 0, mc = 0;
    for (int y = 0; y < s; ++y)
        for (int x = 0; x < s; ++x) { ml += luma_->at(x0 + x, y0 + y); mc += org_->at(x0 + x, y0 + y); }
    ml /= s * s; mc /= s * s;
    double sxy = 0, sxx = 0;
    for (int y = 0; y < s; ++y)
        for (int x = 0; x < s; ++x) {
            const double dl = luma_->at(x0 + x, y0 + y) - ml, dc = org_->at(x0 + x, y0 + y) - mc;
            sxy += dl * dc; sxx += dl * dl;
        }
    const double a = sxx > 0 ? sxy / sxx : 0.0;
    return std::clamp(static_cast<int>(std::lround(a * 8.0)), -16, 16);
}

void BlockCoder::intra_cfl(int x0, int y0, int s, int alpha, int32_t* pred) const {
    intra_angular(x0, y0, s, kModeDC, pred);
    int64_t sum = 0;
    for (int y = 0; y < s; ++y) for (int x = 0; x < s; ++x) sum += luma_->at(x0 + x, y0 + y);
    const int64_t n = static_cast<int64_t>(s) * s;
    const int32_t avg = static_cast<int32_t>(sum >= 0 ? (sum + n / 2) / n : -((-sum + n / 2) / n));
    for (int y = 0; y < s; ++y)
        for (int x = 0; x < s; ++x) {
            const int64_t d = static_cast<int64_t>(alpha) * (luma_->at(x0 + x, y0 + y) - avg);
            pred[y * s + x] += static_cast<int32_t>((d + 4) >> 3);
        }
}

void BlockCoder::predict(const Leaf& lf, int x0, int y0, int l, int32_t* pred) const {
    const int s = 1 << l;
    if (lf.pt == 1) {
        for (int y = 0; y < s; ++y)
            for (int x = 0; x < s; ++x) pred[y * s + x] = rec_->at(x0 + lf.bvx + x, y0 + lf.bvy + y);
    } else if (lf.mode == kModeCfl) {
        intra_cfl(x0, y0, s, lf.alpha, pred);
    } else {
        intra_angular(x0, y0, s, lf.mode, pred);
    }
}

// ---------------- IBC (§6.1) ----------------
bool BlockCoder::ibc_valid(int rx, int ry, int s) const {
    if (rx < 0 || ry < 0 || rx + s > rec_->w || ry + s > rec_->h) return false;
    if (ry + s <= cy_) return true;                                     // 上の CTU 行
    return rx + s <= cx_ && ry >= cy_ && ry + s <= cy_ + ctu_;          // 同 CTU 行の左 CTU
}

void BlockCoder::ibc_search(int x0, int y0, int s, int& bx, int& by) const {
    int64_t best = INT64_MAX;
    bx = by = 0;
    auto cost = [&](int dx, int dy, int64_t bound) {
        int64_t sad = 0;
        for (int y = 0; y < s && sad < bound; y += 2)
            for (int x = 0; x < s; x += 2) sad += std::abs(org_->at(x0 + x, y0 + y) - rec_->at(x0 + dx + x, y0 + dy + y));
        return sad;
    };
    auto test = [&](int dx, int dy) {
        if ((dx == 0 && dy == 0) || !ibc_valid(x0 + dx, y0 + dy, s)) return;
        const int64_t c = cost(dx, dy, best) + 4 * (std::abs(dx - bv_px_) + std::abs(dy - bv_py_)) / std::max(1, s / 8);
        if (c < best) { best = c; bx = dx; by = dy; }
    };
    test(bv_px_, bv_py_);
    const int R = search_.ibc_range;
    for (int dy = -R; dy <= 0; ++dy)
        for (int dx = -R; dx <= R; ++dx) test(dx, dy);
}

// ---------------- 量子化 / 再構成 ----------------
void BlockCoder::quantize(Leaf& lf, int x0, int y0, int l, const int32_t* pred) const {
    const int s = 1 << l, n = s * s;
    const auto& scan = diag_scan(l);
    std::vector<double> r(n), c(n), e(n);
    for (int y = 0; y < s; ++y)
        for (int x = 0; x < s; ++x) r[y * s + x] = org_->at(x0 + x, y0 + y) - pred[y * s + x];
    forward_2d(tx_for(s), tx_for(s), r.data(), s, s, c.data());
    for (int i = 0; i < n; ++i) e[i] = c[scan[i]];
    if (lf.tns_on) {
        lf.tns = tns_design(e.data(), n, 4, 4);
        if (lf.tns.order == 0) lf.tns_on = false;
        else { std::vector<double> t(n); tns_analysis(lf.tns, e.data(), n, t.data()); e.swap(t); }
    }
    lf.last = -1;
    lf.nf = 0;
    if (lf.qmode == 0) {
        lf.q.assign(n, 0);
        for (int i = 0; i < n; ++i) {
            const int32_t v = quant_dz(e[i], step_, 1.0 / 3.0);
            lf.q[scan[i]] = v;
            if (v) lf.last = i;
        }
        if (tools_.nf && lf.last + 1 < n) {
            double en = 0;
            for (int i = lf.last + 1; i < n; ++i) en += e[i] * e[i];
            const double rms = std::sqrt(en / (n - lf.last - 1));
            lf.nf = std::clamp(static_cast<int>(std::lround(rms / step_ * 8.0)), 0, 3);
        }
    } else {
        int lastsig = -1;
        for (int i = 0; i < n; ++i) if (std::abs(e[i]) >= 0.5 * step_) lastsig = i;
        const int nch = (lastsig + 8) / 8;
        lf.q.assign(static_cast<size_t>(nch) * 8, 0);
        int lastnz = -1;
        for (int ch = 0; ch < nch; ++ch) {
            double xs[8], ys[8];
            for (int k = 0; k < 8; ++k) { const int i = ch * 8 + k; xs[k] = i < n ? e[i] / step_ : 0.0; }
            nearest_e8(xs, ys);
            bool nz = false;
            for (int k = 0; k < 8; ++k) {
                lf.q[ch * 8 + k] = static_cast<int32_t>(std::lround(2.0 * ys[k]));
                nz |= lf.q[ch * 8 + k] != 0;
            }
            if (nz) lastnz = ch;
        }
        lf.q.resize(static_cast<size_t>(lastnz + 1) * 8);
        lf.last = lastnz;
    }
}

void BlockCoder::reconstruct(const Leaf& lf, int x0, int y0, int l, const int32_t* pred) {
    const int s = 1 << l, n = s * s;
    const auto& scan = diag_scan(l);
    std::vector<double> r(n, 0.0);
    if (lf.last >= 0 || lf.nf) {
        std::vector<double> e(n, 0.0), c(n);
        if (lf.qmode == 0) {
            for (int i = 0; i <= lf.last; ++i) e[i] = lf.q[scan[i]] * step_;
            if (lf.nf) {
                SplitMix64 rng(static_cast<uint64_t>(x0) * 0x10001ull + static_cast<uint64_t>(y0) * 0x9E37ull + plane_);
                const double amp = lf.nf / 8.0 * step_;
                for (int i = lf.last + 1; i < n; ++i) e[i] = (rng.next() >> 63) ? amp : -amp;
            }
        } else {
            for (size_t i = 0; i < lf.q.size() && static_cast<int>(i) < n; ++i) e[i] = lf.q[i] * 0.5 * step_;
        }
        if (lf.tns_on) { std::vector<double> t(n); tns_synthesis(lf.tns, e.data(), n, t.data()); e.swap(t); }
        for (int i = 0; i < n; ++i) c[scan[i]] = e[i];
        inverse_2d(tx_for(s), tx_for(s), c.data(), s, s, r.data());
    }
    for (int y = 0; y < s; ++y)
        for (int x = 0; x < s; ++x)
            rec_->at(x0 + x, y0 + y) = std::clamp(pred[y * s + x] + static_cast<int32_t>(std::lround(r[y * s + x])), lo_, hi_);
}

double BlockCoder::leaf_bits(const Leaf& lf, int x0, int y0, int l) const {
    double b = 1.0;  // cbf
    if (tools_.ibc) b += 0.5;
    if (lf.pt == 1) {
        b += 4.0 + 2.0 * std::log2(1.0 + std::abs(lf.bvx - bv_px_)) + 2.0 * std::log2(1.0 + std::abs(lf.bvy - bv_py_));
    } else if (lf.mode == kModeCfl) {
        b += 1.0 + 2.0 + std::log2(1.0 + std::abs(lf.alpha));
    } else {
        int m0, m1;
        mpm(x0, y0, m0, m1);
        b += (lf.mode == m0 || lf.mode == m1) ? 2.0 : (tools_.all_angular ? 6.5 : 3.0);
    }
    if (lf.last < 0) return b;
    if (tools_.e8) b += 1.0;
    if (lf.tns_on) b += 2.0 + 4.0 * lf.tns.order;
    if (lf.qmode == 0) {
        const auto& scan = diag_scan(l);
        b += 2.0 + 2.0 * std::log2(2.0 + lf.last);
        for (int i = 0; i <= lf.last; ++i) {
            const int32_t a = std::abs(lf.q[scan[i]]);
            b += a ? 2.0 + 2.0 * std::log2(1.0 + a) : 0.6;
        }
    } else {
        b += 2.0 + 2.0 * std::log2(2.0 + lf.last);
        for (int32_t v : lf.q) b += v ? 1.5 + 2.0 * std::log2(1.0 + std::abs(v) * 0.5) : 0.5;
        b += 0.8 * (lf.last + 1);
        b *= 1.4;  // 実測校正: E8 インデックスの実レートは近似式より約 4 割大きい
    }
    return b;
}

double BlockCoder::sse(int x0, int y0, int s) const {
    double e = 0;
    for (int y = 0; y < s; ++y)
        for (int x = 0; x < s; ++x) { const double d = org_->at(x0 + x, y0 + y) - rec_->at(x0 + x, y0 + y); e += d * d; }
    return e;
}

void BlockCoder::save(int x0, int y0, int s, std::vector<int32_t>& b) const {
    b.resize(static_cast<size_t>(s) * s);
    for (int y = 0; y < s; ++y) std::memcpy(&b[y * s], &rec_->v[(y0 + y) * rec_->w + x0], s * sizeof(int32_t));
}

void BlockCoder::restore(int x0, int y0, int s, const std::vector<int32_t>& b) {
    for (int y = 0; y < s; ++y) std::memcpy(&rec_->v[(y0 + y) * rec_->w + x0], &b[y * s], s * sizeof(int32_t));
}

// ---------------- RD 探索 (符号器) ----------------
double BlockCoder::rd_node(int x0, int y0, int l) {
    const int s = 1 << l, n = s * s;
    std::vector<int32_t> before, best_rec, pred(n);
    save(x0, y0, s, before);

    // 候補生成: 予測 SAD で絞り込み
    std::vector<Leaf> cands;
    {
        std::vector<std::pair<int64_t, int>> sads;
        std::vector<int> modes;
        if (tools_.all_angular) for (int m = 0; m < kNumIntra; ++m) modes.push_back(m);
        else modes = {kModeDC, kModePlanar, kModeHor, kModeVer};
        Leaf t;
        for (int m : modes) {
            t.mode = m;
            predict(t, x0, y0, l, pred.data());
            int64_t sad = 0;
            for (int y = 0; y < s; ++y)
                for (int x = 0; x < s; ++x) sad += std::abs(org_->at(x0 + x, y0 + y) - pred[y * s + x]);
            sads.push_back({sad, m});
        }
        std::sort(sads.begin(), sads.end());
        int m0, m1;
        mpm(x0, y0, m0, m1);
        std::vector<int> pick;
        for (size_t i = 0; i < sads.size() && static_cast<int>(pick.size()) < search_.rd_modes; ++i) pick.push_back(sads[i].second);
        for (int m : {m0, m1})
            if (std::find(pick.begin(), pick.end(), m) == pick.end() && std::find(modes.begin(), modes.end(), m) != modes.end())
                pick.push_back(m);
        for (int m : pick) { Leaf lf; lf.mode = m; cands.push_back(lf); }
        if (tools_.cfl) { Leaf lf; lf.mode = kModeCfl; lf.alpha = fit_cfl_alpha(x0, y0, s); cands.push_back(lf); }
        if (tools_.ibc && s >= 8 && s <= 32) {
            Leaf lf; lf.pt = 1;
            ibc_search(x0, y0, s, lf.bvx, lf.bvy);
            if (lf.bvx || lf.bvy) cands.push_back(lf);
        }
    }
    // 量子化モード/TNS の組み合わせ
    double best = 1e300;
    Leaf best_leaf;
    for (const Leaf& base : cands) {
        predict(base, x0, y0, l, pred.data());
        for (int qm = 0; qm <= (tools_.e8 && search_.try_e8 ? 1 : 0); ++qm)
            for (int tn = 0; tn <= (tools_.tns && search_.try_tns && s >= 8 ? 1 : 0); ++tn) {
                Leaf lf = base;
                lf.qmode = qm;
                lf.tns_on = tn;
                quantize(lf, x0, y0, l, pred.data());
                if (tn && !lf.tns_on) continue;
                reconstruct(lf, x0, y0, l, pred.data());
                const double j = sse(x0, y0, s) + lambda_ * (leaf_bits(lf, x0, y0, l) + (l > min_log2_ ? 1.0 : 0.0));
                if (j < best) { best = j; best_leaf = lf; save(x0, y0, s, best_rec); }
                restore(x0, y0, s, before);
            }
    }
    leaf_at(x0, y0, l) = best_leaf;
    restore(x0, y0, s, best_rec);
    set_modes4(x0, y0, s, best_leaf.pt ? kModePlanar : best_leaf.mode);
    if (l <= min_log2_) return best;
    if (best < lambda_ * 4.0) { split_at(x0, y0, l) = 0; return best; }
    restore(x0, y0, s, before);
    const int sbx = bv_px_, sby = bv_py_;
    const int h = s / 2;
    double js = lambda_;
    js += rd_node(x0, y0, l - 1);
    js += rd_node(x0 + h, y0, l - 1);
    js += rd_node(x0, y0 + h, l - 1);
    js += rd_node(x0 + h, y0 + h, l - 1);
    if (best <= js) {
        split_at(x0, y0, l) = 0;
        restore(x0, y0, s, best_rec);
        set_modes4(x0, y0, s, best_leaf.pt ? kModePlanar : best_leaf.mode);
        bv_px_ = sbx; bv_py_ = sby;
        if (best_leaf.pt == 1) { bv_px_ = best_leaf.bvx; bv_py_ = best_leaf.bvy; }
        return best;
    }
    split_at(x0, y0, l) = 1;
    return js;
}

// ---------------- 構文 (符号器/復号器共通) ----------------
void BlockCoder::code_leaf(SymIO& io, Models& md, Leaf& lf, int x0, int y0, int l) {
    const int s = 1 << l, n = s * s;
    const uint32_t L = static_cast<uint32_t>(l), pc = plane_ ? 1u : 0u;
    if (tools_.ibc) lf.pt = io.bit(md.ibc, 0, L, pc, lf.pt);
    if (lf.pt == 1) {
        lf.bvx = bv_px_ + io.sint(md.ibc, 1, L, lf.bvx - bv_px_);
        lf.bvy = bv_py_ + io.sint(md.ibc, 2, L, lf.bvy - bv_py_);
        if (!ibc_valid(x0 + lf.bvx, y0 + lf.bvy, s)) throw std::runtime_error("corrupt stream: block vector");
        bv_px_ = lf.bvx; bv_py_ = lf.bvy;
    } else {
        int is_cfl = 0;
        if (tools_.cfl) is_cfl = io.bit(md.cfl, 0, L, 0, lf.mode == kModeCfl);
        if (is_cfl) {
            lf.mode = kModeCfl;
            lf.alpha = io.sint(md.cfl, 1, L, lf.alpha);
            if (lf.alpha < -16 || lf.alpha > 16) throw std::runtime_error("corrupt stream: cfl alpha");
        } else {
            int m0, m1;
            mpm(x0, y0, m0, m1);
            const int is_mpm = io.bit(md.mode, 0, L, pc, lf.mode == m0 || lf.mode == m1);
            if (is_mpm) {
                lf.mode = io.bit(md.mode, 1, L, pc, lf.mode == m1) ? m1 : m0;
            } else if (tools_.all_angular) {
                lf.mode = static_cast<int>(io.uint(md.mode, 2, pc, static_cast<uint32_t>(lf.mode)));
            } else {
                static constexpr int kSet[4] = {kModeDC, kModePlanar, kModeHor, kModeVer};
                int idx = 0;
                for (int i = 0; i < 4; ++i) if (kSet[i] == lf.mode) idx = i;
                idx = io.bit(md.mode, 3, 0, pc, idx >> 1) * 2;
                idx += io.bit(md.mode, 4, 0, pc, io.w ? (lf.mode == kSet[idx + 1]) : 0);
                lf.mode = kSet[idx];
            }
            if (lf.mode < 0 || lf.mode >= kNumIntra) throw std::runtime_error("corrupt stream: intra mode");
        }
    }
    std::vector<int32_t> pred(n);
    predict(lf, x0, y0, l, pred.data());
    if (io.w) quantize(lf, x0, y0, l, pred.data());  // 再構成ずれを防ぐため現在の再構成から再量子化

    const int cbf = io.bit(md.cbf, 0, L, pc * 64 + static_cast<uint32_t>(lf.pt ? 63 : lf.mode), lf.last >= 0);
    if (!cbf) {
        lf.last = -1;
        lf.q.assign(lf.qmode == 0 ? n : 0, 0);
        lf.tns_on = false;
        lf.nf = 0;
    } else {
        if (tools_.e8) lf.qmode = io.bit(md.e8, 0, L, pc, lf.qmode);
        else lf.qmode = 0;
        if (tools_.tns && s >= 8) {
            lf.tns_on = io.bit(md.tns, 0, L, pc, lf.tns_on);
            if (lf.tns_on) {
                lf.tns.qbits = 4;
                lf.tns.order = static_cast<int>(io.uint(md.tns, 1, 0, static_cast<uint32_t>(lf.tns.order - 1))) + 1;
                if (lf.tns.order > 8) throw std::runtime_error("corrupt stream: tns order");
                lf.tns.parcor_q.resize(lf.tns.order);
                for (int i = 0; i < lf.tns.order; ++i) {
                    lf.tns.parcor_q[i] = io.sint(md.tns, 2 + i, 0, lf.tns.parcor_q[i]);
                    if (std::abs(lf.tns.parcor_q[i]) > 7) throw std::runtime_error("corrupt stream: tns parcor");
                }
            }
        } else {
            lf.tns_on = false;
        }
        const auto& scan = diag_scan(l);
        if (lf.qmode == 0) {
            lf.last = static_cast<int>(io.uint(md.last, L, pc, static_cast<uint32_t>(lf.last)));
            if (lf.last >= n) throw std::runtime_error("corrupt stream: last");
            if (!io.w) lf.q.assign(n, 0);
            CMModel& cm = plane_ ? md.coef_c : md.coef_y;
            for (int i = 0; i <= lf.last; ++i) {
                const int pos = scan[i];
                const int u = pos & (s - 1), v = pos >> l;
                const int32_t n1 = i > 0 ? std::abs(lf.q[scan[i - 1]]) : 0;
                const int32_t n2 = i > 1 ? std::abs(lf.q[scan[i - 2]]) : 0;
                const uint32_t a = static_cast<uint32_t>(std::min(n1 + n2, 15)) | (static_cast<uint32_t>(std::min(u + v, 15)) << 4);
                const uint32_t b = L * 4 + (i == lf.last ? 1u : 0u) + (i == 0 ? 2u : 0u) + (lf.tns_on ? 64u : 0u);
                if (i == lf.last) {
                    const int32_t cv = lf.q[pos];
                    const uint32_t mag = io.uint(cm, a, b, io.w ? static_cast<uint32_t>(std::abs(cv)) - 1u : 0u) + 1u;
                    const int neg = io.bit(cm, 1, a, b, cv < 0);
                    lf.q[pos] = neg ? -static_cast<int32_t>(mag) : static_cast<int32_t>(mag);
                } else {
                    lf.q[pos] = io.sint(cm, a, b, lf.q[pos]);
                }
            }
            if (tools_.nf) {
                if (lf.last + 1 < n) lf.nf = static_cast<int>(io.uint(md.nf, L, pc, static_cast<uint32_t>(lf.nf)));
                else lf.nf = 0;
                if (lf.nf > 3) throw std::runtime_error("corrupt stream: nf");
            }
        } else {
            lf.last = static_cast<int>(io.uint(md.last, L, pc + 2, static_cast<uint32_t>(lf.last)));
            if ((lf.last + 1) * 8 > std::max(n, 8)) throw std::runtime_error("corrupt stream: e8 chunks");
            if (!io.w) lf.q.assign(static_cast<size_t>(lf.last + 1) * 8, 0);
            int32_t prev_l1 = 0;
            for (int ch = 0; ch <= lf.last; ++ch) {
                double y[8];
                for (int k = 0; k < 8; ++k) y[k] = lf.q[ch * 8 + k] * 0.5;
                LatticeIndex idx;
                if (io.w) idx = lattice_to_index(Lattice::E8, y, 8);
                else idx.z.assign(8, 0);
                const uint32_t ca = static_cast<uint32_t>(std::min(prev_l1, 15));
                const uint32_t cb = static_cast<uint32_t>(std::min(ch, 15)) + 16 * pc;
                idx.coset = io.bit(md.e8, 1, ca, cb, idx.coset);
                int32_t l1 = 0;
                for (int k = 0; k < 8; ++k) {
                    idx.z[k] = io.sint(md.e8, ca | (static_cast<uint32_t>(k) << 4), cb, idx.z[k]);
                    l1 += std::abs(idx.z[k]);
                }
                prev_l1 = l1;
                if (!io.w) {
                    index_to_lattice(Lattice::E8, idx, 8, y);
                    for (int k = 0; k < 8; ++k) lf.q[ch * 8 + k] = static_cast<int32_t>(std::lround(2.0 * y[k]));
                }
            }
        }
    }
    reconstruct(lf, x0, y0, l, pred.data());
    set_modes4(x0, y0, s, lf.pt ? kModePlanar : lf.mode);
}

void BlockCoder::code_node(SymIO& io, Models& md, int x0, int y0, int l) {
    int split = 0;
    if (l > min_log2_) split = io.bit(md.split, 0, static_cast<uint32_t>(l), static_cast<uint32_t>(plane_), io.w ? split_at(x0, y0, l) : 0);
    if (split) {
        const int h = 1 << (l - 1);
        code_node(io, md, x0, y0, l - 1);
        code_node(io, md, x0 + h, y0, l - 1);
        code_node(io, md, x0, y0 + h, l - 1);
        code_node(io, md, x0 + h, y0 + h, l - 1);
        return;
    }
    Leaf lf;
    if (io.w) lf = leaf_at(x0, y0, l);
    code_leaf(io, md, lf, x0, y0, l);
}

void BlockCoder::code_ctu(SymIO& io, Models& md, int cx, int cy, int ctu) {
    cx_ = cx; cy_ = cy; ctu_ = ctu;
    const int bs = 1 << max_log2_;
    for (int by = cy; by < cy + ctu; by += bs)
        for (int bx = cx; bx < cx + ctu; bx += bs) {
            if (io.w) {
                // RD 探索は再構成・MPM・BV 予測子を変更するので、符号化前に状態を戻す
                std::vector<int32_t> rb;
                save(bx, by, bs, rb);
                const std::vector<int8_t> m4 = modes4_;
                const int px = bv_px_, py = bv_py_;
                rd_node(bx, by, max_log2_);
                restore(bx, by, bs, rb);
                modes4_ = m4;
                bv_px_ = px; bv_py_ = py;
            }
            code_node(io, md, bx, by, max_log2_);
        }
}

}  // namespace fvc
