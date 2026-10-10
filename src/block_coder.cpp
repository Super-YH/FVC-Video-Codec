#include "block_coder.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "dictionary.hpp"
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

// 4x4 アダマール変換の絶対値和 (SATD)
int64_t satd4(const int32_t* r, int s) {
    int64_t total = 0;
    for (int by = 0; by < s; by += 4)
        for (int bx = 0; bx < s; bx += 4) {
            int32_t m[16], t[16];
            for (int i = 0; i < 4; ++i) {
                const int32_t* p = r + (by + i) * s + bx;
                const int32_t a = p[0] + p[3], b = p[1] + p[2], c = p[1] - p[2], d = p[0] - p[3];
                m[i * 4 + 0] = a + b; m[i * 4 + 1] = d + c; m[i * 4 + 2] = a - b; m[i * 4 + 3] = d - c;
            }
            for (int j = 0; j < 4; ++j) {
                const int32_t a = m[j] + m[12 + j], b = m[4 + j] + m[8 + j], c = m[4 + j] - m[8 + j], d = m[j] - m[12 + j];
                t[j] = a + b; t[4 + j] = d + c; t[8 + j] = a - b; t[12 + j] = d - c;
            }
            for (int k = 0; k < 16; ++k) total += std::abs(t[k]);
        }
    return total / 2;
}

// 係数の文脈: 走査順で既に符号化済みの低域側 2D テンプレート
//   (u-1,v) (u,v-1) (u-1,v-1) (u-2,v) (u,v-2) の min(|q|,3) の和と非ゼロ数、および周波数位置 u+v
uint32_t coef_ctx(const int32_t* q, int u, int v, int l) {
    const int s = 1 << l;
    int sum = 0, cnt = 0;
    auto add = [&](int x, int y) {
        if (x < 0 || y < 0) return;
        const int32_t a = std::abs(q[(y << l) + x]);
        sum += std::min<int32_t>(a, 3);
        cnt += a != 0;
    };
    add(u - 1, v); add(u, v - 1); add(u - 1, v - 1); add(u - 2, v); add(u, v - 2);
    (void)s;
    return static_cast<uint32_t>(std::min(sum, 15)) | (static_cast<uint32_t>(std::min(cnt, 5)) << 4) |
           (static_cast<uint32_t>(std::min(u + v, 15)) << 7);
}

namespace {
// HEVC 互換の角度表 (モード 2..34)
constexpr int kAngle[35] = {0, 0, 32, 26, 21, 17, 13, 9, 5, 2, 0, -2, -5, -9, -13, -17, -21, -26,
                            -32, -26, -21, -17, -13, -9, -5, -2, 0, 2, 5, 9, 13, 17, 21, 26, 32};
int inv_angle(int a) { return (256 * 32 + (a < 0 ? -a : a) / 2) / a; }  // a<0 のみ使用 (負値)

int ilog2(int s) { int l = 0; while ((1 << l) < s) ++l; return l; }
}  // namespace

BlockCoder::BlockCoder(Plane* rec, const Plane* org, const Plane* luma, int plane, int32_t lo, int32_t hi, double step,
                       double lambda, int min_log2, int max_log2, const Tools& tools, const Search& search,
                       const InterCtx* inter, Dictionary* dict, int tx0, int ty0, int tw, int th, int32_t leaf_base)
    : rec_(rec), org_(org), luma_(luma), plane_(plane), lo_(lo), hi_(hi), mid_((lo + hi + 1) / 2), step_(step),
      lambda_(lambda), min_log2_(min_log2), max_log2_(max_log2), tools_(tools), search_(search), dict_(dict) {
    if (!luma_) tools_.cfl = false;
    if (inter) inter_ = *inter;
    tx0_ = tx0; ty0_ = ty0;
    tx1_ = tw > 0 ? tx0 + tw : rec_->w;
    ty1_ = th > 0 ? ty0 + th : rec_->h;
    leaf_counter_ = leaf_base;
    if (!dict_ || plane_ != 0) tools_.dict = false;
    modes4_.assign(static_cast<size_t>(rec_->w / 4) * (rec_->h / 4), static_cast<int8_t>(kModePlanar));
    leaf4_.assign(modes4_.size(), -1);
    flags4_.assign(modes4_.size(), 0);
    split_map_.resize(max_log2_ + 1);
    leaf_map_.resize(max_log2_ + 1);
    if (org_)
        for (int l = min_log2_; l <= max_log2_; ++l) {
            split_map_[l].assign(static_cast<size_t>((tx1_ - tx0_) >> l) * ((ty1_ - ty0_) >> l), 0);
            leaf_map_[l].resize(static_cast<size_t>((tx1_ - tx0_) >> l) * ((ty1_ - ty0_) >> l));
        }
}

// ---------------- 予測 ----------------
void BlockCoder::mpm(int x0, int y0, int& m0, int& m1) const {
    const int gw = rec_->w / 4;
    const int left = x0 > tx0_ ? modes4_[(y0 / 4) * gw + (x0 / 4 - 1)] : kModePlanar;
    const int above = y0 > ty0_ ? modes4_[(y0 / 4 - 1) * gw + (x0 / 4)] : kModePlanar;
    m0 = left;
    m1 = above != left ? above : (left != kModePlanar ? kModePlanar : kModeDC);
}

void BlockCoder::mpm6(int x0, int y0, int* out) const {
    int m0, m1;
    mpm(x0, y0, m0, m1);
    const int gw = rec_->w / 4;
    const int left = x0 > tx0_ ? modes4_[(y0 / 4) * gw + (x0 / 4 - 1)] : kModePlanar;
    const int above = y0 > ty0_ ? modes4_[(y0 / 4 - 1) * gw + (x0 / 4)] : kModePlanar;
    int n = 0;
    auto push = [&](int m) {
        if (n >= 6 || m < 0 || m >= kNumIntra || m == kModeCfl) return;
        for (int i = 0; i < n; ++i) if (out[i] == m) return;
        out[n++] = m;
    };
    push(left); push(above); push(kModePlanar); push(kModeDC);
    for (int m : {left, above})
        if (m >= 2 && m <= 34) { push(m == 2 ? 34 : m - 1); push(m == 34 ? 2 : m + 1); }
    for (int m : {kModeVer, kModeHor, 2, 18, kModeSmooth, kModePaeth}) push(m);
    (void)m0; (void)m1;
}

void BlockCoder::set_modes4(int x0, int y0, int s, int mode) {
    const int gw = rec_->w / 4;
    const int m = (mode == kModeCfl || mode < 0) ? kModePlanar : mode;
    for (int y = y0 / 4; y < (y0 + s) / 4; ++y)
        for (int x = x0 / 4; x < (x0 + s) / 4; ++x) modes4_[y * gw + x] = static_cast<int8_t>(m);
}

// (px,py) の 4x4 単位がブロック (x0,y0) より前に再構成済みか (タイル内の CTU ラスタ順 + CTU 内 Z 順)。
// RD の状態に依存しない幾何学的規則なので符号器/復号器で一致する。
bool BlockCoder::coded_before(int px, int py, int x0, int y0) const {
    if (px < tx0_ || py < ty0_ || px >= tx1_ || py >= ty1_) return false;
    const int cx = px / ctu_ * ctu_, cy = py / ctu_ * ctu_;
    if (cy != cy_) return cy < cy_;
    if (cx != cx_) return cx < cx_;
    auto morton = [](int x, int y) {
        int m = 0;
        for (int b = 0; b < 8; ++b) m |= (((x >> b) & 1) << (2 * b)) | (((y >> b) & 1) << (2 * b + 1));
        return m;
    };
    return morton((px - cx) >> 2, (py - cy) >> 2) < morton((x0 - cx_) >> 2, (y0 - cy_) >> 2);
}

void BlockCoder::intra_angular(int x0, int y0, int s, int mode, int32_t* pred) const {
    const Plane& r = *rec_;
    const bool ht = y0 > ty0_, hl = x0 > tx0_;
    // 参照: corner, top[0..2s), left[0..2s)。右上/左下は復号済みなら実画素、なければ端値で延長
    std::vector<int32_t> top(2 * s), left(2 * s);
    for (int i = 0; i < s; ++i) {
        top[i] = ht ? r.at(x0 + i, y0 - 1) : (hl ? r.at(x0 - 1, y0) : mid_);
        left[i] = hl ? r.at(x0 - 1, y0 + i) : (ht ? r.at(x0, y0 - 1) : mid_);
    }
    for (int i = s; i < 2 * s; ++i) {
        top[i] = (ht && coded_before(x0 + i, y0 - 1, x0, y0)) ? r.at(x0 + i, y0 - 1) : top[i - 1];
        left[i] = (hl && coded_before(x0 - 1, y0 + i, x0, y0)) ? r.at(x0 - 1, y0 + i) : left[i - 1];
    }
    int32_t corner = (ht && hl) ? r.at(x0 - 1, y0 - 1) : (ht ? top[0] : left[0]);
    if (mode == kModePaeth) {
        for (int y = 0; y < s; ++y)
            for (int x = 0; x < s; ++x) {
                const int32_t base = top[x] + left[y] - corner;
                const int32_t pl = std::abs(base - left[y]), pt = std::abs(base - top[x]), pc = std::abs(base - corner);
                pred[y * s + x] = (pl <= pt && pl <= pc) ? left[y] : (pt <= pc ? top[x] : corner);
            }
        return;
    }
    if (mode == kModeSmooth) {
        // 2 次の重み w(i) = 256 (1 - i/s)^2 で上/左と右上/左下を補間
        const int64_t ss = static_cast<int64_t>(s) * s;
        auto w = [&](int i) { return static_cast<int64_t>((256 * (s - i) * (s - i) + ss / 2) / ss); };
        const int32_t tr = top[s - 1], bl = left[s - 1];
        for (int y = 0; y < s; ++y)
            for (int x = 0; x < s; ++x) {
                const int64_t wy = w(y), wx = w(x);
                const int64_t v = wy * top[x] + (256 - wy) * bl + wx * left[y] + (256 - wx) * tr;
                pred[y * s + x] = static_cast<int32_t>((v + 256) >> 9);
            }
        return;
    }
    // 参照画素の平滑化 [1,2,1] (HEVC 規則: 8x8 以上、Planar と H/V から離れた角度)
    if (s >= 8 && mode != kModeDC) {
        const int thr = s == 8 ? 7 : s == 16 ? 1 : 0;
        const int dist = mode == kModePlanar ? 99 : std::min(std::abs(mode - kModeHor), std::abs(mode - kModeVer));
        if (dist > thr) {
            std::vector<int32_t> t2 = top, l2 = left;
            const int32_t c2 = (left[0] + 2 * corner + top[0] + 2) >> 2;
            t2[0] = (corner + 2 * top[0] + top[1] + 2) >> 2;
            l2[0] = (corner + 2 * left[0] + left[1] + 2) >> 2;
            for (int i = 1; i < 2 * s - 1; ++i) {
                t2[i] = (top[i - 1] + 2 * top[i] + top[i + 1] + 2) >> 2;
                l2[i] = (left[i - 1] + 2 * left[i] + left[i + 1] + 2) >> 2;
            }
            top.swap(t2); left.swap(l2); corner = c2;
        }
    }
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

// OBMC (§7.3): 上・左の (符号化済み) インター近傍の動きで予測した値を、境界から離れるほど重みが 0 になる
// 余弦窓で重ね合わせる。動きの異なるブロック境界の段差 (ブロック状の継ぎ目) を予測段階で消す。
void BlockCoder::obmc(const MotionInfo& cur, int x0, int y0, int s, int32_t* pred) const {
    if (!inter_.mf || s < 8) return;
    const int ov = std::min(s / 2, 16);
    int w[16];
    for (int i = 0; i < ov; ++i) w[i] = static_cast<int>(std::lround(16.0 * (1.0 + std::cos(3.14159265358979 * (i + 0.5) / ov))));
    std::vector<int32_t> nb(static_cast<size_t>(16) * 16);
    const int W4 = inter_.mf->w4 * 4, H4 = inter_.mf->h4 * 4;
    for (int side = 0; side < 2; ++side) {  // 0: 上, 1: 左
        if (side == 0 ? (y0 <= ty0_ || y0 - 1 >= H4) : (x0 <= tx0_ || x0 - 1 >= W4)) continue;
        for (int t = 0; t < s; t += 4) {
            const int nx = side == 0 ? x0 + t : x0 - 1, ny = side == 0 ? y0 - 1 : y0 + t;
            if (nx >= W4 || ny >= H4) continue;
            const MotionInfo& m = inter_.mf->at(nx, ny);
            if (m.dir == 0 || m == cur) continue;
            const int bw = side == 0 ? 4 : ov, bh = side == 0 ? ov : 4;
            const int bx = side == 0 ? x0 + t : x0, by = side == 0 ? y0 : y0 + t;
            inter_predict(m, inter_.l0, inter_.l1, bx, by, bw, bh, 0, lo_, hi_, nb.data());
            for (int y = 0; y < bh; ++y)
                for (int x = 0; x < bw; ++x) {
                    const int wi = w[side == 0 ? y : x];
                    int32_t& p = pred[(by - y0 + y) * s + (bx - x0 + x)];
                    p = (p * (64 - wi) + nb[y * bw + x] * wi + 32) >> 6;
                }
        }
    }
}

// FIR 動き基底 (§7.3): P' = P + (ψ0 ∂x P + ψ1 ∂y P + ψ2 ∂xx P + ψ3 ∂yy P) / 16
//  ∂x, ∂y は局所変位 (動き成分摘出)、∂xx, ∂yy は異方性のぼけ/鮮鋭化。差分はブロック内で端を複製
namespace {
void fir_basis(const int32_t* p, int s, int k, std::vector<double>& out) {
    out.resize(static_cast<size_t>(s) * s);
    auto at = [&](int x, int y) { return static_cast<double>(p[std::clamp(y, 0, s - 1) * s + std::clamp(x, 0, s - 1)]); };
    for (int y = 0; y < s; ++y)
        for (int x = 0; x < s; ++x) {
            double v = 0;
            switch (k) {
            case 0: v = (at(x + 1, y) - at(x - 1, y)) * 0.5; break;
            case 1: v = (at(x, y + 1) - at(x, y - 1)) * 0.5; break;
            case 2: v = at(x + 1, y) - 2 * at(x, y) + at(x - 1, y); break;
            default: v = at(x, y + 1) - 2 * at(x, y) + at(x, y - 1); break;
            }
            out[y * s + x] = v;
        }
}
}  // namespace

void BlockCoder::fir_apply(const int* psi, int s, int32_t* pred) const {
    if (!psi[0] && !psi[1] && !psi[2] && !psi[3]) return;
    std::vector<double> acc(static_cast<size_t>(s) * s, 0.0), b;
    for (int k = 0; k < 4; ++k) {
        if (!psi[k]) continue;
        fir_basis(pred, s, k, b);
        for (size_t i = 0; i < acc.size(); ++i) acc[i] += psi[k] * b[i];
    }
    for (size_t i = 0; i < acc.size(); ++i)
        pred[i] = std::clamp(pred[i] + static_cast<int32_t>(std::lround(acc[i] / 16.0)), lo_, hi_);
}

bool BlockCoder::fir_fit(const Leaf& lf, int x0, int y0, int s, int* psi) const {
    const int n = s * s;
    std::vector<int32_t> p(n);
    inter_predict(lf.mi, inter_.l0, inter_.l1, x0, y0, s, s, 0, lo_, hi_, p.data());
    std::vector<double> B[4];
    for (int k = 0; k < 4; ++k) fir_basis(p.data(), s, k, B[k]);
    double M[4][5] = {};
    for (int i = 0; i < n; ++i) {
        const double r = org_->at(x0 + i % s, y0 + i / s) - p[i];
        for (int a = 0; a < 4; ++a) {
            for (int c = 0; c < 4; ++c) M[a][c] += B[a][i] * B[c][i];
            M[a][4] += B[a][i] * r;
        }
    }
    for (int a = 0; a < 4; ++a) M[a][a] += 1e-3 * n;
    for (int c = 0; c < 4; ++c) {
        int pv = c;
        for (int r = c + 1; r < 4; ++r) if (std::abs(M[r][c]) > std::abs(M[pv][c])) pv = r;
        if (std::abs(M[pv][c]) < 1e-9) return false;
        for (int k = 0; k < 5; ++k) std::swap(M[c][k], M[pv][k]);
        for (int r = 0; r < 4; ++r) {
            if (r == c) continue;
            const double f = M[r][c] / M[c][c];
            for (int k = c; k < 5; ++k) M[r][k] -= f * M[c][k];
        }
    }
    bool any = false;
    for (int k = 0; k < 4; ++k) {
        psi[k] = std::clamp(static_cast<int>(std::lround(16.0 * M[k][4] / M[k][k])), -16, 16);
        any |= psi[k] != 0;
    }
    return any;
}

void BlockCoder::predict(const Leaf& lf, int x0, int y0, int l, int32_t* pred) const {
    const int s = 1 << l;
    if (lf.pt == 2) {
        if (is_luma()) {
            if (lf.part == 0) {
                inter_predict(lf.mi, inter_.l0, inter_.l1, x0, y0, s, s, 0, lo_, hi_, pred);
                fir_apply(lf.fir, s, pred);
                obmc(lf.mi, x0, y0, s, pred);
            } else {
                // 長方形予測分割 (1: 上下 2NxN, 2: 左右 Nx2N)。変換は正方形のまま
                const int pw = lf.part == 2 ? s / 2 : s, ph = lf.part == 1 ? s / 2 : s;
                std::vector<int32_t> sub(static_cast<size_t>(pw) * ph);
                for (int k = 0; k < 2; ++k) {
                    const int ox = lf.part == 2 ? k * pw : 0, oy = lf.part == 1 ? k * ph : 0;
                    inter_predict(k ? lf.mi2 : lf.mi, inter_.l0, inter_.l1, x0 + ox, y0 + oy, pw, ph, 0, lo_, hi_, sub.data());
                    for (int y = 0; y < ph; ++y)
                        for (int x = 0; x < pw; ++x) pred[(oy + y) * s + ox + x] = sub[y * pw + x];
                }
            }
        } else {
            // 色差: 輝度の動きベクトル場から 4x4 単位で導出 (動き情報は送らない)
            const int cs = inter_.chroma_shift, b = 4;
            auto mi_at = [&](int bx, int by) {
                const int lx = std::min((x0 + bx) << cs, inter_.mf->w4 * 4 - 1);
                const int ly = std::min((y0 + by) << cs, inter_.mf->h4 * 4 - 1);
                MotionInfo mi = inter_.mf->at(lx, ly);
                if (mi.dir == 0) {
                    // 輝度がイントラの位置: 近傍 (左・上・右・下、最大 16 輝度画素) のインター動きを借りる。
                    // 動きベクトル 0 で代用すると動物体の色が前位置に残る (色の残像・褪色) ため。
                    const int W4 = inter_.mf->w4 * 4, H4 = inter_.mf->h4 * 4;
                    for (int d = 4; d <= 16 && mi.dir == 0; d += 4)
                        for (const auto& o : {std::pair<int, int>{-d, 0}, {0, -d}, {d, 0}, {0, d}}) {
                            const int nx = lx + o.first, ny = ly + o.second;
                            if (nx < 0 || ny < 0 || nx >= W4 || ny >= H4) continue;
                            const MotionInfo& m2 = inter_.mf->at(nx, ny);
                            if (m2.dir != 0) { mi = m2; break; }
                        }
                    if (mi.dir == 0) { mi = MotionInfo{}; mi.dir = 1; }
                }
                return mi;
            };
            // 高速経路: ブロック内の動きが一様なら 1 回で補償
            const MotionInfo m0 = mi_at(0, 0);
            bool uniform = true;
            for (int by = 0; by < s && uniform; by += b)
                for (int bx = 0; bx < s && uniform; bx += b) uniform = mi_at(bx, by) == m0;
            if (uniform) {
                inter_predict(m0, inter_.l0, inter_.l1, x0, y0, s, s, cs, lo_, hi_, pred);
                return;
            }
            std::vector<int32_t> sub(b * b);
            for (int by = 0; by < s; by += b)
                for (int bx = 0; bx < s; bx += b) {
                    const int lx = std::min((x0 + bx) << cs, inter_.mf->w4 * 4 - 1);
                    const int ly = std::min((y0 + by) << cs, inter_.mf->h4 * 4 - 1);
                    MotionInfo mi = inter_.mf->at(lx, ly);
                    if (mi.dir == 0) { mi = MotionInfo{}; mi.dir = 1; }
                    inter_predict(mi, inter_.l0, inter_.l1, x0 + bx, y0 + by, b, b, cs, lo_, hi_, sub.data());
                    for (int y = 0; y < b; ++y)
                        for (int x = 0; x < b; ++x) pred[(by + y) * s + bx + x] = sub[y * b + x];
                }
        }
    } else if (lf.pt == 4) {
        for (int y = 0; y < s; ++y)
            for (int x = 0; x < s; ++x) {
                const int64_t v = static_cast<int64_t>(lf.xgain) * (lf.xsrc ? tband_ : xband_)->at(x0 + x, y0 + y);
                pred[y * s + x] = std::clamp(static_cast<int32_t>(v >= 0 ? (v + 2) / 4 : -((-v + 2) / 4)), lo_, hi_);
            }
    } else if (lf.pt == 3) {
        intra_angular(x0, y0, s, kModeDC, pred);
        const std::vector<int32_t>& d = dict_->data(dict_->ids_for(s)[lf.dict_idx]);
        for (int i = 0; i < s * s; ++i)
            pred[i] = std::clamp(pred[i] + static_cast<int32_t>((static_cast<int64_t>(lf.dict_gain) * d[i] + 8) >> 4), lo_, hi_);
    } else if (lf.pt == 1) {
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
    if (rx < tx0_ || ry < ty0_ || rx + s > tx1_ || ry + s > ty1_) return false;
    if (ry + s <= cy_) return true;                                     // 上の CTU 行 (タイル内)
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


// ---------------- インター (§7.3) ----------------
// 時間方向候補 (§7.3): 同位置ピクチャの動きを POC 距離で伸縮
bool BlockCoder::temporal_cand(int x, int y, MotionInfo& out) const {
    if (!tools_.tmvp || !inter_.col) return false;
    const MotionField& c = *inter_.col;
    const int cx = std::min(x, c.w4 * 4 - 1), cy = std::min(y, c.h4 * 4 - 1);
    const MotionInfo& m = c.at(cx, cy);
    if (m.dir == 0) return false;
    const int l = (m.dir & 1) ? 0 : 1;
    const int td = inter_.col_poc - inter_.col_ref_poc[l][m.ref[l]];
    if (td == 0) return false;
    auto scale = [&](int v, int tb) {
        const int64_t num = static_cast<int64_t>(v) * tb;
        const int64_t r = num >= 0 ? (num + std::abs(td) / 2) / td : -((-num + std::abs(td) / 2) / td);
        return static_cast<int16_t>(std::clamp<int64_t>(r, -16000, 16000));
    };
    out = MotionInfo{};
    const int tb0 = inter_.cur_poc - inter_.ref_poc[0][0];
    out.mvx[0] = scale(m.mvx[l], tb0);
    out.mvy[0] = scale(m.mvy[l], tb0);
    out.dir = 1;
    if (inter_.bframe && inter_.nref[1] > 0) {
        const int tb1 = inter_.cur_poc - inter_.ref_poc[1][0];
        out.mvx[1] = scale(m.mvx[l], tb1);
        out.mvy[1] = scale(m.mvy[l], tb1);
        out.dir = 3;
    }
    return true;
}

int BlockCoder::merge_list(int x0, int y0, int w, int h, MotionInfo* out) const {
    int n = 0;
    auto push = [&](const MotionInfo& m) {
        if (m.dir == 0 || n >= kMaxMerge) return;
        if (((m.dir >> 1) & 1) && inter_.nref[1] == 0) return;
        for (int i = 0; i < n; ++i) if (out[i] == m) return;
        out[n++] = m;
    };
    if (x0 > tx0_) push(inter_.mf->at(x0 - 1, y0 + h - 1));
    if (y0 > ty0_) push(inter_.mf->at(x0 + w - 1, y0 - 1));
    MotionInfo t;
    if (temporal_cand(x0 + w / 2, y0 + h / 2, t)) push(t);
    if (x0 > tx0_ && y0 > ty0_) push(inter_.mf->at(x0 - 1, y0 - 1));
    MotionInfo g;
    g.dir = 1;
    {
        int gx, gy;
        inter_.gmv_at(x0 + w / 2, y0 + h / 2, gx, gy);
        g.mvx[0] = static_cast<int16_t>(gx);
        g.mvy[0] = static_cast<int16_t>(gy);
    }
    push(g);
    MotionInfo z;
    z.dir = 1;
    push(z);
    return n;
}

void BlockCoder::mv_pred(int x0, int y0, int list, int ref, int& px, int& py) const {
    auto try_mi = [&](const MotionInfo& m) {
        if (((m.dir >> list) & 1) && m.ref[list] == ref) { px = m.mvx[list]; py = m.mvy[list]; return true; }
        return false;
    };
    if (x0 > tx0_ && try_mi(inter_.mf->at(x0 - 1, y0))) return;
    if (y0 > ty0_ && try_mi(inter_.mf->at(x0, y0 - 1))) return;
    MotionInfo t;
    if (ref == 0 && temporal_cand(x0, y0, t) && try_mi(t)) return;
    if (list == 0) inter_.gmv_at(x0, y0, px, py);
    else px = py = 0;
}

int64_t BlockCoder::me_cost(int x0, int y0, int w, int h, const MotionInfo& mi) const {
    if (search_.approx_subpel && mi.psi == 0 && (mi.dir == 1 || mi.dir == 2)) {
        // 探索用近似: 1/4 画素位置を双線形補間で評価 (8 タップより大幅に軽い)
        const int l = mi.dir == 2 ? 1 : 0;
        const RefPlane& rp = (l ? inter_.l1 : inter_.l0)[mi.ref[l]];
        const Plane& p = *rp.p;
        const int ix = mi.mvx[l] >> 2, iy = mi.mvy[l] >> 2, fx = mi.mvx[l] & 3, fy = mi.mvy[l] & 3;
        const int w00 = (4 - fx) * (4 - fy), w10 = fx * (4 - fy), w01 = (4 - fx) * fy, w11 = fx * fy;
        int64_t sad = 0;
        for (int y = 0; y < h; ++y) {
            const int ry0 = std::clamp(y0 + y + iy, 0, p.h - 1), ry1 = std::clamp(y0 + y + iy + 1, 0, p.h - 1);
            for (int x = 0; x < w; ++x) {
                const int rx0 = std::clamp(x0 + x + ix, 0, p.w - 1), rx1 = std::clamp(x0 + x + ix + 1, 0, p.w - 1);
                int32_t v = (w00 * p.at(rx0, ry0) + w10 * p.at(rx1, ry0) + w01 * p.at(rx0, ry1) + w11 * p.at(rx1, ry1) + 8) >> 4;
                v = ((v * rp.gain_q + 32) >> 6) + rp.off;
                sad += std::abs(org_->at(x0 + x, y0 + y) - v);
            }
        }
        return sad;
    }
    std::vector<int32_t> pred(static_cast<size_t>(w) * h);
    inter_predict(mi, inter_.l0, inter_.l1, x0, y0, w, h, 0, lo_, hi_, pred.data());
    int64_t sad = 0;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) sad += std::abs(org_->at(x0 + x, y0 + y) - pred[y * w + x]);
    return sad;
}

void BlockCoder::motion_search(int x0, int y0, int w, int h, std::vector<Leaf>& cands) const {
    const double lsad = std::sqrt(lambda_);
    auto mvbits = [&](int dx, int dy) { return 2.0 + 2.0 * std::log2(1.0 + std::abs(dx)) + 2.0 * std::log2(1.0 + std::abs(dy)); };
    // マージ候補
    MotionInfo ml[kMaxMerge];
    const int nm = merge_list(x0, y0, w, h, ml);
    for (int i = 0; i < nm; ++i) { Leaf lf; lf.pt = 2; lf.mi = ml[i]; lf.merge = i; cands.push_back(lf); }
    MotionInfo best_uni[2];
    double best_uni_cost[2] = {1e300, 1e300};
    for (int list = 0; list < (inter_.bframe ? 2 : 1); ++list)
        for (int ref = 0; ref < inter_.nref[list]; ++ref) {
            const Plane& rp = *(list ? inter_.l1 : inter_.l0)[ref].p;
            int px, py;
            mv_pred(x0, y0, list, ref, px, py);
            // 整数探索 (SAD、参照はクランプ)
            auto isad = [&](int ix, int iy, int64_t bound) {
                int64_t sad = 0;
                for (int y = 0; y < h && sad < bound; ++y) {
                    const int ry = std::clamp(y0 + y + iy, 0, rp.h - 1);
                    for (int x = 0; x < w; ++x)
                        sad += std::abs(org_->at(x0 + x, y0 + y) - rp.at(std::clamp(x0 + x + ix, 0, rp.w - 1), ry));
                }
                return sad;
            };
            int bx = (px + 2) >> 2, by = (py + 2) >> 2;
            double bc = 1e300;
            auto test_int = [&](int ix, int iy) {
                const double c = static_cast<double>(isad(ix, iy, static_cast<int64_t>(bc) + 1)) + lsad * mvbits(ix * 4 - px, iy * 4 - py);
                if (c < bc) { bc = c; bx = ix; by = iy; }
            };
            test_int((px + 2) >> 2, (py + 2) >> 2);
            test_int(0, 0);
            {
                int gx, gy;
                inter_.gmv_at(x0 + w / 2, y0 + h / 2, gx, gy);
                test_int((gx + 2) >> 2, (gy + 2) >> 2);
            }
            for (int i = 0; i < nm; ++i)
                if ((ml[i].dir >> list) & 1) test_int((ml[i].mvx[list] + 2) >> 2, (ml[i].mvy[list] + 2) >> 2);
            const int R = search_.me_range;
            const int cx = bx, cy = by;
            if (R >= 8) {
                // 粗探索 (ステップ 2) → 中心近傍の全探索
                for (int dy = -R; dy <= R; dy += 2)
                    for (int dx = -R; dx <= R; dx += 2) test_int(cx + dx, cy + dy);
                const int fx = bx, fy = by;
                for (int dy = -2; dy <= 2; ++dy)
                    for (int dx = -2; dx <= 2; ++dx) test_int(fx + dx, fy + dy);
            } else {
                // 小ダイヤモンド反復
                for (int it = 0; it < R * 4; ++it) {
                    const int ox = bx, oy = by;
                    test_int(ox + 1, oy); test_int(ox - 1, oy); test_int(ox, oy + 1); test_int(ox, oy - 1);
                    if (ox == bx && oy == by) break;
                }
            }
            // 1/2, 1/4 画素精密化
            MotionInfo mi;
            mi.dir = static_cast<uint8_t>(1 << list);
            mi.ref[list] = static_cast<int8_t>(ref);
            int mx = bx * 4, my = by * 4;
            auto subcost = [&](int vx, int vy) {
                MotionInfo t = mi;
                t.mvx[list] = static_cast<int16_t>(vx);
                t.mvy[list] = static_cast<int16_t>(vy);
                return static_cast<double>(me_cost(x0, y0, w, h, t)) + lsad * (mvbits(vx - px, vy - py) + (inter_.nref[list] > 1 ? ref : 0));
            };
            double sc = subcost(mx, my);
            for (int step = 2; step >= (search_.qpel ? 1 : 2); step >>= 1) {
                const int ox = mx, oy = my;
                for (int dy = -step; dy <= step; dy += step)
                    for (int dx = -step; dx <= step; dx += step) {
                        if (!dx && !dy) continue;
                        const double c = subcost(ox + dx, oy + dy);
                        if (c < sc) { sc = c; mx = ox + dx; my = oy + dy; }
                    }
            }
            mi.mvx[list] = static_cast<int16_t>(mx);
            mi.mvy[list] = static_cast<int16_t>(my);
            if (sc < best_uni_cost[list]) { best_uni_cost[list] = sc; best_uni[list] = mi; }
        }
    for (int list = 0; list < 2; ++list) {
        if (best_uni_cost[list] >= 1e300) continue;
        Leaf lf; lf.pt = 2; lf.mi = best_uni[list];
        cands.push_back(lf);
        if (tools_.fir && search_.try_fir) {
            // FIR 平滑強度 (動きぼけ/フォーカス変化) の選択
            int64_t bsad = me_cost(x0, y0, w, h, lf.mi);
            int bpsi = 0;
            for (int psi = 1; psi <= 3; ++psi) {
                MotionInfo t = lf.mi;
                t.psi = static_cast<uint8_t>(psi);
                const int64_t c = me_cost(x0, y0, w, h, t);
                if (c < bsad) { bsad = c; bpsi = psi; }
            }
            if (bpsi) { Leaf l2 = lf; l2.mi.psi = static_cast<uint8_t>(bpsi); cands.push_back(l2); }
        }
    }
    if (inter_.bframe && search_.me_bi && best_uni_cost[0] < 1e300 && best_uni_cost[1] < 1e300) {
        Leaf lf; lf.pt = 2;
        lf.mi = best_uni[0];
        lf.mi.dir = 3;
        lf.mi.ref[1] = best_uni[1].ref[1];
        lf.mi.mvx[1] = best_uni[1].mvx[1];
        lf.mi.mvy[1] = best_uni[1].mvy[1];
        cands.push_back(lf);
    }
}

// 長方形予測分割の探索: 各区画で動き探索し、SAD+λ·ビット最小の動きを組み合わせる
void BlockCoder::rect_search(int x0, int y0, int s, std::vector<Leaf>& cands) {
    const double lsad = std::sqrt(lambda_);
    for (int part = 1; part <= 2; ++part) {
        const int pw = part == 2 ? s / 2 : s, ph = part == 1 ? s / 2 : s;
        Leaf lf;
        lf.pt = 2;
        lf.part = part;
        double total = 0;
        for (int k = 0; k < 2; ++k) {
            const int px = x0 + (part == 2 ? k * pw : 0), py = y0 + (part == 1 ? k * ph : 0);
            std::vector<Leaf> c;
            motion_search(px, py, pw, ph, c);
            double best = 1e300;
            MotionInfo bm;
            for (const Leaf& t : c) {
                const double j = static_cast<double>(me_cost(px, py, pw, ph, t.mi)) + lsad * (t.merge >= 0 ? 2.0 : 12.0);
                if (j < best) { best = j; bm = t.mi; }
            }
            if (best >= 1e300) return;
            total += best;
            (k ? lf.mi2 : lf.mi) = bm;
            if (k == 0) inter_.mf->fill(px, py, pw, ph, bm);  // 第 2 区画の候補用 (RD 後に上書きされる)
        }
        if (!(lf.mi == lf.mi2)) cands.push_back(lf);
        (void)total;
    }
}

// ---------------- 辞書予測 (§10.3) ----------------
void BlockCoder::dict_search(int x0, int y0, int s, std::vector<Leaf>& cands) const {
    const auto& ids = dict_->ids_for(s);
    std::vector<int32_t> dc(static_cast<size_t>(s) * s);
    intra_angular(x0, y0, s, kModeDC, dc.data());
    std::vector<double> r(static_cast<size_t>(s) * s);
    for (int y = 0; y < s; ++y)
        for (int x = 0; x < s; ++x) r[y * s + x] = org_->at(x0 + x, y0 + y) - dc[y * s + x];
    double best = 0;
    int bi = -1, bg = 0;
    for (size_t k = 0; k < ids.size(); ++k) {
        const auto& d = dict_->data(ids[k]);
        double num = 0, den = 0;
        for (size_t i = 0; i < d.size(); ++i) { num += r[i] * d[i]; den += static_cast<double>(d[i]) * d[i]; }
        if (den <= 0) continue;
        const double gain = num * num / den;  // SSE 減少量
        if (gain > best) {
            best = gain;
            bi = static_cast<int>(k);
            bg = std::clamp(static_cast<int>(std::lround(num / den * 16.0)), -64, 64);
        }
    }
    if (bi >= 0 && bg != 0) { Leaf lf; lf.pt = 3; lf.dict_idx = bi; lf.dict_gain = bg; cands.push_back(lf); }
}

// 帯域間文脈: ブロック位置の隣接帯域の平均振幅 (量子化ステップ単位) を 4 段階 (+1, 0 = 情報なし)
uint32_t BlockCoder::xb_ctx(int x0, int y0, int s) const {
    if (!xband_) return 0;
    double a = 0;
    for (int y = 0; y < s; ++y)
        for (int x = 0; x < s; ++x) a += std::abs(xband_->at(x0 + x, y0 + y));
    a /= static_cast<double>(s) * s * step_;
    return a < 0.25 ? 1u : a < 1.0 ? 2u : a < 3.0 ? 3u : 4u;
}

// 変化マスク文脈: LL 帯域の時間変化量 (ステップ単位) を 3 段階 (+1, 0 = 情報なし)
uint32_t BlockCoder::cb_ctx(int x0, int y0, int s) const {
    if (!cband_) return 0;
    double a = 0;
    for (int y = 0; y < s; ++y)
        for (int x = 0; x < s; ++x) a += std::abs(cband_->at(x0 + x, y0 + y));
    a /= static_cast<double>(s) * s * step_;
    return a < 0.3 ? 1u : a < 1.5 ? 2u : 3u;
}

// ---------------- RDOQ (§13.1) ----------------
// 走査順に、各係数のレベルを {round, round-1, 0} から D + λR 最小で貪欲に選ぶ。
// R は CM モデルの現在確率から (構文と同じ文脈で) 見積もる。最後に末尾の打ち切り位置を最適化。
void BlockCoder::rdoq(Leaf& lf, int l, const std::vector<double>& e, uint32_t xb, uint32_t cb) const {
    const int s = 1 << l, n = s * s;
    const auto& scan = diag_scan(l);
    CMModel& cm = plane_ ? md_->coef_c : md_->coef_y;
    const uint32_t L = static_cast<uint32_t>(l);
    double cost_acc = 0;
    SymIO io;
    io.cost = &cost_acc;
    io.enc = true;
    auto bits_of = [&](int32_t v, uint32_t a, uint32_t b) {
        cost_acc = 0;
        io.sint(cm, a, b, v);
        return cost_acc;
    };
    const int last0 = lf.last;
    std::vector<int32_t> lv(n, 0);
    std::vector<double> jcum(n + 1, 0.0);  // 位置 i までの (選択後) コスト累積
    double tail_d = 0;                     // 打ち切った場合の歪み (e^2) の累積
    std::vector<double> d0(n);
    for (int i = 0; i < n; ++i) d0[i] = e[i] * e[i];
    std::vector<int32_t> rq(n, 0);  // 決定済みレベル (ラスタ順、文脈用)
    for (int i = 0; i <= last0; ++i) {
        const int pos = scan[i];
        const int u = pos & (s - 1), v = pos >> l;
        const uint32_t a = coef_ctx(rq.data(), u, v, l);
        const uint32_t b = (L * 4 + (i == 0 ? 2u : 0u)) | (xb << 8) | (cb << 11);
        const double x = e[i] / step_;
        const int32_t r = static_cast<int32_t>(std::lround(std::abs(x)));
        double best = 1e300;
        int32_t bv = 0;
        for (int32_t m : {r, r - 1, 0}) {
            if (m < 0) continue;
            const int32_t vv = e[i] < 0 ? -m : m;
            const double d = (e[i] - vv * step_) * (e[i] - vv * step_);
            const double j = d + lambda_ * bits_of(vv, a, b);
            if (j < best) { best = j; bv = vv; }
            if (m == 0) break;
        }
        lv[i] = bv;
        rq[pos] = bv;
        jcum[i + 1] = jcum[i] + best;
    }
    (void)tail_d;
    // 末尾打ち切り: last = k を選ぶと k より後は 0 (歪み e^2)。last の符号化コストも加える
    std::vector<double> suffix(last0 + 2, 0.0);
    for (int i = last0; i >= 0; --i) suffix[i] = suffix[i + 1] + d0[i];
    double bestj = suffix[0] + lambda_ * 0.5;  // 全ゼロ (cbf=0)
    int bestk = -1;
    for (int k = 0; k <= last0; ++k) {
        if (lv[k] == 0) continue;
        cost_acc = 0;
        io.uintc(md_->last, L, plane_ ? 1u : 0u, static_cast<uint32_t>(k));
        const double j = jcum[k + 1] + suffix[k + 1] + lambda_ * cost_acc;
        if (j < bestj) { bestj = j; bestk = k; }
    }
    std::fill(lf.q.begin(), lf.q.end(), 0);
    for (int i = 0; i <= bestk; ++i) lf.q[scan[i]] = lv[i];
    lf.last = bestk;
}

// ---------------- 量子化 / 再構成 ----------------
void BlockCoder::quantize(Leaf& lf, int x0, int y0, int l, const int32_t* pred) const {
    const int s = 1 << l, n = s * s;
    if (tools_.pred_only) {
        lf.q.assign(n, 0); lf.last = -1; lf.nf = 0; lf.tns_on = false; lf.mts = 0; lf.qmode = 0;
        return;
    }
    const auto& scan = diag_scan(l);
    std::vector<double> r(n), c(n), e(n);
    for (int y = 0; y < s; ++y)
        for (int x = 0; x < s; ++x) r[y * s + x] = org_->at(x0 + x, y0 + y) - pred[y * s + x];
    TxType th, tv;
    tx_pair(lf.mts, s, th, tv);
    forward_2d(th, tv, r.data(), s, s, c.data());
    for (int i = 0; i < n; ++i) e[i] = c[scan[i]];
    if (lf.tns_on) {
        lf.tns = tns_design(e.data(), n, 4, 4);
        if (lf.tns.order == 0) lf.tns_on = false;
        else { std::vector<double> t(n); tns_analysis(lf.tns, e.data(), n, t.data()); e.swap(t); }
    }
    lf.last = -1;
    lf.nf = 0;
    if (lf.qmode == 2) {
        // 利得形状分離 + バンド/パーティション + パラメトリック補完 (§5.3–5.7)
        lf.q.clear();
        lf.tns_on = false;
        std::vector<double> xs(n);
        for (int i = 0; i < n; ++i) xs[i] = e[i] / step_;
        CMModel& gm = md_->gs;
        gs_encode(xs.data(), l, lambda_ / (step_ * step_), gm, plane_ ? 1u : 0u, gs_seed(x0, y0), lf.gs);
        lf.last = gs_nonzero(lf.gs) ? 0 : -1;
        return;
    }
    if (lf.qmode == 0) {
        lf.q.assign(n, 0);
        for (int i = 0; i < n; ++i) {
            const int32_t v = quant_dz(e[i], step_, 1.0 / 3.0);
            lf.q[scan[i]] = v;
            if (v) lf.last = i;
        }
        if (search_.rdoq && md_ && !lf.tns_on && lf.last >= 0) rdoq(lf, l, e, xb_ctx(x0, y0, s), cb_ctx(x0, y0, s));
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
        if (lf.qmode == 2) {
            gs_reconstruct(lf.gs, l, gs_seed(x0, y0), e.data());
            for (int i = 0; i < n; ++i) e[i] *= step_;
        } else if (lf.qmode == 0) {
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
        TxType th, tv;
        tx_pair(lf.mts, s, th, tv);
        inverse_2d(th, tv, c.data(), s, s, r.data());
    }
    if (lf.ns && lf.pt == 2) {
        // インターの高域ノイズ置換: 白色ノイズを 3x3 二項フィルタで高域通過し、
        // 予測ブロックの高域振幅を包絡として掛ける (テクスチャのある場所ほど粒状感を残す)
        SplitMix64 rng(noise_seed_ * 0x9E3779B97F4A7C15ull + static_cast<uint64_t>(x0) * 0x10001ull + static_cast<uint64_t>(y0) + 7);
        std::vector<double> w(static_cast<size_t>(n)), hp(static_cast<size_t>(n)), env(static_cast<size_t>(n));
        for (auto& v : w) v = (rng.next() >> 63) ? 1.0 : -1.0;
        auto blur = [&](const std::vector<double>& a, int x, int y) {
            auto at = [&](int xx, int yy) { return a[std::clamp(yy, 0, s - 1) * s + std::clamp(xx, 0, s - 1)]; };
            return (4 * at(x, y) + 2 * (at(x - 1, y) + at(x + 1, y) + at(x, y - 1) + at(x, y + 1)) + at(x - 1, y - 1) +
                    at(x + 1, y - 1) + at(x - 1, y + 1) + at(x + 1, y + 1)) / 16.0;
        };
        std::vector<double> pd(pred, pred + n);
        double hpe = 0, pm = 0;
        for (int y = 0; y < s; ++y)
            for (int x = 0; x < s; ++x) {
                hp[y * s + x] = w[y * s + x] - blur(w, x, y);
                hpe += hp[y * s + x] * hp[y * s + x];
                env[y * s + x] = std::abs(pd[y * s + x] - blur(pd, x, y));
                pm += env[y * s + x];
            }
        const double norm = std::sqrt(n / std::max(hpe, 1e-9));  // 高域ノイズを単位 RMS に
        pm = pm / n + 1e-9;
        const double amp = lf.ns / 8.0 * step_;
        for (int i = 0; i < n; ++i) r[i] += amp * norm * hp[i] * std::clamp(env[i] / pm, 0.5, 1.5);
    } else if (lf.ns && tband_) {
        // ノイズ置換: 振幅 = ns/8·Δ × 包絡 (参照帯域の |値| / ブロック平均, [0.25, 2] に制限)
        double m = 0;
        for (int y = 0; y < s; ++y)
            for (int x = 0; x < s; ++x) m += std::abs(tband_->at(x0 + x, y0 + y));
        m = m / n + 1e-9;
        SplitMix64 rng(noise_seed_ * 0x9E3779B97F4A7C15ull + static_cast<uint64_t>(x0) * 0x10001ull + static_cast<uint64_t>(y0));
        const double amp = lf.ns / 8.0 * step_;
        for (int y = 0; y < s; ++y)
            for (int x = 0; x < s; ++x) {
                const double env = std::clamp(std::abs(tband_->at(x0 + x, y0 + y)) / m, 0.25, 2.0);
                r[y * s + x] += ((rng.next() >> 63) ? amp : -amp) * env;
            }
    }
    for (int y = 0; y < s; ++y)
        for (int x = 0; x < s; ++x)
            rec_->at(x0 + x, y0 + y) = std::clamp(pred[y * s + x] + static_cast<int32_t>(std::lround(r[y * s + x])), lo_, hi_);
}

double BlockCoder::leaf_bits(const Leaf& lf, int x0, int y0, int l) const {
    double b = 1.0;  // cbf
    if (tools_.ibc) b += 0.5;
    if (inter_.enabled) b += 1.0;
    if (lf.pt == 2) {
        if (!is_luma()) b += 0.0;
        else if (lf.merge >= 0) b += 1.0 + 1.5;
        else {
            b += 1.0 + (inter_.bframe ? 1.5 : 0.0) + (tools_.fir ? 2.0 : 0.0);
            for (int l = 0; l < 2; ++l)
                if ((lf.mi.dir >> l) & 1) {
                    int px, py;
                    mv_pred(x0, y0, l, lf.mi.ref[l], px, py);
                    b += 2.0 + 2.0 * std::log2(1.0 + std::abs(lf.mi.mvx[l] - px)) + 2.0 * std::log2(1.0 + std::abs(lf.mi.mvy[l] - py));
                    if (inter_.nref[l] > 1) b += 1.0 + lf.mi.ref[l];
                }
        }
    } else if (lf.pt == 4) {
        b += 1.0 + 3.0;
    } else if (lf.pt == 3) {
        b += 1.0 + std::log2(2.0 + dict_->ids_for(1 << l).size()) + 2.0 + 2.0 * std::log2(1.0 + std::abs(lf.dict_gain));
    } else if (lf.pt == 1) {
        b += 4.0 + 2.0 * std::log2(1.0 + std::abs(lf.bvx - bv_px_)) + 2.0 * std::log2(1.0 + std::abs(lf.bvy - bv_py_));
    } else if (lf.mode == kModeCfl) {
        b += 1.0 + 2.0 + std::log2(1.0 + std::abs(lf.alpha));
    } else {
        if (tools_.all_angular) {
            int mp[6];
            mpm6(x0, y0, mp);
            b += std::find(mp, mp + 6, lf.mode) != mp + 6 ? 2.5 : 6.0;
        } else {
            int m0, m1;
            mpm(x0, y0, m0, m1);
            b += (lf.mode == m0 || lf.mode == m1) ? 2.0 : 3.0;
        }
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
    // 色差の誤差は数値が小さくても彩度変化として目立つため重み付け (輝度と同 QP 時に約 2 倍)
    const double cw = plane_ ? search_.chroma_weight : 1.0;
    if (search_.psy <= 0) return e * cw;
    // 心理視覚歪み (§13.2): 誤差を 3x3 二項フィルタで低域 e_lp と高域 e_hf に分ける。
    //   D = ||e_lp||² + w_hf ||e_hf||²,  w_hf = 1 / (1 + psy · σ²_org / Δ²)
    // 低域誤差 (ブロック化・構造の崩れ) は常に評価し、高域誤差はテクスチャ部でマスキングする。
    // 平坦部では w_hf ≈ 1 なのでモスキートノイズを抑え、テクスチャ部のノイズ再符号化 (ちらつき) を避ける。
    // マスキングは局所的に: 4x4 セルごとの原画分散の「自身と上下左右セルの最小値」を使う。
    // ブロック全体の分散だと、平坦部上の物体 (エッジ) までテクスチャと見なして構造の誤りを見逃す。
    // 最小値をとるとエッジ (片側が平坦) はマスクされず、全方向に分散が大きい真のテクスチャだけがマスクされる。
    std::vector<double> err(static_cast<size_t>(s) * s);
    for (int y = 0; y < s; ++y)
        for (int x = 0; x < s; ++x) err[y * s + x] = org_->at(x0 + x, y0 + y) - rec_->at(x0 + x, y0 + y);
    const int c4 = std::max(1, s / 4);
    std::vector<double> cv(static_cast<size_t>(c4) * c4, 0.0), wc(cv.size(), 1.0);
    for (int cy = 0; cy < c4; ++cy)
        for (int cx = 0; cx < c4; ++cx) {
            const int n = std::min(4, s);
            double m = 0, v = 0;
            for (int y = 0; y < n; ++y)
                for (int x = 0; x < n; ++x) m += org_->at(x0 + cx * 4 + x, y0 + cy * 4 + y);
            m /= n * n;
            for (int y = 0; y < n; ++y)
                for (int x = 0; x < n; ++x) { const double d = org_->at(x0 + cx * 4 + x, y0 + cy * 4 + y) - m; v += d * d; }
            cv[cy * c4 + cx] = v / (n * n);
        }
    for (int cy = 0; cy < c4; ++cy)
        for (int cx = 0; cx < c4; ++cx) {
            double v = cv[cy * c4 + cx];
            if (cx > 0) v = std::min(v, cv[cy * c4 + cx - 1]);
            if (cx + 1 < c4) v = std::min(v, cv[cy * c4 + cx + 1]);
            if (cy > 0) v = std::min(v, cv[(cy - 1) * c4 + cx]);
            if (cy + 1 < c4) v = std::min(v, cv[(cy + 1) * c4 + cx]);
            wc[cy * c4 + cx] = 1.0 / (1.0 + search_.psy * v / (step_ * step_));
        }
    double lp = 0, hf = 0;
    auto at = [&](int x, int y) { return err[std::clamp(y, 0, s - 1) * s + std::clamp(x, 0, s - 1)]; };
    for (int y = 0; y < s; ++y)
        for (int x = 0; x < s; ++x) {
            const double l = (4 * at(x, y) + 2 * (at(x - 1, y) + at(x + 1, y) + at(x, y - 1) + at(x, y + 1)) +
                              at(x - 1, y - 1) + at(x + 1, y - 1) + at(x - 1, y + 1) + at(x + 1, y + 1)) / 16.0;
            const double h = at(x, y) - l;
            lp += l * l;
            hf += wc[std::min(y / 4, c4 - 1) * c4 + std::min(x / 4, c4 - 1)] * h * h;
        }
    (void)e;
    return (lp + hf) * cw;
}

void BlockCoder::save(int x0, int y0, int s, std::vector<int32_t>& b) const {
    b.resize(static_cast<size_t>(s) * s);
    for (int y = 0; y < s; ++y) std::memcpy(&b[y * s], &rec_->v[(y0 + y) * rec_->w + x0], s * sizeof(int32_t));
}

void BlockCoder::restore(int x0, int y0, int s, const std::vector<int32_t>& b) {
    for (int y = 0; y < s; ++y) std::memcpy(&rec_->v[(y0 + y) * rec_->w + x0], &b[y * s], s * sizeof(int32_t));
}

// ---------------- RD 探索 (符号器) ----------------
// 色差ブロックに対応する輝度領域の動きが一様か (複数の動き/イントラ混在なら false)
bool BlockCoder::luma_motion_uniform(int x0, int y0, int s) const {
    if (!inter_.enabled || is_luma() || !inter_.mf) return true;
    const int cs = inter_.chroma_shift;
    const int W4 = inter_.mf->w4 * 4, H4 = inter_.mf->h4 * 4;
    const MotionInfo& m0 = inter_.mf->at(std::min(x0 << cs, W4 - 1), std::min(y0 << cs, H4 - 1));
    for (int y = 0; y < (s << cs); y += 4)
        for (int x = 0; x < (s << cs); x += 4) {
            const int lx = std::min((x0 << cs) + x, W4 - 1), ly = std::min((y0 << cs) + y, H4 - 1);
            if (!(inter_.mf->at(lx, ly) == m0)) return false;
        }
    return true;
}

double BlockCoder::rd_node(int x0, int y0, int l) {
    const int s = 1 << l, n = s * s;
    // 色差: 対応する輝度の動きが一様でなければ、この大きさの葉は作らず分割する
    // (動物体の周辺で色差だけ大ブロックになり、色の残像・褪色が出るのを防ぐ)
    if (l > min_log2_ && !luma_motion_uniform(x0, y0, s)) {
        split_at(x0, y0, l) = 1;
        const int h = s / 2;
        double js = lambda_ * split_rate(x0, y0, l, 1);
        js += rd_node(x0, y0, l - 1);
        js += rd_node(x0 + h, y0, l - 1);
        js += rd_node(x0, y0 + h, l - 1);
        js += rd_node(x0 + h, y0 + h, l - 1);
        return js;
    }
    std::vector<int32_t> before, best_rec, pred(n);
    save(x0, y0, s, before);

    // 候補生成: 予測 SAD で絞り込み
    std::vector<Leaf> cands;
    {
        std::vector<std::pair<int64_t, int>> sads;
        std::vector<int> modes;
        if (tools_.all_angular) for (int m = 0; m < kNumIntra; ++m) { if (m != kModeCfl) modes.push_back(m); }
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
        if (inter_.enabled) {
            if (is_luma()) {
                const size_t c0 = cands.size();
                motion_search(x0, y0, s, s, cands);
                // FIR 動き基底: 各インター候補 (2Nx2N) に最小二乗の ψ を付けた変種を追加
                if (tools_.firb && s >= 16) {
                    const size_t c1 = cands.size();
                    for (size_t i = c0; i < c1; ++i) {
                        if (cands[i].pt != 2 || cands[i].part != 0) continue;
                        Leaf t = cands[i];
                        if (fir_fit(t, x0, y0, s, t.fir)) cands.push_back(t);
                    }
                }
                if (tools_.rect && s >= 8) rect_search(x0, y0, s, cands);
            }
            else { Leaf lf; lf.pt = 2; cands.push_back(lf); }
        }
        if (tools_.dict && (s == 8 || s == 16)) dict_search(x0, y0, s, cands);
        if (xband_) {
            // 帯域間予測: 隣接帯域の最小二乗ゲインを量子化
            double num = 0, den = 0;
            for (int y = 0; y < s; ++y)
                for (int x = 0; x < s; ++x) {
                    const double r = xband_->at(x0 + x, y0 + y);
                    num += r * org_->at(x0 + x, y0 + y);
                    den += r * r;
                }
            if (den > 0) {
                const int g = std::clamp(static_cast<int>(std::lround(num / den * 4.0)), -4, 4);
                // 帯域間の値の相関はほぼ 0 (bandviz 実測) のため候補にしない。構文は互換のため残す
                (void)g;
            }
        }
        if (tband_) {
            // 時間方向パラメトリック予測: ゲイン 1.0 と最小二乗ゲイン
            double num = 0, den = 0;
            for (int y = 0; y < s; ++y)
                for (int x = 0; x < s; ++x) {
                    const double r = tband_->at(x0 + x, y0 + y);
                    num += r * org_->at(x0 + x, y0 + y);
                    den += r * r;
                }
            Leaf lf; lf.pt = 4; lf.xsrc = 1; lf.xgain = 4;
            cands.push_back(lf);
            if (den > 0) {
                const int g = std::clamp(static_cast<int>(std::lround(num / den * 4.0)), 1, 5);
                if (g != 4) { lf.xgain = g; cands.push_back(lf); }
            }
        }
        if (tools_.ibc && s >= 8 && s <= 32) {
            Leaf lf; lf.pt = 1;
            ibc_search(x0, y0, s, lf.bvx, lf.bvy);
            if (lf.bvx || lf.bvy) cands.push_back(lf);
        }
    }
    // 候補ごとの予測を一度だけ計算して以降で再利用
    std::vector<std::vector<int32_t>> preds(cands.size());
    for (size_t i = 0; i < cands.size(); ++i) {
        preds[i].resize(static_cast<size_t>(n));
        predict(cands[i], x0, y0, l, preds[i].data());
    }
    auto keep_only = [&](const std::vector<size_t>& idx) {
        std::vector<Leaf> c2;
        std::vector<std::vector<int32_t>> p2;
        for (size_t i : idx) { c2.push_back(cands[i]); p2.push_back(std::move(preds[i])); }
        cands.swap(c2);
        preds.swap(p2);
    };
    // 高速化: インターで十分よい候補 (残差なし相当) があればイントラ候補を省く
    if (inter_.enabled && search_.inter_skip_intra && is_luma()) {
        int64_t best_sad = INT64_MAX;
        for (size_t i = 0; i < cands.size(); ++i)
            if (cands[i].pt == 2) {
                int64_t sad = 0;
                for (int y = 0; y < s; ++y)
                    for (int x = 0; x < s; ++x) sad += std::abs(org_->at(x0 + x, y0 + y) - preds[i][y * s + x]);
                best_sad = std::min(best_sad, sad);
            }
        if (best_sad < static_cast<int64_t>(step_ * 0.5 * n)) {
            std::vector<size_t> idx;
            for (size_t i = 0; i < cands.size(); ++i) if (cands[i].pt == 2) idx.push_back(i);
            keep_only(idx);
        }
    }
    // 高速化: SATD 上位 K 候補のみ RD 評価
    if (search_.max_rd_cands > 0 && static_cast<int>(cands.size()) > search_.max_rd_cands) {
        std::vector<std::pair<double, size_t>> rank;
        std::vector<int32_t> res(static_cast<size_t>(n));
        const double lsad = std::sqrt(lambda_);
        for (size_t i = 0; i < cands.size(); ++i) {
            for (int y = 0; y < s; ++y)
                for (int x = 0; x < s; ++x) res[y * s + x] = org_->at(x0 + x, y0 + y) - preds[i][y * s + x];
            Leaf t = cands[i];
            t.last = -1;
            rank.push_back({static_cast<double>(satd4(res.data(), s)) + lsad * leaf_bits(t, x0, y0, l), i});
        }
        std::stable_sort(rank.begin(), rank.end());
        std::vector<size_t> idx;
        for (int k = 0; k < search_.max_rd_cands; ++k) idx.push_back(rank[k].second);
        keep_only(idx);
    }
    // 量子化モード/TNS の組み合わせ
    double best = 1e300;
    Leaf best_leaf;
    for (size_t ci = 0; ci < cands.size(); ++ci) {
        const Leaf& base = cands[ci];
        std::copy(preds[ci].begin(), preds[ci].end(), pred.begin());
        const int nmts = (tools_.mts && search_.try_mts && is_luma() && s <= 32) ? (s <= 16 ? 9 : 5) : 1;
        for (int mt = 0; mt < nmts; ++mt)
        for (int qm = 0; qm <= 2; ++qm)
            for (int tn = 0; tn <= (tools_.tns && search_.try_tns && s >= 8 && qm == 0 ? 1 : 0); ++tn) {
                if (qm == 1 && !(tools_.e8 && search_.try_e8)) continue;
                if (qm == 2 && !(tools_.gs && md_ && !tools_.pred_only)) continue;
                Leaf lf = base;
                lf.qmode = qm;
                lf.mts = mt;
                lf.tns_on = tn;
                quantize(lf, x0, y0, l, pred.data());
                if (tn && !lf.tns_on) continue;
                reconstruct(lf, x0, y0, l, pred.data());
                double dist;
                if (tools_.pred_only) {
                    // 予測のみの段階: 残差は後段 (帯域符号化) で変換符号化されるため、
                    // 残差の SATD を符号量+歪みの近似として評価する (SSE だと残差コストを無視してしまう)
                    std::vector<int32_t> res(static_cast<size_t>(n));
                    for (int y = 0; y < s; ++y)
                        for (int x = 0; x < s; ++x) res[y * s + x] = org_->at(x0 + x, y0 + y) - rec_->at(x0 + x, y0 + y);
                    dist = static_cast<double>(satd4(res.data(), s)) * std::sqrt(lambda_) * 2.0;
                } else {
                    dist = sse(x0, y0, s);
                }
                const double j = dist + lambda_ * (leaf_rate(lf, x0, y0, l) + split_rate(x0, y0, l, 0));
                if (j < best) { best = j; best_leaf = lf; save(x0, y0, s, best_rec); }
                restore(x0, y0, s, before);
            }
    }
    leaf_at(x0, y0, l) = best_leaf;
    restore(x0, y0, s, best_rec);
    fill_mf(best_leaf, x0, y0, s);
    set_modes4(x0, y0, s, best_leaf.pt ? kModePlanar : best_leaf.mode);
    if (l <= min_log2_) return best;
    if (best < lambda_ * 4.0 || (search_.skip_split_on_skip && best_leaf.pt == 2 && best_leaf.last < 0)) {
        split_at(x0, y0, l) = 0;
        return best;
    }
    restore(x0, y0, s, before);
    const int sbx = bv_px_, sby = bv_py_;
    const int h = s / 2;
    double js = lambda_ * split_rate(x0, y0, l, 1);
    js += rd_node(x0, y0, l - 1);
    js += rd_node(x0 + h, y0, l - 1);
    js += rd_node(x0, y0 + h, l - 1);
    js += rd_node(x0 + h, y0 + h, l - 1);
    if (best <= js) {
        split_at(x0, y0, l) = 0;
        restore(x0, y0, s, best_rec);
        set_modes4(x0, y0, s, best_leaf.pt ? kModePlanar : best_leaf.mode);
        fill_mf(best_leaf, x0, y0, s);
        bv_px_ = sbx; bv_py_ = sby;
        if (best_leaf.pt == 1) { bv_px_ = best_leaf.bvx; bv_py_ = best_leaf.bvy; }
        return best;
    }
    split_at(x0, y0, l) = 1;
    return js;
}

// ---------------- 構文 (符号器/復号器共通) ----------------
void BlockCoder::leaf_syntax(SymIO& io, Models& md, Leaf& lf, int x0, int y0, int l) {
    const int s = 1 << l, n = s * s;
    const uint32_t L = static_cast<uint32_t>(l), pc = plane_ ? 1u : 0u;
    int is_inter = 0;
    if (inter_.enabled) is_inter = io.bit(md.inter, 0, L, pc, lf.pt == 2);
    if (is_inter) {
        lf.pt = 2;
        if (is_luma()) {
            int part = 0;
            if (tools_.rect && s >= 8) {
                const int is_part = io.bit(md.inter, 12, L, 0, lf.part != 0);
                part = is_part ? 1 + io.bit(md.inter, 13, L, 0, lf.part == 2) : 0;
            }
            lf.part = part;
            if (part == 0) {
                code_motion(io, md, L, x0, y0, s, s, lf.mi, lf.merge);
                if (tools_.firb && s >= 16) {
                    const int on = io.bit(md.inter, 40, L, 0, lf.fir[0] || lf.fir[1] || lf.fir[2] || lf.fir[3]);
                    for (int k = 0; k < 4; ++k) {
                        lf.fir[k] = on ? io.sint(md.inter, 41 + static_cast<uint32_t>(k), L, lf.fir[k]) : 0;
                        if (std::abs(lf.fir[k]) > 16) throw std::runtime_error("corrupt stream: fir psi");
                    }
                } else {
                    for (int& v : lf.fir) v = 0;
                }
            } else {
                const int pw = part == 2 ? s / 2 : s, ph = part == 1 ? s / 2 : s;
                code_motion(io, md, L, x0, y0, pw, ph, lf.mi, lf.merge);
                // 第 2 区画のマージ/予測ベクトルは第 1 区画の動きを参照する
                if (!io.cost) inter_.mf->fill(x0, y0, pw, ph, lf.mi);
                code_motion(io, md, L, x0 + (part == 2 ? pw : 0), y0 + (part == 1 ? ph : 0), pw, ph, lf.mi2, lf.merge2);
            }
        }
    } else if ((xband_ || tband_) && io.bit(md.dict, 20, L, 0, lf.pt == 4)) {
        lf.pt = 4;
        if (xband_ && tband_) lf.xsrc = io.bit(md.dict, 22, L, 0, lf.xsrc);
        else lf.xsrc = tband_ ? 1 : 0;
        if (lf.xsrc) {
            lf.xgain = 4 + io.sint(md.dict, 23, L, lf.xgain - 4);
            if (lf.xgain < 1 || lf.xgain > 5) throw std::runtime_error("corrupt stream: tband gain");
        } else {
            lf.xgain = io.sint(md.dict, 21, L, lf.xgain);
            if (lf.xgain == 0 || std::abs(lf.xgain) > 4) throw std::runtime_error("corrupt stream: xband gain");
        }
    } else if (tools_.dict && (s == 8 || s == 16) && io.bit(md.dict, 0, L, 0, lf.pt == 3)) {
        lf.pt = 3;
        const auto& ids = dict_->ids_for(s);
        lf.dict_idx = static_cast<int>(io.uint(md.dict, 1, L, static_cast<uint32_t>(lf.dict_idx)));
        if (lf.dict_idx >= static_cast<int>(ids.size())) throw std::runtime_error("corrupt stream: dict index");
        lf.dict_gain = io.sint(md.dict, 2, L, lf.dict_gain);
        if (std::abs(lf.dict_gain) > 64) throw std::runtime_error("corrupt stream: dict gain");
    } else {
    if (lf.pt >= 2) lf.pt = 0;
    if (tools_.ibc) lf.pt = io.bit(md.ibc, 0, L, pc, lf.pt);
    if (lf.pt == 1) {
        lf.bvx = bv_px_ + io.sint(md.ibc, 1, L, lf.bvx - bv_px_);
        lf.bvy = bv_py_ + io.sint(md.ibc, 2, L, lf.bvy - bv_py_);
        if (!ibc_valid(x0 + lf.bvx, y0 + lf.bvy, s)) throw std::runtime_error("corrupt stream: block vector");
        if (!io.cost) { bv_px_ = lf.bvx; bv_py_ = lf.bvy; }
    } else {
        int is_cfl = 0;
        if (tools_.cfl) is_cfl = io.bit(md.cfl, 0, L, 0, lf.mode == kModeCfl);
        if (is_cfl) {
            lf.mode = kModeCfl;
            lf.alpha = io.sint(md.cfl, 1, L, lf.alpha);
            if (lf.alpha < -16 || lf.alpha > 16) throw std::runtime_error("corrupt stream: cfl alpha");
        } else {
            if (tools_.all_angular) {
                // 6 候補 MPM: 切り詰め単進のインデックス、それ以外は残り 31 モードを 5 ビット (二分木文脈)
                int mp[6];
                mpm6(x0, y0, mp);
                int idx = -1;
                for (int i = 0; i < 6; ++i) if (mp[i] == lf.mode) idx = i;
                const int is_mpm = io.bit(md.mode, 0, L, pc, idx >= 0);
                if (is_mpm) {
                    int k = 0;
                    while (k < 5 && io.bit(md.mode, 10 + static_cast<uint32_t>(k), L, pc, idx > k)) ++k;
                    lf.mode = mp[k];
                } else {
                    std::vector<int> rest;
                    for (int m = 0; m < kNumIntra; ++m)
                        if (m != kModeCfl && std::find(mp, mp + 6, m) == mp + 6) rest.push_back(m);
                    int r = 0;
                    if (io.enc) r = static_cast<int>(std::find(rest.begin(), rest.end(), lf.mode) - rest.begin());
                    uint32_t tree = 1;
                    int val = 0;
                    for (int b = 4; b >= 0; --b) {
                        const int bt = io.bit(md.mode, 32 + tree, L, pc, (r >> b) & 1);
                        tree = (tree << 1) | static_cast<uint32_t>(bt);
                        val = (val << 1) | bt;
                    }
                    if (val >= static_cast<int>(rest.size())) throw std::runtime_error("corrupt stream: intra mode rest");
                    lf.mode = rest[val];
                }
            } else {
            int m0, m1;
            mpm(x0, y0, m0, m1);
            const int is_mpm = io.bit(md.mode, 0, L, pc, lf.mode == m0 || lf.mode == m1);
            if (is_mpm) {
                lf.mode = io.bit(md.mode, 1, L, pc, lf.mode == m1) ? m1 : m0;
            } else {
                static constexpr int kSet[4] = {kModeDC, kModePlanar, kModeHor, kModeVer};
                int idx = 0;
                for (int i = 0; i < 4; ++i) if (kSet[i] == lf.mode) idx = i;
                idx = io.bit(md.mode, 3, 0, pc, idx >> 1) * 2;
                idx += io.bit(md.mode, 4, 0, pc, io.enc ? (lf.mode == kSet[idx + 1]) : 0);
                lf.mode = kSet[idx];
            }
            }
            if (lf.mode < 0 || lf.mode >= kNumIntra || lf.mode == kModeCfl) throw std::runtime_error("corrupt stream: intra mode");
        }
    }
    }

    const uint32_t cbc = cb_ctx(x0, y0, s);
    const int cbf = tools_.pred_only ? 0 : io.bit(md.cbf, 0, L | (cbc << 4), pc * 64 + static_cast<uint32_t>(lf.pt ? 60 + lf.pt : lf.mode), lf.last >= 0);
    if (!cbf) {
        lf.last = -1;
        lf.q.assign(lf.qmode == 0 ? n : 0, 0);
        lf.tns_on = false;
        lf.nf = 0;
        lf.mts = 0;
        // 時間方向予測で残差なし: 高域ノイズを強さだけで置換 (包絡は参照帯域の振幅から)
        if (ns_allowed(lf)) {
            lf.ns = static_cast<int>(io.uint(md.nf, 8 + (lf.pt == 2 ? 1u : 0u), cbc, static_cast<uint32_t>(lf.ns)));
            if (lf.ns > 7) throw std::runtime_error("corrupt stream: band noise");
        } else {
            lf.ns = 0;
        }
    } else {
        lf.ns = 0;
        if (tools_.mts && is_luma() && s <= 32) {
            int m = io.bit(md.e8, 5, L, static_cast<uint32_t>(lf.pt), lf.mts != 0);
            if (m) {
                // 1..4: DST7/DCT8 の組, 5..8 (16x16 以下): DCT2 混合と恒等変換
                const int ext = s <= 16 ? io.bit(md.e8, 8, L, 0, lf.mts >= 5) : 0;
                const int v = ext ? lf.mts - 5 : lf.mts - 1;
                const int hi2 = io.bit(md.e8, 6 + 3 * ext, L, 0, v >> 1);
                const int lo2 = io.bit(md.e8, 7 + 3 * ext, L, static_cast<uint32_t>(hi2), v & 1);
                m = 1 + ext * 4 + hi2 * 2 + lo2;
            }
            lf.mts = m;
        } else {
            lf.mts = 0;
        }
        if (tools_.e8 || tools_.gs) {
            const int nzq = io.bit(md.e8, 0, L, pc, lf.qmode != 0);
            if (!nzq) lf.qmode = 0;
            else if (tools_.e8 && tools_.gs) lf.qmode = 1 + io.bit(md.e8, 12, L, pc, lf.qmode == 2);
            else lf.qmode = tools_.e8 ? 1 : 2;
        } else {
            lf.qmode = 0;
        }
        if (tools_.tns && s >= 8 && lf.qmode != 2) {
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
        if (lf.qmode == 2) {
            if (lf.tns_on) throw std::runtime_error("corrupt stream: gs with tns");
            gs_syntax(io, md.gs, lf.gs, l, pc);
            lf.last = 0;
            lf.nf = 0;
        } else if (lf.qmode == 0) {
            lf.last = static_cast<int>(io.uintc(md.last, L, pc, static_cast<uint32_t>(lf.last)));
            if (lf.last >= n) throw std::runtime_error("corrupt stream: last");
            if (!io.enc) lf.q.assign(n, 0);
            CMModel& cm = plane_ ? md.coef_c : md.coef_y;
            const uint32_t xbc = xb_ctx(x0, y0, s);
            for (int i = 0; i <= lf.last; ++i) {
                const int pos = scan[i];
                const int u = pos & (s - 1), v = pos >> l;
                const uint32_t a = coef_ctx(lf.q.data(), u, v, l);
                const uint32_t b = (L * 4 + (i == lf.last ? 1u : 0u) + (i == 0 ? 2u : 0u) + (lf.tns_on ? 64u : 0u)) | (xbc << 8) | (cbc << 11);
                if (i == lf.last) {
                    const int32_t cv = lf.q[pos];
                    const uint32_t mag = io.uint(cm, a, b, io.enc ? static_cast<uint32_t>(std::abs(cv)) - 1u : 0u) + 1u;
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
            lf.last = static_cast<int>(io.uintc(md.last, L, pc + 2, static_cast<uint32_t>(lf.last)));
            if ((lf.last + 1) * 8 > std::max(n, 8)) throw std::runtime_error("corrupt stream: e8 chunks");
            if (!io.enc) lf.q.assign(static_cast<size_t>(lf.last + 1) * 8, 0);
            int32_t prev_l1 = 0;
            for (int ch = 0; ch <= lf.last; ++ch) {
                double y[8];
                for (int k = 0; k < 8; ++k) y[k] = lf.q[ch * 8 + k] * 0.5;
                LatticeIndex idx;
                if (io.enc) idx = lattice_to_index(Lattice::E8, y, 8);
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
                if (!io.enc) {
                    index_to_lattice(Lattice::E8, idx, 8, y);
                    for (int k = 0; k < 8; ++k) lf.q[ch * 8 + k] = static_cast<int32_t>(std::lround(2.0 * y[k]));
                }
            }
        }
    }
}

void BlockCoder::code_motion(SymIO& io, Models& md, uint32_t L, int x0, int y0, int w, int h, MotionInfo& m, int& merge) {
    MotionInfo ml[kMaxMerge];
    const int nm = merge_list(x0, y0, w, h, ml);
    if (io.enc && merge >= 0 && !(merge < nm && ml[merge] == m)) merge = -1;  // 候補変化時は明示
    if (io.enc && merge < 0)
        for (int i = 0; i < nm; ++i) if (ml[i] == m) { merge = i; break; }
    const int is_merge = nm > 0 ? io.bit(md.inter, 1, L, 0, merge >= 0) : 0;
    if (is_merge) {
        int idx = 0;
        while (idx < nm - 1 && io.bit(md.inter, 20 + idx, 0, 0, merge > idx)) ++idx;  // 切り詰め単進
        merge = idx;
        m = ml[idx];
        return;
    }
    merge = -1;
    if (inter_.bframe) {
        const int bi = io.bit(md.inter, 4, L, 0, m.dir == 3);
        m.dir = bi ? 3 : (io.bit(md.inter, 5, L, 0, m.dir == 2) ? 2 : 1);
    } else {
        m.dir = 1;
    }
    for (int li = 0; li < 2; ++li) {
        if (!((m.dir >> li) & 1)) { m.ref[li] = 0; m.mvx[li] = m.mvy[li] = 0; continue; }
        if (inter_.nref[li] > 1) m.ref[li] = static_cast<int8_t>(io.uint(md.inter, 6 + li, 0, static_cast<uint32_t>(m.ref[li])));
        if (m.ref[li] < 0 || m.ref[li] >= inter_.nref[li]) throw std::runtime_error("corrupt stream: ref idx");
        int px, py;
        mv_pred(x0, y0, li, m.ref[li], px, py);
        const int vx = px + io.sint(md.mvd, 0, static_cast<uint32_t>(li), m.mvx[li] - px);
        const int vy = py + io.sint(md.mvd, 1, static_cast<uint32_t>(li), m.mvy[li] - py);
        if (std::abs(vx) > 32000 || std::abs(vy) > 32000) throw std::runtime_error("corrupt stream: mv");
        m.mvx[li] = static_cast<int16_t>(vx);
        m.mvy[li] = static_cast<int16_t>(vy);
    }
    if (tools_.fir) {
        int psi = io.bit(md.inter, 8, 0, 0, m.psi >> 1) << 1;
        psi |= io.bit(md.inter, 9 + (psi >> 1), 0, 0, m.psi & 1);
        m.psi = static_cast<uint8_t>(psi);
    } else {
        m.psi = 0;
    }
}

void BlockCoder::fill_mf(const Leaf& lf, int x0, int y0, int s) {
    if (!(inter_.enabled && is_luma())) return;
    if (lf.pt != 2) { inter_.mf->fill(x0, y0, s, s, MotionInfo{}); return; }
    if (lf.part == 0) { inter_.mf->fill(x0, y0, s, s, lf.mi); return; }
    const int pw = lf.part == 2 ? s / 2 : s, ph = lf.part == 1 ? s / 2 : s;
    inter_.mf->fill(x0, y0, pw, ph, lf.mi);
    inter_.mf->fill(x0 + (lf.part == 2 ? pw : 0), y0 + (lf.part == 1 ? ph : 0), pw, ph, lf.mi2);
}

void BlockCoder::code_leaf(SymIO& io, Models& md, Leaf& lf, int x0, int y0, int l) {
    const int s = 1 << l, n = s * s;
    std::vector<int32_t> pred(n);
    if (io.enc) {
        predict(lf, x0, y0, l, pred.data());
        quantize(lf, x0, y0, l, pred.data());  // 再構成ずれを防ぐため現在の再構成から再量子化
        lf.ns = 0;
        if (ns_allowed(lf) && lf.last < 0) {
            // 失われた残差の (インターは高域成分の) エネルギーを測る
            std::vector<double> d(static_cast<size_t>(n));
            for (int y = 0; y < s; ++y)
                for (int x = 0; x < s; ++x) d[y * s + x] = org_->at(x0 + x, y0 + y) - pred[y * s + x];
            if (lf.pt == 2) {
                std::vector<double> lp(static_cast<size_t>(n));
                auto at = [&](int x, int y) { return d[std::clamp(y, 0, s - 1) * s + std::clamp(x, 0, s - 1)]; };
                for (int y = 0; y < s; ++y)
                    for (int x = 0; x < s; ++x)
                        lp[y * s + x] = (4 * at(x, y) + 2 * (at(x - 1, y) + at(x + 1, y) + at(x, y - 1) + at(x, y + 1)) +
                                         at(x - 1, y - 1) + at(x + 1, y - 1) + at(x - 1, y + 1) + at(x + 1, y + 1)) / 16.0;
                for (int i = 0; i < n; ++i) d[i] -= lp[i];
            }
            double e = 0;
            for (double v : d) e += v * v;
            // 失われた残差エネルギーの 7 割をノイズで補う (知覚上のざらつきを保ちつつ過剰にしない)
            lf.ns = std::clamp(static_cast<int>(std::lround(0.7 * std::sqrt(e / n) / (step_ / 8.0))), 0, 7);
        }
    }
    leaf_syntax(io, md, lf, x0, y0, l);
    if (!io.enc) predict(lf, x0, y0, l, pred.data());
    reconstruct(lf, x0, y0, l, pred.data());
    {
        const int gw = rec_->w / 4;
        const uint8_t fl = static_cast<uint8_t>((lf.pt != 2 ? 1 : 0) | (lf.last >= 0 || lf.nf ? 2 : 0));
        for (int y = y0 / 4; y < (y0 + s) / 4; ++y)
            for (int x = x0 / 4; x < (x0 + s) / 4; ++x) { leaf4_[y * gw + x] = leaf_counter_; flags4_[y * gw + x] = fl; }
        ++leaf_counter_;
    }
    set_modes4(x0, y0, s, lf.pt ? kModePlanar : lf.mode);
    fill_mf(lf, x0, y0, s);
    if (io.w) {  // 符号器の統計 (輝度面積)
        const uint64_t a = static_cast<uint64_t>(s) * s;
        if (plane_ == 0) {
            ++usage_.leaves;
            usage_.size[std::clamp(l - 2, 0, 6)] += a;
            if (lf.pt == 2) {
                usage_.inter += a;
                if (lf.part) usage_.rect += a;
                else if (lf.merge >= 0) { usage_.merge += a; if (lf.last < 0) usage_.skip += a; }
                if (lf.mi.dir == 3) usage_.bi += a;
            } else if (lf.pt == 1) usage_.ibc += a;
            else if (lf.pt == 3) usage_.dict += a;
            else usage_.intra += a;
            if (lf.tns_on) usage_.tns += a;
            if (lf.qmode == 1 && lf.last >= 0) usage_.e8 += a;
            if (lf.qmode == 2 && lf.last >= 0) usage_.gs += a;
        } else if (lf.pt == 0 && lf.mode == kModeCfl) {
            usage_.cfl += a;
        }
    }
}

double BlockCoder::leaf_rate(Leaf& lf, int x0, int y0, int l) {
    if (!md_) return leaf_bits(lf, x0, y0, l);
    double c = 0;
    SymIO io;
    io.cost = &c;
    io.enc = true;
    Leaf t = lf;
    leaf_syntax(io, *md_, t, x0, y0, l);
    return c;
}

double BlockCoder::split_rate(int x0, int y0, int l, int split) {
    if (l <= min_log2_) return 0.0;
    if (!md_) return 1.0;
    double c = 0;
    SymIO io;
    io.cost = &c;
    io.enc = true;
    io.bit(md_->split, 0, static_cast<uint32_t>(l), static_cast<uint32_t>(plane_), split);
    (void)x0; (void)y0;
    return c;
}

void BlockCoder::code_node(SymIO& io, Models& md, int x0, int y0, int l) {
    int split = 0;
    if (l > min_log2_) split = io.bit(md.split, 0, static_cast<uint32_t>(l), static_cast<uint32_t>(plane_), io.enc ? split_at(x0, y0, l) : 0);
    if (split) {
        const int h = 1 << (l - 1);
        code_node(io, md, x0, y0, l - 1);
        code_node(io, md, x0 + h, y0, l - 1);
        code_node(io, md, x0, y0 + h, l - 1);
        code_node(io, md, x0 + h, y0 + h, l - 1);
        return;
    }
    Leaf lf;
    if (io.enc) lf = leaf_at(x0, y0, l);
    code_leaf(io, md, lf, x0, y0, l);
}

void BlockCoder::enable_aqp(int base_qp, int bit_depth, int (*fn)(void*, int, int), void* ctx) {
    aqp_ = true;
    base_qp_ = base_qp;
    bit_depth_ = bit_depth;
    dqp_fn_ = fn;
    dqp_ctx_ = ctx;
}

void BlockCoder::code_ctu(SymIO& io, Models& md, int cx, int cy, int ctu) {
    cx_ = cx; cy_ = cy; ctu_ = ctu;
    md_ = &md;
    if (aqp_) {
        int dqp = (io.enc && dqp_fn_) ? dqp_fn_(dqp_ctx_, cx, cy) : 0;
        dqp = io.sint(md.aqp, static_cast<uint32_t>(std::min(std::abs(prev_dqp_), 7)), static_cast<uint32_t>(plane_ ? 1 : 0), dqp);
        if (dqp < -12 || dqp > 12) throw std::runtime_error("corrupt stream: dqp");
        prev_dqp_ = dqp;
        const int q = std::clamp(base_qp_ + dqp, 0, 63);
        step_ = qp_step(q, bit_depth_);
        lambda_ = 0.57 * std::pow(2.0, (q - 12) / 3.0) * std::pow(4.0, bit_depth_ - 8);
    }
    const int bs = 1 << max_log2_;
    for (int by = cy; by < cy + ctu; by += bs)
        for (int bx = cx; bx < cx + ctu; bx += bs) {
            if (io.enc) {
                // RD 探索は再構成・MPM・BV 予測子を変更するので、符号化前に状態を戻す
                std::vector<int32_t> rb;
                save(bx, by, bs, rb);
                const std::vector<int8_t> m4 = modes4_;
                const int px = bv_px_, py = bv_py_;
                const bool mfs = inter_.enabled && is_luma();
                std::vector<MotionInfo> mfb;
                MotionField* mf = inter_.mf;
                if (mfs)
                    for (int y = by / 4; y < (by + bs) / 4; ++y)
                        mfb.insert(mfb.end(), mf->mi.begin() + y * mf->w4 + bx / 4, mf->mi.begin() + y * mf->w4 + (bx + bs) / 4);
                rd_node(bx, by, max_log2_);
                restore(bx, by, bs, rb);
                if (mfs)
                    for (int y = by / 4, k = 0; y < (by + bs) / 4; ++y, k += bs / 4)
                        std::copy(mfb.begin() + k, mfb.begin() + k + bs / 4, mf->mi.begin() + y * mf->w4 + bx / 4);
                modes4_ = m4;
                bv_px_ = px; bv_py_ = py;
            }
            code_node(io, md, bx, by, max_log2_);
        }
}

}  // namespace fvc

namespace fvc {
double BlockCoder::rd_ctu(int cx, int cy, int ctu) {
    cx_ = cx; cy_ = cy; ctu_ = ctu;
    md_ = nullptr;  // 図形採否の比較用: 近似レート
    const int bs = 1 << max_log2_;
    double j = 0;
    for (int by = cy; by < cy + ctu; by += bs)
        for (int bx = cx; bx < cx + ctu; bx += bs) j += rd_node(bx, by, max_log2_);
    return j;
}
}  // namespace fvc
