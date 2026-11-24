#include "hf_layer.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "block_coder.hpp"
#include "fvc/entropy.hpp"
#include "fvc/quant.hpp"

namespace fvc {

namespace {
constexpr int kBands = 4;
constexpr double kBandEdge[kBands - 1] = {1.0, 2.5, 6.0};  // 局所活動度 (8 ビット換算の RMS) の段の境界
constexpr int kNa = 9, kNg = 9;
constexpr double kAlpha[kNa] = {0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0};
constexpr double kGain[kNg] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.5, 7.5};  // 足す質感の RMS (8 ビット換算)

using Img = std::vector<double>;

// 5 タップ二項フィルタ [1 4 6 4 1]/16 の分離畳み込み (端は延長)
Img blur5(const Img& a, int w, int h) {
    static constexpr double k[5] = {1 / 16.0, 4 / 16.0, 6 / 16.0, 4 / 16.0, 1 / 16.0};
    Img t(a.size()), o(a.size());
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            double s = 0;
            for (int j = -2; j <= 2; ++j) s += k[j + 2] * a[static_cast<size_t>(y) * w + std::clamp(x + j, 0, w - 1)];
            t[static_cast<size_t>(y) * w + x] = s;
        }
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            double s = 0;
            for (int j = -2; j <= 2; ++j) s += k[j + 2] * t[static_cast<size_t>(std::clamp(y + j, 0, h - 1)) * w + x];
            o[static_cast<size_t>(y) * w + x] = s;
        }
    return o;
}
// 半径 r の箱平均 (積分画像)
Img box(const Img& a, int w, int h, int r) {
    std::vector<double> S(static_cast<size_t>(w + 1) * (h + 1), 0.0);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            S[static_cast<size_t>(y + 1) * (w + 1) + x + 1] = a[static_cast<size_t>(y) * w + x] + S[static_cast<size_t>(y) * (w + 1) + x + 1] +
                                                              S[static_cast<size_t>(y + 1) * (w + 1) + x] - S[static_cast<size_t>(y) * (w + 1) + x];
    Img o(a.size());
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const int x0 = std::max(0, x - r), x1 = std::min(w, x + r + 1), y0 = std::max(0, y - r), y1 = std::min(h, y + r + 1);
            const double s = S[static_cast<size_t>(y1) * (w + 1) + x1] - S[static_cast<size_t>(y0) * (w + 1) + x1] -
                             S[static_cast<size_t>(y1) * (w + 1) + x0] + S[static_cast<size_t>(y0) * (w + 1) + x0];
            o[static_cast<size_t>(y) * w + x] = s / ((x1 - x0) * (y1 - y0));
        }
    return o;
}

// 復号器と符号器で共通の解析: U の詳細 D、活動度の段、局所の向き (0: 等方, 1..4)、向きの確からしさ
struct Analysis {
    Img u, d;
    std::vector<uint8_t> band, ori;
    Img coh;
    Img gm;  // 平滑化した U の勾配の大きさ
};
Analysis analyze(const Plane& U, int bit_depth) {
    const int w = U.w, h = U.h;
    const double sc = std::ldexp(1.0, bit_depth - 8);
    Analysis A;
    A.u.assign(U.v.begin(), U.v.end());
    const Img b = blur5(A.u, w, h);
    A.d.resize(A.u.size());
    A.gm.resize(A.u.size());
    Img d2(A.u.size()), jxx(A.u.size()), jyy(A.u.size()), jxy(A.u.size());
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const size_t i = static_cast<size_t>(y) * w + x;
            A.d[i] = A.u[i] - b[i];
            d2[i] = A.d[i] * A.d[i];
            const double gx = (b[static_cast<size_t>(y) * w + std::min(x + 1, w - 1)] - b[static_cast<size_t>(y) * w + std::max(x - 1, 0)]) * 0.5;
            const double gy = (b[static_cast<size_t>(std::min(y + 1, h - 1)) * w + x] - b[static_cast<size_t>(std::max(y - 1, 0)) * w + x]) * 0.5;
            jxx[i] = gx * gx; jyy[i] = gy * gy; jxy[i] = gx * gy;
            A.gm[i] = std::sqrt(gx * gx + gy * gy);
        }
    const Img act = box(d2, w, h, 4), sxx = box(jxx, w, h, 3), syy = box(jyy, w, h, 3), sxy = box(jxy, w, h, 3);
    A.band.resize(A.u.size());
    A.ori.resize(A.u.size());
    A.coh.resize(A.u.size());
    for (size_t i = 0; i < A.u.size(); ++i) {
        const double a = std::sqrt(act[i]) / sc;
        int bnd = 0;
        while (bnd < kBands - 1 && a >= kBandEdge[bnd]) ++bnd;
        A.band[i] = static_cast<uint8_t>(bnd);
        const double tr = sxx[i] + syy[i];
        const double c = tr > 1e-9 ? std::sqrt((sxx[i] - syy[i]) * (sxx[i] - syy[i]) + 4 * sxy[i] * sxy[i]) / tr : 0.0;
        A.coh[i] = c;
        double ang = 0.5 * std::atan2(2 * sxy[i], sxx[i] - syy[i]) * 180.0 / 3.14159265358979323846;  // 勾配の向き
        if (ang < 0) ang += 180.0;
        A.ori[i] = static_cast<uint8_t>(1 + static_cast<int>(std::lround(ang / 45.0)) % 4);
    }
    return A;
}

// 向き o (1..4) の縞: 勾配の向き (o-1)·45° に垂直な方向へ 5 タップで平均した雑音の高域。o = 0 は等方。RMS 1 に正規化
Img tex_field(const Img& noise, int w, int h, int o) {
    Img s = noise;
    if (o > 0) {
        static constexpr int kDx[4] = {0, -1, 1, 1}, kDy[4] = {1, 1, 0, 1};
        const int dx = kDx[o - 1], dy = kDy[o - 1];
        static constexpr double kw[5] = {0.5, 1.0, 1.0, 1.0, 0.5};
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                double a = 0;
                for (int t = -2; t <= 2; ++t)
                    a += kw[t + 2] * noise[static_cast<size_t>(std::clamp(y + t * dy, 0, h - 1)) * w + std::clamp(x + t * dx, 0, w - 1)];
                s[static_cast<size_t>(y) * w + x] = a / 4.0;
            }
    }
    const Img b = blur5(s, w, h);
    double e = 0;
    for (size_t i = 0; i < s.size(); ++i) { s[i] -= b[i]; e += s[i] * s[i]; }
    const double r = e > 0 ? std::sqrt(s.size() / e) : 0.0;
    for (double& v : s) v *= r;
    return s;
}

struct Params { int dr = 0, a[kBands] = {}, g[kBands] = {}; };

// デリンギング (Lee のシグマフィルタ, 1983): 強い輪郭の近く (半径 4) の画素だけを、輝度差が σ 以内の 5x5 近傍の
// 平均で置き換える。輪郭をまたぐ画素は混ぜないので輪郭は保たれ、低レートの DCT が作る輪郭の横の波だけが消える
constexpr int kNd = 6;
constexpr double kSigma[kNd] = {0.0, 2.0, 4.0, 6.0, 9.0, 13.0};  // 8 ビット換算
void dering(Plane& U, int bit_depth, int dr) {
    if (dr <= 0) return;
    const int w = U.w, h = U.h;
    const double sc = std::ldexp(1.0, bit_depth - 8), sg = kSigma[dr] * sc;
    const Img u(U.v.begin(), U.v.end()), b = blur5(u, w, h);
    Img e(u.size(), 0.0);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const double gx = (b[static_cast<size_t>(y) * w + std::min(x + 1, w - 1)] - b[static_cast<size_t>(y) * w + std::max(x - 1, 0)]) * 0.5;
            const double gy = (b[static_cast<size_t>(std::min(y + 1, h - 1)) * w + x] - b[static_cast<size_t>(std::max(y - 1, 0)) * w + x]) * 0.5;
            e[static_cast<size_t>(y) * w + x] = std::sqrt(gx * gx + gy * gy) > 6.0 * sc ? 1.0 : 0.0;
        }
    const Img zone = box(e, w, h, 4);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const size_t i = static_cast<size_t>(y) * w + x;
            if (zone[i] <= 0) continue;
            double s = 0;
            int n = 0;
            for (int dy = -2; dy <= 2; ++dy)
                for (int dx = -2; dx <= 2; ++dx) {
                    const double q = u[static_cast<size_t>(std::clamp(y + dy, 0, h - 1)) * w + std::clamp(x + dx, 0, w - 1)];
                    if (std::abs(q - u[i]) <= sg) { s += q; ++n; }
                }
            U.v[i] = static_cast<int32_t>(std::lround(s / n));
        }
}

void code_params(SymIO& io, CMModel& m, Params& p) {
    p.dr = static_cast<int>(io.uint(m, 0, 0, static_cast<uint32_t>(p.dr)));
    if (p.dr >= kNd) throw std::runtime_error("corrupt hf layer");
    for (int k = 0; k < kBands; ++k) {
        p.a[k] = static_cast<int>(io.uint(m, 1, static_cast<uint32_t>(k), static_cast<uint32_t>(p.a[k])));
        p.g[k] = static_cast<int>(io.uint(m, 2, static_cast<uint32_t>(k), static_cast<uint32_t>(p.g[k])));
        if (p.a[k] >= kNa || p.g[k] >= kNg) throw std::runtime_error("corrupt hf layer");
    }
}

void apply(const Params& p, const Analysis& A, Plane& U, int bit_depth, uint64_t seed) {
    const int w = U.w, h = U.h;
    const double sc = std::ldexp(1.0, bit_depth - 8), hi = (1 << bit_depth) - 1;
    bool any_g = false;
    for (int k = 0; k < kBands; ++k) any_g |= p.g[k] > 0;
    std::vector<Img> fields;
    if (any_g) {
        SplitMix64 rng(seed * 0x9E3779B97F4A7C15ull + 0x5851F42D4C957F2Dull);
        Img noise(A.u.size());
        for (double& v : noise) v = static_cast<double>(rng.next() >> 11) * (2.0 / 9007199254740992.0) - 1.0;
        fields.push_back(tex_field(noise, w, h, 0));
    }
    // 段ごとの量は段の境界でなめらかにつなぐ (活動度の箱平均で段を決めているので、さらに 9x9 で平均)
    Img al(A.u.size()), gn(A.u.size());
    for (size_t i = 0; i < A.u.size(); ++i) { al[i] = kAlpha[p.a[A.band[i]]]; gn[i] = kGain[p.g[A.band[i]]]; }
    al = box(al, w, h, 4);
    if (any_g) gn = box(gn, w, h, 4);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const size_t i = static_cast<size_t>(y) * w + x;
            // 輪郭の鋭さ: 強調後の値を周囲 3x3 の最小〜最大に収める (はみ出し = 輪郭の横の二重の輪郭を出さない)
            double mn = A.u[i], mx = A.u[i];
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    const double q = A.u[static_cast<size_t>(std::clamp(y + dy, 0, h - 1)) * w + std::clamp(x + dx, 0, w - 1)];
                    mn = std::min(mn, q); mx = std::max(mx, q);
                }
            double v = std::clamp(A.u[i] + al[i] * A.d[i], mn, mx);
            if (any_g && gn[i] > 0) {
                // 質感: 等方の雑音だけを使い (輪郭に平行な縞は輪郭の複製に見える)、強い輪郭の近くでは弱める
                const double damp = 1.0 / (1.0 + (A.gm[i] / (6.0 * sc)) * (A.gm[i] / (6.0 * sc)));
                v += gn[i] * sc * damp * fields[0][i];
            }
            U.v[i] = static_cast<int32_t>(std::clamp(std::lround(v), 0L, static_cast<long>(hi)));
        }
}
}  // namespace

std::vector<uint8_t> hf_encode(const Plane& org, const Plane& U0, int bit_depth, uint64_t seed, bool texture) {
    (void)seed;
    const int w = U0.w, h = U0.h;
    const double sc = std::ldexp(1.0, bit_depth - 8);
    Params p;
    // 0. デリンギングの σ: 元画像との SSE が最小のもの
    Plane U = U0;
    {
        double best = 1e300;
        for (int d = 0; d < kNd; ++d) {
            Plane t = U0;
            dering(t, bit_depth, d);
            double e = 0;
            for (size_t i = 0; i < t.v.size(); ++i) { const double df = t.v[i] - org.v[i]; e += df * df; }
            if (e < best) { best = e; p.dr = d; U = std::move(t); }
        }
    }
    const Analysis A = analyze(U, bit_depth);
    const Img o(org.v.begin(), org.v.end());
    const Img ob = blur5(o, w, h);
    // 1. α: 段ごとに残差 R = org - U と D の最小二乗
    double num[kBands] = {}, den[kBands] = {};
    for (size_t i = 0; i < A.u.size(); ++i) { num[A.band[i]] += (o[i] - A.u[i]) * A.d[i]; den[A.band[i]] += A.d[i] * A.d[i]; }
    double alpha[kBands];
    for (int k = 0; k < kBands; ++k) {
        const double a = den[k] > 0 ? num[k] / den[k] : 0.0;
        int best = 0;
        for (int j = 1; j < kNa; ++j) if (std::abs(kAlpha[j] - a) < std::abs(kAlpha[best] - a)) best = j;
        p.a[k] = best;
        alpha[k] = kAlpha[best];
    }
    // 2. 質感: 元画像の高域エネルギーから、α 適用後の拡大画像の高域エネルギーを引いた不足分 (段ごと)
    double eo[kBands] = {}, eu[kBands] = {};
    int64_t cnt[kBands] = {};
    for (size_t i = 0; i < A.u.size(); ++i) {
        const int k = A.band[i];
        const double oh = o[i] - ob[i], uh = (1 + alpha[k]) * A.d[i];
        eo[k] += oh * oh; eu[k] += uh * uh; ++cnt[k];
    }
    for (int k = 0; k < kBands; ++k) {
        const double def = cnt[k] ? std::sqrt(std::max(0.0, (eo[k] - eu[k]) / cnt[k])) / sc : 0.0;
        const double g = 0.6 * def;  // 不足分の 6 割 (足しすぎると雑音に見える)
        int best = 0;
        for (int j = 1; j < kNg; ++j) if (std::abs(kGain[j] - g) < std::abs(kGain[best] - g)) best = j;
        p.g[k] = texture ? best : 0;
    }
    EntropyWriter ew;
    CMModel m;
    SymIO io;
    io.w = &ew;
    io.enc = true;
    code_params(io, m, p);
    return ew.finish();
}

bool hf_apply(const uint8_t* d, size_t n, Plane& U, int bit_depth, uint64_t seed) {
    Params p;
    try {
        EntropyReader er(d, n);
        CMModel m;
        SymIO io;
        io.r = &er;
        code_params(io, m, p);
    } catch (const std::exception&) {
        return false;
    }
    dering(U, bit_depth, p.dr);
    const Analysis A = analyze(U, bit_depth);
    apply(p, A, U, bit_depth, seed);
    return true;
}

}  // namespace fvc
