#include "inter.hpp"

#include <algorithm>
#include <cstdlib>

namespace fvc {

namespace {
// HEVC 輝度 8 タップ (1/4 画素)
constexpr int kLuma[4][8] = {{0, 0, 0, 64, 0, 0, 0, 0},
                             {-1, 4, -10, 58, 17, -5, 1, 0},
                             {-1, 4, -11, 40, 40, -11, 4, -1},
                             {0, 1, -5, 17, 58, -10, 4, -1}};
// HEVC 色差 4 タップ (1/8 画素)
constexpr int kChroma[8][4] = {{0, 64, 0, 0},     {-2, 58, 10, -2}, {-4, 54, 16, -2}, {-6, 46, 28, -4},
                               {-4, 36, 36, -4},  {-4, 28, 46, -6}, {-2, 16, 54, -4}, {-2, 10, 58, -2}};

inline int floor_div(int a, int b) { return a >= 0 ? a / b : -((-a + b - 1) / b); }
}  // namespace

void motion_compensate(const RefPlane& ref, int x0, int y0, int w, int h, int mvx, int mvy, int frac_bits, int psi,
                       std::vector<int32_t>& out64) {
    const Plane& p = *ref.p;
    const int one = 1 << frac_bits;
    const int ix = floor_div(mvx, one), iy = floor_div(mvy, one);
    const int fx = mvx - ix * one, fy = mvy - iy * one;
    const bool luma = frac_bits == 2;
    const int taps = luma ? 8 : 4, half = taps / 2 - 1;
    const int* cx = luma ? kLuma[fx] : kChroma[fx];
    const int* cy = luma ? kLuma[fy] : kChroma[fy];
    // 高速経路: 整数ベクトル・平滑なし (参照内部ならクランプも不要)
    if (fx == 0 && fy == 0 && psi == 0) {
        out64.resize(static_cast<size_t>(w) * h);
        const int sx = x0 + ix, sy = y0 + iy;
        const bool inside = sx >= 0 && sy >= 0 && sx + w <= p.w && sy + h <= p.h;
        for (int y = 0; y < h; ++y) {
            int32_t* o = &out64[static_cast<size_t>(y) * w];
            if (inside) {
                const int32_t* r = &p.v[static_cast<size_t>(sy + y) * p.w + sx];
                for (int x = 0; x < w; ++x) o[x] = r[x] << 6;
            } else {
                const int ry = std::clamp(sy + y, 0, p.h - 1);
                for (int x = 0; x < w; ++x) o[x] = p.at(std::clamp(sx + x, 0, p.w - 1), ry) << 6;
            }
        }
        return;
    }
    // 平滑用に 1 画素マージン
    const int m = psi ? 1 : 0;
    const int W = w + 2 * m, H = h + 2 * m;
    auto px = [&](int x, int y) { return p.at(std::clamp(x, 0, p.w - 1), std::clamp(y, 0, p.h - 1)); };
    std::vector<int32_t> tmp(static_cast<size_t>(W) * (H + taps - 1));
    const int bx = x0 + ix - m, by = y0 + iy - m;
    const bool inside = bx - half >= 0 && by - half >= 0 && bx + W + taps - half <= p.w && by + H + taps - half <= p.h;
    for (int y = 0; y < H + taps - 1; ++y) {
        int32_t* t = &tmp[static_cast<size_t>(y) * W];
        if (inside) {
            const int32_t* r = &p.v[static_cast<size_t>(by + y - half) * p.w + bx - half];
            if (fx == 0) { for (int x = 0; x < W; ++x) t[x] = r[x + half] << 6; continue; }
            for (int x = 0; x < W; ++x) {
                int32_t s = 0;
                for (int k = 0; k < taps; ++k) s += cx[k] * r[x + k];
                t[x] = s;
            }
        } else {
            for (int x = 0; x < W; ++x) {
                int32_t s = 0;
                for (int k = 0; k < taps; ++k) s += cx[k] * px(bx + x + k - half, by + y - half);
                t[x] = s;  // ×64
            }
        }
    }
    std::vector<int32_t> t2(static_cast<size_t>(W) * H);
    if (fy == 0) {
        for (int y = 0; y < H; ++y)
            std::copy(&tmp[static_cast<size_t>(y + half) * W], &tmp[static_cast<size_t>(y + half) * W] + W, &t2[static_cast<size_t>(y) * W]);
    } else
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            int64_t s = 0;
            for (int k = 0; k < taps; ++k) s += static_cast<int64_t>(cy[k]) * tmp[static_cast<size_t>(y + k) * W + x];
            t2[static_cast<size_t>(y) * W + x] = static_cast<int32_t>((s + 32) >> 6);  // ×64
        }
    out64.resize(static_cast<size_t>(w) * h);
    if (!psi) { out64 = t2; return; }
    const int a = psi;  // a/16
    std::vector<int32_t> t3(static_cast<size_t>(W) * H);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const int xl = std::max(x - 1, 0), xr = std::min(x + 1, W - 1);
            t3[static_cast<size_t>(y) * W + x] =
                (a * t2[y * W + xl] + (16 - 2 * a) * t2[y * W + x] + a * t2[y * W + xr] + 8) >> 4;
        }
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const int yy = y + 1, xx = x + 1;
            out64[static_cast<size_t>(y) * w + x] =
                (a * t3[(yy - 1) * W + xx] + (16 - 2 * a) * t3[yy * W + xx] + a * t3[(yy + 1) * W + xx] + 8) >> 4;
        }
}

void inter_predict(const MotionInfo& mi, const RefPlane* l0, const RefPlane* l1, int x0, int y0, int w, int h,
                   int chroma_shift, int32_t lo, int32_t hi, int32_t* pred) {
    const int fb = 2 + chroma_shift;
    std::vector<int32_t> a, b;
    auto weighted = [&](const RefPlane& r, int l, std::vector<int32_t>& o) {
        motion_compensate(r, x0, y0, w, h, mi.mvx[l], mi.mvy[l], fb, mi.psi, o);
        for (auto& v : o) v = static_cast<int32_t>((static_cast<int64_t>(v) * r.gain_q + 32) >> 6) + r.off * 64;
    };
    const bool u0 = mi.dir & 1, u1 = (mi.dir >> 1) & 1;
    if (u0) weighted(l0[mi.ref[0]], 0, a);
    if (u1) weighted(l1[mi.ref[1]], 1, b);
    const size_t n = static_cast<size_t>(w) * h;
    for (size_t i = 0; i < n; ++i) {
        int64_t v;
        if (u0 && u1) v = (static_cast<int64_t>(a[i]) + b[i] + 64) >> 7;
        else v = ((u0 ? a[i] : b[i]) + 32) >> 6;
        pred[i] = static_cast<int32_t>(std::clamp<int64_t>(v, lo, hi));
    }
}

}  // namespace fvc
