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

// 独自の補間フィルタ (特許安全モード): 輝度は Lanczos-3 窓付き sinc (Duchon 1979) の 6 タップ、
// 色差は Keys の 3 次畳み込み (a = -0.5, 1981) の 4 タップ。いずれも係数和 64 に丸めたもの
constexpr int kLumaOwn[4][8] = {{0, 0, 0, 64, 0, 0, 0, 0},
                                {0, 2, -9, 57, 17, -4, 1, 0},
                                {0, 2, -9, 39, 39, -9, 2, 0},
                                {0, 1, -4, 17, 57, -9, 2, 0}};
constexpr int kChromaOwn[8][4] = {{0, 64, 0, 0},     {-3, 62, 6, -1},  {-4, 56, 14, -2}, {-5, 47, 25, -3},
                                  {-4, 36, 36, -4},  {-3, 25, 47, -5}, {-2, 14, 56, -4}, {-1, 6, 62, -3}};

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
    const int* cx = luma ? (ref.own_filter ? kLumaOwn[fx] : kLuma[fx]) : (ref.own_filter ? kChromaOwn[fx] : kChroma[fx]);
    const int* cy = luma ? (ref.own_filter ? kLumaOwn[fy] : kLuma[fy]) : (ref.own_filter ? kChromaOwn[fy] : kChroma[fy]);
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
    const int SW = W + taps - 1, SH = H + taps - 1;
    // 作業領域はスレッドごとに使い回す (割り当てを避ける)
    thread_local std::vector<int32_t> src, tmp, t2, t3;
    const int bx = x0 + ix - m - half, by = y0 + iy - m - half;
    const bool inside = bx >= 0 && by >= 0 && bx + SW <= p.w && by + SH <= p.h;
    const int32_t* sp;
    int ss;
    if (inside) {
        sp = &p.v[static_cast<size_t>(by) * p.w + bx];
        ss = p.w;
    } else {
        src.resize(static_cast<size_t>(SW) * SH);
        for (int y = 0; y < SH; ++y) {
            const int32_t* r = &p.v[static_cast<size_t>(std::clamp(by + y, 0, p.h - 1)) * p.w];
            int32_t* d = &src[static_cast<size_t>(y) * SW];
            for (int x = 0; x < SW; ++x) d[x] = r[std::clamp(bx + x, 0, p.w - 1)];
        }
        sp = src.data();
        ss = SW;
    }
    // 水平 (×64)
    tmp.resize(static_cast<size_t>(W) * SH);
    for (int y = 0; y < SH; ++y) {
        const int32_t* r = sp + static_cast<size_t>(y) * ss;
        int32_t* t = &tmp[static_cast<size_t>(y) * W];
        if (fx == 0) {
            for (int x = 0; x < W; ++x) t[x] = r[x + half] << 6;
        } else if (luma) {
            const int c0 = cx[0], c1 = cx[1], c2 = cx[2], c3 = cx[3], c4 = cx[4], c5 = cx[5], c6 = cx[6], c7 = cx[7];
            for (int x = 0; x < W; ++x) {
                const int32_t* q = r + x;
                t[x] = c0 * q[0] + c1 * q[1] + c2 * q[2] + c3 * q[3] + c4 * q[4] + c5 * q[5] + c6 * q[6] + c7 * q[7];
            }
        } else {
            const int c0 = cx[0], c1 = cx[1], c2 = cx[2], c3 = cx[3];
            for (int x = 0; x < W; ++x) {
                const int32_t* q = r + x;
                t[x] = c0 * q[0] + c1 * q[1] + c2 * q[2] + c3 * q[3];
            }
        }
    }
    // 垂直 (×64 を保つ)。出力先は平滑なしなら out64 に直接書く
    std::vector<int32_t>& dst = psi ? t2 : out64;
    dst.resize(static_cast<size_t>(W) * H);
    if (fy == 0) {
        std::copy(&tmp[static_cast<size_t>(half) * W], &tmp[static_cast<size_t>(half + H) * W], dst.begin());
    } else {
        for (int y = 0; y < H; ++y) {
            int32_t* o = &dst[static_cast<size_t>(y) * W];
            const int32_t* t = &tmp[static_cast<size_t>(y) * W];
            if (luma) {
                const int c0 = cy[0], c1 = cy[1], c2 = cy[2], c3 = cy[3], c4 = cy[4], c5 = cy[5], c6 = cy[6], c7 = cy[7];
                for (int x = 0; x < W; ++x) {
                    const int32_t* q = t + x;
                    const int32_t s = c0 * q[0] + c1 * q[W] + c2 * q[2 * W] + c3 * q[3 * W] + c4 * q[4 * W] +
                                      c5 * q[5 * W] + c6 * q[6 * W] + c7 * q[7 * W];
                    o[x] = (s + 32) >> 6;
                }
            } else {
                const int c0 = cy[0], c1 = cy[1], c2 = cy[2], c3 = cy[3];
                for (int x = 0; x < W; ++x) {
                    const int32_t* q = t + x;
                    o[x] = (c0 * q[0] + c1 * q[W] + c2 * q[2 * W] + c3 * q[3 * W] + 32) >> 6;
                }
            }
        }
    }
    if (!psi) return;
    out64.resize(static_cast<size_t>(w) * h);
    const int a = psi;  // a/16
    t3.resize(static_cast<size_t>(W) * H);
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
