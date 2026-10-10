#include "shapes.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fvc {

namespace {
constexpr double kPi = 3.14159265358979323846;
}

void render_shapes(const std::vector<Shape>& shapes, Plane& S) {
    for (const Shape& s : shapes) {
        const double r1 = std::pow(2.0, s.l1 / 8.0), r2 = std::pow(2.0, s.l2 / 8.0);
        const double th = s.theta * 2.0 * kPi / 64.0, c = std::cos(th), sn = std::sin(th);
        const double cx = s.cx / 4.0, cy = s.cy / 4.0;
        const double sig = std::pow(2.0, s.soft / 4.0 - 2.0);
        const double ext = std::max(r1, r2) * (s.type == 0 ? 3.5 : 1.0) + 8.0 * sig + 1.0;
        const int x0 = std::max(0, static_cast<int>(cx - ext)), x1 = std::min(S.w - 1, static_cast<int>(cx + ext));
        const int y0 = std::max(0, static_cast<int>(cy - ext)), y1 = std::min(S.h - 1, static_cast<int>(cy + ext));
        for (int y = y0; y <= y1; ++y)
            for (int x = x0; x <= x1; ++x) {
                const double dx = x - cx, dy = y - cy;
                const double u = (c * dx + sn * dy) / r1, v = (-sn * dx + c * dy) / r2;  // 正準座標 u = A^-1 (x - t)
                double m;
                switch (s.type) {
                case 0: m = std::exp(-0.5 * (u * u + v * v)); break;
                case 1: { const double d = (std::sqrt(u * u + v * v) - 1.0) * std::min(r1, r2); m = 1.0 / (1.0 + std::exp(d / sig)); break; }
                case 2: { const double d = (std::max(std::abs(u), std::abs(v)) - 1.0) * std::min(r1, r2); m = 1.0 / (1.0 + std::exp(d / sig)); break; }
                default: {  // 正三角形 (正準座標)
                    const double d1 = v - 0.5, d2 = -0.866 * u - 0.5 * v - 0.5, d3 = 0.866 * u - 0.5 * v - 0.5;
                    const double d = std::max(d1, std::max(d2, d3)) * std::min(r1, r2);
                    m = 1.0 / (1.0 + std::exp(d / sig));
                }
                }
                const double val = s.amp * (1.0 + (s.gx * u + s.gy * v) / 16.0);
                S.at(x, y) += static_cast<int32_t>(std::lround(m * val));
            }
    }
}

void code_shapes(SymIO& io, CMModel& m, std::vector<Shape>& shapes, const std::vector<Shape>* tref) {
    const uint32_t n = io.uint(m, 0, 0, static_cast<uint32_t>(shapes.size()));
    if (n > 4096) throw std::runtime_error("corrupt stream: shape count");
    if (!io.w) shapes.assign(n, Shape{});
    Shape prev;
    prev.cx = prev.cy = 0;
    for (uint32_t i = 0; i < n; ++i) {
        Shape& s = shapes[i];
        // 時間方向: 前フレームの同番号図形からの差分 (図形の動き §7.4)
        if (tref && i < tref->size()) {
            const Shape& t = (*tref)[i];
            const int temporal = io.bit(m, 12, 0, 0, s.type == t.type && s.l1 == t.l1 && s.l2 == t.l2 && s.theta == t.theta && s.soft == t.soft);
            if (temporal) {
                s.type = t.type; s.l1 = t.l1; s.l2 = t.l2; s.theta = t.theta; s.soft = t.soft;
                s.cx = t.cx + io.sint(m, 13, 0, s.cx - t.cx);
                s.cy = t.cy + io.sint(m, 14, 0, s.cy - t.cy);
                s.amp = t.amp + io.sint(m, 15, 0, s.amp - t.amp);
                s.gx = t.gx + io.sint(m, 16, 0, s.gx - t.gx);
                s.gy = t.gy + io.sint(m, 17, 0, s.gy - t.gy);
                prev = s;
                continue;
            }
        }
        // ref = 直前図形 (§4.3)。copy_flags: bit0 型+スケール+角度+ソフトネスが同一
        const int same = io.bit(m, 1, 0, 0, s.type == prev.type && s.l1 == prev.l1 && s.l2 == prev.l2 && s.theta == prev.theta && s.soft == prev.soft);
        if (same) {
            s.type = prev.type; s.l1 = prev.l1; s.l2 = prev.l2; s.theta = prev.theta; s.soft = prev.soft;
        } else {
            s.type = static_cast<int>(io.uint(m, 2, 0, static_cast<uint32_t>(s.type)));
            s.l1 = prev.l1 + io.sint(m, 3, 0, s.l1 - prev.l1);
            s.l2 = s.l1 + io.sint(m, 4, 0, s.l2 - s.l1);
            s.theta = io.uint(m, 5, 0, static_cast<uint32_t>(s.theta)) & 63;
            s.soft = static_cast<int>(io.uint(m, 6, 0, static_cast<uint32_t>(s.soft)));
            if (s.type > 3 || s.l1 < -16 || s.l1 > 96 || s.l2 < -16 || s.l2 > 96 || s.soft > 32)
                throw std::runtime_error("corrupt stream: shape params");
        }
        s.cx = prev.cx + io.sint(m, 7, 0, s.cx - prev.cx);
        s.cy = prev.cy + io.sint(m, 8, 0, s.cy - prev.cy);
        s.amp = io.sint(m, 9, 0, s.amp);
        s.gx = io.sint(m, 10, 0, s.gx);
        s.gy = io.sint(m, 11, 0, s.gy);
        prev = s;
    }
}

std::vector<Shape> fit_shapes(const Plane& org, int max_shapes) {
    // 1/8 縮小・平均除去した残差に、等方ガウスを貪欲追跡 (位置=|r| 最大、スケール総当たり、振幅=最小二乗)
    const int f = 8, w = std::max(1, org.w / f), h = std::max(1, org.h / f);
    std::vector<double> r(static_cast<size_t>(w) * h, 0.0);
    double mean = 0;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            double s = 0;
            for (int j = 0; j < f; ++j) for (int i = 0; i < f; ++i) s += org.at(std::min(x * f + i, org.w - 1), std::min(y * f + j, org.h - 1));
            r[y * w + x] = s / (f * f);
            mean += r[y * w + x];
        }
    mean /= r.size();
    for (double& v : r) v -= mean;
    std::vector<Shape> out;
    for (int it = 0; it < max_shapes; ++it) {
        double best_gain = 0;
        Shape best;
        std::vector<double> best_g;
        // 粗いグリッド上の候補中心
        for (int cy = 0; cy < h; cy += 2)
            for (int cx = 0; cx < w; cx += 2)
                for (int l = 24; l <= 56; l += 8) {  // 半径 2^(l/8) 画素 = 8..128
                    const double rad = std::pow(2.0, l / 8.0) / f;
                    const int ext = static_cast<int>(rad * 3) + 1;
                    double num = 0, den = 0;
                    for (int y = std::max(0, cy - ext); y <= std::min(h - 1, cy + ext); ++y)
                        for (int x = std::max(0, cx - ext); x <= std::min(w - 1, cx + ext); ++x) {
                            const double d2 = ((x - cx) * (x - cx) + (y - cy) * (y - cy)) / (rad * rad);
                            const double g = std::exp(-0.5 * d2);
                            num += g * r[y * w + x];
                            den += g * g;
                        }
                    if (den <= 0) continue;
                    const double gain = num * num / den;  // エネルギー減少量
                    if (gain > best_gain) {
                        best_gain = gain;
                        best = Shape{};
                        best.type = 0;
                        best.cx = (cx * f + f / 2) * 4;
                        best.cy = (cy * f + f / 2) * 4;
                        best.l1 = best.l2 = l;
                        best.amp = static_cast<int>(std::lround(num / den));
                    }
                }
        if (best.amp == 0 || best_gain < 50.0 * f * f / (f * f)) break;
        // 当てはめた図形を残差から除去
        const double rad = std::pow(2.0, best.l1 / 8.0) / f;
        const double ccx = (best.cx / 4.0 - f / 2) / f, ccy = (best.cy / 4.0 - f / 2) / f;
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                const double d2 = ((x - ccx) * (x - ccx) + (y - ccy) * (y - ccy)) / (rad * rad);
                r[y * w + x] -= best.amp * std::exp(-0.5 * d2);
            }
        out.push_back(best);
    }
    return out;
}

double shapes_bits_estimate(const std::vector<Shape>& shapes) {
    double b = 4;
    for (const Shape& s : shapes) b += 30.0 + 2.0 * std::log2(1.0 + std::abs(s.amp));
    return b;
}

}  // namespace fvc
