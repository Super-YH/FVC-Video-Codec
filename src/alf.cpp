#include "alf.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fvc {

namespace {
constexpr int kTaps = 12;  // |dx|+|dy| <= 3 の半平面
constexpr int kOff[kTaps][2] = {{1, 0}, {2, 0}, {3, 0}, {-2, 1}, {-1, 1}, {0, 1}, {1, 1}, {2, 1},
                                {-1, 2}, {0, 2}, {1, 2}, {0, 3}};
constexpr int kCtuA = 64;

int classify(const Plane& r, int bx, int by, int bd) {
    int64_t gh = 0, gv = 0;
    for (int y = by; y < by + 4; ++y)
        for (int x = bx; x < bx + 4; ++x) {
            const int32_t c = r.at(x, y);
            const int32_t l = r.at(std::max(x - 1, 0), y), rr = r.at(std::min(x + 1, r.w - 1), y);
            const int32_t u = r.at(x, std::max(y - 1, 0)), d = r.at(x, std::min(y + 1, r.h - 1));
            gh += std::abs(2 * c - l - rr);
            gv += std::abs(2 * c - u - d);
        }
    const int64_t act = (gh + gv) >> (bd - 8);
    const int aq = act <= 32 ? 0 : act <= 96 ? 1 : act <= 256 ? 2 : 3;
    const int dir = gh > 2 * gv ? 1 : gv > 2 * gh ? 2 : 0;
    return aq + 4 * dir;
}

// コレスキー分解で (A + εI) x = b を解く
bool solve(std::vector<double> A, std::vector<double> b, int n, std::vector<double>& x) {
    for (int i = 0; i < n; ++i) A[i * n + i] += 1e-3 * (A[i * n + i] + 1.0);
    for (int j = 0; j < n; ++j) {
        double d = A[j * n + j];
        for (int k = 0; k < j; ++k) d -= A[j * n + k] * A[j * n + k];
        if (d <= 0) return false;
        d = std::sqrt(d);
        A[j * n + j] = d;
        for (int i = j + 1; i < n; ++i) {
            double s = A[i * n + j];
            for (int k = 0; k < j; ++k) s -= A[i * n + k] * A[j * n + k];
            A[i * n + j] = s / d;
        }
    }
    std::vector<double> y(n);
    for (int i = 0; i < n; ++i) {
        double s = b[i];
        for (int k = 0; k < i; ++k) s -= A[i * n + k] * y[k];
        y[i] = s / A[i * n + i];
    }
    x.assign(n, 0.0);
    for (int i = n - 1; i >= 0; --i) {
        double s = y[i];
        for (int k = i + 1; k < n; ++k) s -= A[k * n + i] * x[k];
        x[i] = s / A[i * n + i];
    }
    return true;
}
}  // namespace

void code_alf(SymIO& io, CMModel& m, Plane& R, const Plane* org, bool luma, int bit_depth, double lambda, int32_t lo,
              int32_t hi) {
    const int ncls = luma ? 12 : 1;
    const int W = R.w, H = R.h, bw = W / 4, bh = H / 4;
    std::vector<uint8_t> cls(static_cast<size_t>(bw) * bh, 0);
    if (luma)
        for (int by = 0; by < bh; ++by)
            for (int bx = 0; bx < bw; ++bx) cls[by * bw + bx] = static_cast<uint8_t>(classify(R, bx * 4, by * 4, bit_depth));
    auto feat = [&](int x, int y, int k) {
        const int32_t c = R.at(x, y);
        const int x1 = std::clamp(x + kOff[k][0], 0, W - 1), y1 = std::clamp(y + kOff[k][1], 0, H - 1);
        const int x2 = std::clamp(x - kOff[k][0], 0, W - 1), y2 = std::clamp(y - kOff[k][1], 0, H - 1);
        return R.at(x1, y1) + R.at(x2, y2) - 2 * c;
    };
    // 係数の推定 (符号器)
    std::vector<std::vector<int>> coef(ncls, std::vector<int>(kTaps, 0));
    std::vector<int> on(ncls, 0);
    if (io.enc) {
        std::vector<std::vector<double>> A(ncls, std::vector<double>(kTaps * kTaps, 0.0)), B(ncls, std::vector<double>(kTaps, 0.0));
        std::vector<double> E0(ncls, 0.0);
        std::vector<int64_t> cnt(ncls, 0);
        double f[kTaps];
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                const int c = cls[(y / 4) * bw + x / 4];
                const double t = org->at(x, y) - R.at(x, y);
                for (int k = 0; k < kTaps; ++k) f[k] = feat(x, y, k);
                for (int i = 0; i < kTaps; ++i) {
                    B[c][i] += f[i] * t;
                    for (int j = 0; j <= i; ++j) A[c][i * kTaps + j] += f[i] * f[j];
                }
                E0[c] += t * t;
                ++cnt[c];
            }
        for (int c = 0; c < ncls; ++c) {
            for (int i = 0; i < kTaps; ++i)
                for (int j = i + 1; j < kTaps; ++j) A[c][i * kTaps + j] = A[c][j * kTaps + i];
            std::vector<double> x;
            if (cnt[c] < 256 || !solve(A[c], B[c], kTaps, x)) continue;
            double gain = 0;  // SSE 減少量 ≈ 2 b·x - xᵀAx (量子化後の係数で評価)
            std::vector<double> xq(kTaps);
            for (int k = 0; k < kTaps; ++k) {
                coef[c][k] = std::clamp(static_cast<int>(std::lround(x[k] * 128.0)), -64, 63);
                xq[k] = coef[c][k] / 128.0;
            }
            for (int i = 0; i < kTaps; ++i) {
                gain += 2 * B[c][i] * xq[i];
                for (int j = 0; j < kTaps; ++j) gain -= xq[i] * A[c][i * kTaps + j] * xq[j];
            }
            on[c] = gain > lambda * 6.0 * kTaps;
        }
    }
    int any = 0;
    for (int c = 0; c < ncls; ++c) any |= on[c];
    any = io.bit(m, 30, luma, 0, any);
    if (!any) return;
    for (int c = 0; c < ncls; ++c) {
        on[c] = io.bit(m, 31, luma, 0, on[c]);
        if (!on[c]) { std::fill(coef[c].begin(), coef[c].end(), 0); continue; }
        for (int k = 0; k < kTaps; ++k) {
            coef[c][k] = io.sint(m, 32 + k, luma, coef[c][k]);
            if (coef[c][k] < -64 || coef[c][k] > 63) throw std::runtime_error("corrupt stream: alf coef");
        }
    }
    // フィルタ適用 (全面) → CTU ごとの採否
    Plane F = R;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const int c = cls[(y / 4) * bw + x / 4];
            if (!on[c]) continue;
            int64_t sum = 0;
            for (int k = 0; k < kTaps; ++k) sum += static_cast<int64_t>(coef[c][k]) * feat(x, y, k);
            F.at(x, y) = std::clamp(R.at(x, y) + static_cast<int32_t>((sum + 64) >> 7), lo, hi);
        }
    const int cw = (W + kCtuA - 1) / kCtuA, ch = (H + kCtuA - 1) / kCtuA;
    int prev = 1;
    for (int cy = 0; cy < ch; ++cy)
        for (int cx = 0; cx < cw; ++cx) {
            int use = 0;
            if (io.enc) {
                double e0 = 0, e1 = 0;
                for (int y = cy * kCtuA; y < std::min(H, (cy + 1) * kCtuA); ++y)
                    for (int x = cx * kCtuA; x < std::min(W, (cx + 1) * kCtuA); ++x) {
                        const double a = org->at(x, y) - R.at(x, y), b = org->at(x, y) - F.at(x, y);
                        e0 += a * a; e1 += b * b;
                    }
                use = e1 + lambda * 1.0 < e0;
            }
            use = io.bit(m, 50, static_cast<uint32_t>(prev), luma, use);
            prev = use;
            if (!use) continue;
            for (int y = cy * kCtuA; y < std::min(H, (cy + 1) * kCtuA); ++y)
                for (int x = cx * kCtuA; x < std::min(W, (cx + 1) * kCtuA); ++x) R.at(x, y) = F.at(x, y);
        }
}

}  // namespace fvc
