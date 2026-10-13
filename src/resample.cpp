#include "resample.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace fvc {

namespace {
constexpr double kPi = 3.14159265358979323846;
double lanczos3(double x) {
    x = std::abs(x);
    if (x < 1e-9) return 1.0;
    if (x >= 3.0) return 0.0;
    return 3.0 * std::sin(kPi * x) * std::sin(kPi * x / 3.0) / (kPi * kPi * x * x);
}
// 整数化した係数 (和 = 2^14)
template <int N>
std::array<int, N> taps_q14(double center, double scale, int first) {
    std::array<double, N> t{};
    double s = 0;
    for (int k = 0; k < N; ++k) { t[k] = lanczos3((first + k - center) * scale); s += t[k]; }
    std::array<int, N> q{};
    int sum = 0;
    for (int k = 0; k < N; ++k) { q[k] = static_cast<int>(std::lround(t[k] / s * 16384.0)); sum += q[k]; }
    q[N / 2] += 16384 - sum;
    return q;
}
// 1 次元: 縮小 (出力 j の中心は入力の 2j + 0.5、12 タップ)
void down1d(const int32_t* in, int n, int stride, int32_t* out, int m, int ostride) {
    static const auto t = taps_q14<12>(0.5, 0.5, -5);
    for (int j = 0; j < m; ++j) {
        int64_t acc = 0;
        for (int k = 0; k < 12; ++k) acc += static_cast<int64_t>(t[k]) * in[static_cast<size_t>(std::clamp(2 * j - 5 + k, 0, n - 1)) * stride];
        out[static_cast<size_t>(j) * ostride] = static_cast<int32_t>((acc + 8192) >> 14);
    }
}
// 1 次元: 拡大 (出力 i の中心は入力の i/2 - 0.25、偶数・奇数で 6 タップずつ)
void up1d(const int32_t* in, int n, int stride, int32_t* out, int m, int ostride) {
    static const auto te = taps_q14<6>(-0.25, 1.0, -3);  // 出力 2j: 入力 j-0.25
    static const auto to = taps_q14<6>(0.25, 1.0, -2);   // 出力 2j+1: 入力 j+0.25
    for (int i = 0; i < m; ++i) {
        const int j = i >> 1;
        const auto& t = (i & 1) ? to : te;
        const int f = (i & 1) ? j - 2 : j - 3;
        int64_t acc = 0;
        for (int k = 0; k < 6; ++k) acc += static_cast<int64_t>(t[k]) * in[static_cast<size_t>(std::clamp(f + k, 0, n - 1)) * stride];
        out[static_cast<size_t>(i) * ostride] = static_cast<int32_t>((acc + 8192) >> 14);
    }
}
}  // namespace

Plane downsample2(const Plane& in) {
    const int w2 = (in.w + 1) / 2, h2 = (in.h + 1) / 2;
    Plane t(w2, in.h), o(w2, h2);
    for (int y = 0; y < in.h; ++y) down1d(&in.v[static_cast<size_t>(y) * in.w], in.w, 1, &t.v[static_cast<size_t>(y) * w2], w2, 1);
    for (int x = 0; x < w2; ++x) down1d(&t.v[x], in.h, w2, &o.v[x], h2, w2);
    return o;
}

Plane upsample2(const Plane& in, int w, int h) {
    Plane t(w, in.h), o(w, h);
    for (int y = 0; y < in.h; ++y) up1d(&in.v[static_cast<size_t>(y) * in.w], in.w, 1, &t.v[static_cast<size_t>(y) * w], w, 1);
    for (int x = 0; x < w; ++x) up1d(&t.v[x], in.h, w, &o.v[x], h, w);
    return o;
}

}  // namespace fvc
