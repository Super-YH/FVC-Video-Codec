#include "fvc/color.hpp"

#include <algorithm>
#include <cmath>

namespace fvc {

static int32_t mean_i(const int32_t* v, size_t n) {
    if (n == 0) return 0;
    int64_t s = 0;
    for (size_t i = 0; i < n; ++i) s += v[i];
    // 床除算で整数平均 (エンコーダ/デコーダ同一)
    int64_t q = s / static_cast<int64_t>(n);
    if ((s % static_cast<int64_t>(n)) != 0 && s < 0) --q;
    return static_cast<int32_t>(q);
}

ChromaPredParams fit_chroma_pred(const int32_t* luma, const int32_t* chroma, size_t n) {
    ChromaPredParams p;
    if (n == 0) return p;
    const int32_t my = mean_i(luma, n), mc = mean_i(chroma, n);
    double sxy = 0, sxx = 0;
    for (size_t i = 0; i < n; ++i) {
        const double dy = luma[i] - my, dc = chroma[i] - mc;
        sxy += dy * dc;
        sxx += dy * dy;
    }
    const double alpha = sxx > 0 ? sxy / sxx : 0.0;
    p.alpha_q = static_cast<int32_t>(std::lround(alpha * (1 << ChromaPredParams::kAlphaShift)));
    p.alpha_q = std::clamp(p.alpha_q, -128, 127);
    p.beta = mc;
    return p;
}

void apply_chroma_pred(const ChromaPredParams& p, const int32_t* luma, size_t n, int32_t* pred) {
    const int32_t my = mean_i(luma, n);
    for (size_t i = 0; i < n; ++i) {
        const int64_t d = static_cast<int64_t>(p.alpha_q) * (luma[i] - my);
        pred[i] = static_cast<int32_t>(d >> ChromaPredParams::kAlphaShift) + p.beta;
    }
}

}  // namespace fvc
