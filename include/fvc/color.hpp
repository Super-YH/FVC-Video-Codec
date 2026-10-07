// FVC: 色変換 (YCoCg-R, 可逆) と色予測 (CfL 型線形予測)
#pragma once
#include <cstdint>
#include <vector>

namespace fvc {

// 可逆 YCoCg-R。Co/Cg は入力ビット深度 +1 ビットを要する。
inline void rgb_to_ycocg_r(int32_t r, int32_t g, int32_t b, int32_t& y, int32_t& co, int32_t& cg) {
    co = r - b;
    int32_t t = b + (co >> 1);
    cg = g - t;
    y = t + (cg >> 1);
}

inline void ycocg_r_to_rgb(int32_t y, int32_t co, int32_t cg, int32_t& r, int32_t& g, int32_t& b) {
    int32_t t = y - (cg >> 1);
    g = cg + t;
    b = t - (co >> 1);
    r = b + co;
}

// 色予測パラメータ: C^ = ((alpha_q * (Y - meanY)) >> kAlphaShift) + beta
struct ChromaPredParams {
    static constexpr int kAlphaShift = 6;  // alpha は 1/64 精度
    int32_t alpha_q = 0;                   // [-128, 127] -> alpha ∈ [-2, 2)
    int32_t beta = 0;                      // 予測オフセット (= meanC)
};

// 最小二乗で alpha, beta を求め量子化する。
ChromaPredParams fit_chroma_pred(const int32_t* luma, const int32_t* chroma, size_t n);

// pred[i] = 予測値。residual = chroma - pred をとるかどうかはフラグで選択 (仕様 §2.3)。
void apply_chroma_pred(const ChromaPredParams& p, const int32_t* luma, size_t n, int32_t* pred);

}  // namespace fvc
