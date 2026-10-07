// FVC 内部: インター予測 (仕様 §7) — 動き補償、低次元 FIR、動きベクトル場
#pragma once
#include <cstdint>
#include <vector>

#include "fvc/frame.hpp"

namespace fvc {

// 参照ピクチャ (1 プレーン分) とグローバルゲイン/オフセット (§7.2)
struct RefPlane {
    const Plane* p = nullptr;
    int gain_q = 64;    // Q6 (64 = 1.0)
    int32_t off = 0;
};

// 4x4 (輝度) 単位の動き情報
struct MotionInfo {
    uint8_t dir = 0;      // 0: イントラ, 1: L0, 2: L1, 3: 双方向
    int8_t ref[2] = {0, 0};
    int16_t mvx[2] = {0, 0}, mvy[2] = {0, 0};  // 1/4 画素 (輝度)
    uint8_t psi = 0;      // FIR 平滑化強度 0..3
    bool operator==(const MotionInfo& o) const {
        if (dir != o.dir || psi != o.psi) return false;
        for (int l = 0; l < 2; ++l)
            if ((dir >> l) & 1)
                if (ref[l] != o.ref[l] || mvx[l] != o.mvx[l] || mvy[l] != o.mvy[l]) return false;
        return true;
    }
};

struct MotionField {
    int w4 = 0, h4 = 0;
    std::vector<MotionInfo> mi;
    void init(int W, int H) { w4 = W / 4; h4 = H / 4; mi.assign(static_cast<size_t>(w4) * h4, MotionInfo{}); }
    const MotionInfo& at(int x, int y) const { return mi[static_cast<size_t>(y / 4) * w4 + x / 4]; }
    void fill(int x0, int y0, int w, int h, const MotionInfo& m) {
        for (int y = y0 / 4; y < (y0 + h) / 4; ++y)
            for (int x = x0 / 4; x < (x0 + w) / 4; ++x) mi[static_cast<size_t>(y) * w4 + x] = m;
    }
};

// 動き補償 1 方向。mv は frac_bits の固定小数 (輝度: 2 = 1/4 画素, 4:2:0 色差: 3 = 1/8 画素)。
// 輝度は 8 タップ DCT-IF、色差は 4 タップ。psi>0 で 3x3 分離平滑 [a,1-2a,a], a = psi/16 (§7.3)。
// 出力は 1/64 精度 (×64) の中間値。
void motion_compensate(const RefPlane& ref, int x0, int y0, int w, int h, int mvx, int mvy, int frac_bits, int psi,
                       std::vector<int32_t>& out64);

// 1 方向/双方向を合成して最終予測を作る (ゲイン/オフセット適用、丸め、クリップ)
void inter_predict(const MotionInfo& mi, const RefPlane* l0, const RefPlane* l1, int x0, int y0, int w, int h,
                   int chroma_shift, int32_t lo, int32_t hi, int32_t* pred);

}  // namespace fvc
