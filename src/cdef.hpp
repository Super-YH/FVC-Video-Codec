// FVC 内部: 方向性デリンギングフィルタ (AV1 CDEF 相当, 仕様 §9.5)
//  8x8 ブロックごとに方向を推定し、方向に沿った主フィルタと ±45° の副フィルタを制約関数つきで適用する。
//  プレーンごとに最大 4 組の強さ (主 0..15, 副 {0,1,2,4}) を送り、64x64 ブロックごとに組を選ぶ。
#pragma once
#include "block_coder.hpp"
#include "fvc/frame.hpp"

namespace fvc {

void code_cdef(SymIO& io, CMModel& m, Plane& R, const Plane* org, bool luma, int qp, int bit_depth, double lambda,
               int32_t lo, int32_t hi);

}  // namespace fvc
