// FVC 内部: 適応ウィーナーフィルタ (ALF 相当, 仕様 §9.4)
//  7x7 ダイヤモンドの点対称フィルタ。差分形式 out = r + (Σ c_k (r[p+o_k] + r[p-o_k] - 2 r[p]) + 64) >> 7
//  輝度は 4x4 ブロックの活動度 (4 段) × 方向 (3 種) の 12 クラス、色差は 1 クラス。
//  係数はクラスごとに符号器が最小二乗で求めて伝送し、CTU (64x64) ごとに適用フラグを送る。
#pragma once
#include <cstdint>
#include <vector>

#include "block_coder.hpp"
#include "fvc/frame.hpp"

namespace fvc {

void code_alf(SymIO& io, CMModel& m, Plane& R, const Plane* org, bool luma, int bit_depth, double lambda, int32_t lo,
              int32_t hi);

}  // namespace fvc
