// FVC 内部: ループフィルタ 3 段 (仕様 §9)
//  9.1 デブロック (HEVC 型の境界強度・強/弱フィルタ)
//  9.2 周波数領域ゲインフィルタ (8x8 正弦窓 WOLA, G = max(0, 1 - μσ²/c²))
//  9.3 画素ゲインマップ (32x32 セル, γ ∈ [0, 2], Q6): out = X^ + γ (DF(X^) - X^)
#pragma once
#include <cstdint>
#include <vector>

#include "block_coder.hpp"
#include "fvc/frame.hpp"

namespace fvc {

// 4x4 単位のブロック情報 (BlockCoder が出力)
struct EdgeInfo {
    int w4 = 0, h4 = 0;
    std::vector<int32_t> leaf;   // 葉 ID (同じ ID なら同一ブロック = 境界なし)
    std::vector<uint8_t> flags;  // bit0: イントラ系 (pt != inter), bit1: cbf
};

struct LoopFilterParams {
    bool deblock = true;
    int beta_off = 0, tc_off = 0;
    int mu_q = 0;             // 9.2: μ = mu_q / 4 (0: 無効)
    bool gain_map = false;    // 9.3
    std::vector<int> gamma;   // セルごとの γ (Q6)
};

constexpr int kGainCell = 32;

void deblock_plane(Plane& p, const EdgeInfo& e, const MotionField* mf, bool chroma, int chroma_shift, int qp,
                   int bit_depth, int beta_off, int tc_off, int32_t lo, int32_t hi);
void freq_gain_filter(const Plane& in, Plane& out, double step, int mu_q, int32_t lo, int32_t hi);

// 符号化/復号共通: R (ブロック再構成, パディング済み) にループフィルタを適用する。
// 符号器 (org != nullptr) はパラメータを決めて符号化、復号器は読み取って同じ処理をする。
void code_loop_filter(SymIO& io, CMModel& m, Plane& R, const Plane* org, const EdgeInfo& e, const MotionField* mf,
                      bool chroma, int chroma_shift, int qp, int bit_depth, double step, double lambda, bool allow_freq,
                      bool allow_map, int32_t lo, int32_t hi);

}  // namespace fvc
