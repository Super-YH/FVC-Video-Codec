// FVC 内部: 線に基づく動き (特許安全モードのマージ/時間方向候補の代替)
//  1. 線の抽出: 参照ピクチャ (復号済み輝度) の 16x16 セルごとに最も強いエッジ点を選び、
//     その点と勾配 (法線) で直線をパラメトリックに表す。復号器も同じ参照から同じ線を得るので形は伝送しない。
//  2. 線の動き: 線ごとに「大域動きから変わったか」のフラグと、変わった線だけ動きの差分を伝送する。
//     変わった線は「どちら側が動くか」(法線の + 側 / − 側 / 両側) も伝送する (動きの境界をエッジに一致させる)。
//  3. 動き場: 4x4 ごとに近傍の線の動きを距離の逆数で重み付け平均した密な動き場を作る (線がなければ大域動き)。
//     変わった線は現フレームでの位置 (参照位置 − 動き) に置き、動く側でない画素には効かせない。
//  ブロックは「線の動き場で予測する」1 ビットのモードを選べる (4x4 ごとに異なる動き)。
#pragma once
#include <cstdint>
#include <vector>

#include "block_coder.hpp"
#include "fvc/frame.hpp"
#include "inter.hpp"

namespace fvc {

struct LineSeg {
    int x = 0, y = 0;       // 線上の点 (エッジ点, 輝度画素)
    int gx = 0, gy = 0;     // 勾配 (法線方向, Sobel)
    int mvx = 0, mvy = 0;   // 動き (1/4 画素)
    bool changed = false;   // 大域動きから変わったか
    int side = 2;           // 動く側: 0 = 法線の − 側, 1 = + 側, 2 = 両側
    int bx = 0, by = 0;     // 基準の動き (1/4 画素): 参照ピクチャの動きの時間伸縮、なければ大域動き。
                            // 変わらない線はこの動きで動き、変わった線はこれからの差分を送る
};

constexpr int kLineCell = 16;

// 参照輝度から線を抽出する (符号器・復号器で同一の整数演算)。thr8: 8 ビット換算の |gx|+|gy| 下限
std::vector<LineSeg> extract_lines(const Plane& ref, int vw, int vh, int bit_depth, int thr8 = 96);

// 帯域画像の線予測 (静止画の帯域方式): LL だけから合成した画像 y0 (W x H, 値 - off) の線を、線ごとに
// 「位置の補正 (1/4 画素) と鋭さ」を伝送した理想エッジで置き換えた画像 y1 を作る。y1 を帯域分解した高域を
// 高域帯域の予測に使う。線は y0 から両側で抽出するので形は送らない
// ブロックのエッジモード: 予測ブロック pred (s x s) の中の線 (8x8 区画ごとの最強エッジ) を、鋭さ sh (0..3) と
// 位置補正 dq/4 画素の理想エッジで置き換える。線は予測そのものから抽出するので形は送らない
void edge_sharpen_block(int32_t* pred, int s, int sh, int dq, int32_t lo, int32_t hi, int bit_depth);

void code_band_lines(SymIO& io, CMModel& m, const std::vector<double>& y0, int W, int H, const Plane* org, double off,
                     double lambda, int bit_depth, std::vector<double>& y1);
// 符号器: 線ごとの動きを推定し、変わったかを RD で決める
void estimate_line_motion(std::vector<LineSeg>& lines, const Plane& cur, const Plane& ref, const GlobalModel& gm,
                          double lambda, int range);
// 線ごとの基準の動き (bx, by) を大域動きで初期化する
void set_line_base(std::vector<LineSeg>& lines, const GlobalModel& gm);
// 線の動きの構文 (符号器/復号器共通)。変わらない線の動きは基準の動き (bx, by) になる
void code_line_motion(SymIO& io, CMModel& m, std::vector<LineSeg>& lines, const GlobalModel& gm);
// 4x4 単位の動き場 (mvx, mvy の組, 1/4 画素) を作る
void build_line_field(const std::vector<LineSeg>& lines, const GlobalModel& gm, int w4, int h4, std::vector<int16_t>& field);

}  // namespace fvc
