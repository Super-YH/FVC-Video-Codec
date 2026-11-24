// FVC 内部: 縮小符号化のパラメトリック高周波層 (復号器の拡大後に適用する後処理)
//  拡大画像 U は元画像の高周波を持たない。高周波を「どこに・どの向きに」足すかは復号器が U 自身から推定し
//  (局所活動度と構造テンソルの向き)、符号器は活動度の段ごとの 2 つの量だけを送る (全体で数バイト):
//   1. 輪郭の鋭さ α: 詳細成分 D = U - blur(U) を α 倍して足す (アンシャープマスク)。段ごとの最小二乗で選ぶ
//   2. 質感の量 g: 元画像と (α 適用後の) 拡大画像の高周波エネルギーの差。復号器は種から決まる雑音を
//      U の局所的な向きに沿って伸ばした高周波場を作り、差のエネルギーだけ足す (値は送らず統計だけ送る)
//  元から滑らかな所 (肌など) は差が小さいので質感は足されない
//  その前に、低レートの DCT が輪郭の横に作る波 (二重の輪郭) をシグマフィルタで消す (σ を 1 つ送る)
#pragma once
#include <cstdint>
#include <vector>

#include "fvc/frame.hpp"

namespace fvc {

// 符号器: 元画像 org と拡大画像 U (同じ大きさ) から高周波層のペイロードを作る
// texture: 質感 (雑音) も足す。false ならデリンギングと輪郭の鋭さだけ (PSNR を上げる方向だけ)
std::vector<uint8_t> hf_encode(const Plane& org, const Plane& U, int bit_depth, uint64_t seed, bool texture);
// 復号器: U にペイロードの高周波層を適用する。失敗 (壊れたデータ) なら false (U は変えない)
bool hf_apply(const uint8_t* d, size_t n, Plane& U, int bit_depth, uint64_t seed);

}  // namespace fvc
