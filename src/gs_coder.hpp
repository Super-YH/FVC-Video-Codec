// FVC 内部: 係数のバンド/パーティション分割・利得形状分離・パラメトリック補完 (仕様 §5.3–§5.7, §7.6)
//
//  変換係数 (量子化ステップ単位) を放射状バンド (オクターブ) に分け、各バンドを走査順に長さ ≤16 の
//  パーティションへ分割する。DC は単独のスカラ。各パーティションは利得 g (対数量子化 2^{qg/4}) と
//  モードを持つ:
//    0 = ゼロ, 1 = 形状を量子化 (非線形スカラ+前変換 / E8 格子 VQ / ランダム投影), 2 = ノイズ置換 (PNS),
//    3 = 符号のみ (振幅は包絡モデル ρ^-1), 4 = 係数コピー (直前の非ゼロパーティション、並び替え Π 付き),
//    5 = 包絡補間 (利得は直前 2 つの対数線形外挿、中身はノイズ。利得も送らない)
//  形状はすべて復号時に ĝ へエネルギー正規化する (ĉ = ĝ ŝ/‖ŝ‖)。
#pragma once
#include <cstdint>
#include <vector>

#include "fvc/entropy.hpp"

namespace fvc {

struct SymIO;

constexpr int kGsPart = 16;  // パーティション長 L_p

struct GsUnit {
    uint8_t mode = 0;  // 0..5 (上記)
    uint8_t type = 0;  // mode 1: 0 = 非線形スカラ, 1 = E8, 2 = ランダム投影
    uint8_t pn = 0;    // type 0: 前変換 0..5 (Id, Power.75[NonlinearSQ], Power.6, Asinh, SignedLog, PWL)
    uint8_t perm = 0;  // mode 4: 0 恒等, 1 反転, 2 半周巡回, 3 偶奇交換
    int qg = 0;        // 利得 (mode 5 では復号側で導出した値)
    std::vector<int32_t> sym;  // mode 1: 量子化シンボル, mode 3: 符号 (0/1)
};

struct GsBlock {
    int32_t dc = 0;
    std::vector<GsUnit> u;
};

// ブロック (log2 サイズ l) のパーティション配置: 走査順インデックスの列とその放射周波数
struct GsLayout {
    std::vector<std::vector<int>> idx;   // [unit] -> 走査順インデックス
    std::vector<std::vector<double>> rho;
    std::vector<int> band;               // [unit] -> バンド番号 (1..)
};
const GsLayout& gs_layout(int l);

// 符号器: x (走査順, ステップ単位, 長さ 4^l) からモードと量子化を RD で決める。
//  lam = λ/Δ² (ステップ単位の歪みに対するビット単価)。m は現在のモデル (レート推定に使用、更新しない)。
void gs_encode(const double* x, int l, double lam, CMModel& m, uint32_t pc, uint64_t seed, GsBlock& out);
// 構文 (符号化/復号/レート推定共通)
void gs_syntax(SymIO& io, CMModel& m, GsBlock& g, int l, uint32_t pc);
// 再構成: xh (走査順, ステップ単位) を書く
void gs_reconstruct(const GsBlock& g, int l, uint64_t seed, double* xh);
bool gs_nonzero(const GsBlock& g);

}  // namespace fvc
