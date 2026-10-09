// FVC: ブロック変換 (DCT-II / DST-VII / 恒等, 分離型 4..1024) と TNS。
#pragma once
#include <cstdint>
#include <vector>

namespace fvc {

enum class TxType : uint8_t { DCT2 = 0, DST7 = 1, IDTX = 2, DCT8 = 3 };

// 正規直交基底行列 (N x N, row k = 基底 k)。キャッシュ付き。
const std::vector<double>& tx_matrix(TxType t, int N);

// 分離型 2D 変換。blk は row-major (w x h)。
void forward_2d(TxType tx_h, TxType tx_v, const double* in, int w, int h, double* out);
void inverse_2d(TxType tx_h, TxType tx_v, const double* in, int w, int h, double* out);

// TNS: 係数列 c[0..n) に対する周波数方向 LPC。
//  解析: e[k] = c[k] + Σ_{i=1..p} a_i c[k-i]  (FIR, 符号器)
//  合成: c[k] = e[k] - Σ a_i c[k-i]          (IIR, 復号器)
// a は PARCOR 係数 r_i (|r|<1) を arcsin 量子化して伝送し、step-up で再構成。
struct TnsFilter {
    int order = 0;
    std::vector<int32_t> parcor_q;  // 量子化 PARCOR (4bit 推奨)
    int qbits = 4;
};
TnsFilter tns_design(const double* c, int n, int max_order, int qbits);
std::vector<double> tns_lpc_from_parcor(const TnsFilter& f);
void tns_analysis(const TnsFilter& f, const double* c, int n, double* e);
void tns_synthesis(const TnsFilter& f, const double* e, int n, double* c);

}  // namespace fvc
