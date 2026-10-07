// FVC: 量子化 — 格子ベクトル量子化 (Z^n, D_n, E8)、ランダム投影、非線形スカラ量子化。
#pragma once
#include <cstdint>
#include <vector>

namespace fvc {

enum class Lattice : uint8_t { Zn = 0, Dn = 1, E8 = 2 };

// 最近傍格子点 (Conway & Sloane)。E8 は n==8 必須。
void nearest_zn(const double* x, int n, double* y);
void nearest_dn(const double* x, int n, double* y);
void nearest_e8(const double* x, double* y);

// 格子点 <-> 整数インデックス (エントロピー符号化用シンボル)。
//  Zn: k_i = y_i
//  Dn: k_i = y_i (i<n-1)、最終座標は (y_{n-1} - parity)/2 を送る (和偶数制約)
//  E8: c = 2*y_0 mod 2 (コセットビット)、z_i = (2y_i - c)/2 を i<7 に、z_7 は和制約で 1 ビット削減
struct LatticeIndex {
    std::vector<int32_t> z;  // 送出整数
    int32_t coset = 0;       // E8 のみ
};
LatticeIndex lattice_to_index(Lattice L, const double* y, int n);
void index_to_lattice(Lattice L, const LatticeIndex& idx, int n, double* y);

// 格子 VQ: x をステップ s でスケールして量子化。 y = s * Q_Λ(x / s)
void lattice_quantize(Lattice L, const double* x, int n, double step, double* xq, LatticeIndex* idx);

// 非線形スカラ量子化 (べき乗コンパンディング + デッドゾーン)
//   v = sign(x)|x|^γ / Δ ; q = sign(v) floor(|v| + 1 - θ) ; θ ∈ [0.5,1) はデッドゾーン丸め
//   復号: x^ = sign(q) ((|q| + δ) Δ)^(1/γ)  (q≠0)、δ ∈ [-0.5,0] は再構成オフセット
struct NonlinearSQ {
    double gamma = 0.75, step = 1.0, theta = 0.5, delta = 0.0;
    int32_t quantize(double x) const;
    double dequantize(int32_t q) const;
};

// ランダム投影: シード s から ±1/sqrt(m) の m x n 行列 P を生成。
// 符号器: y = P x を量子化。復号器: x^ = P^T (P P^T)^{-1} y^ (最小ノルム解)。
class RandomProjection {
public:
    RandomProjection(int m, int n, uint64_t seed);
    void project(const double* x, double* y) const;
    void reconstruct(const double* y, double* x) const;
    int m() const { return m_; }
    int n() const { return n_; }

private:
    int m_, n_;
    std::vector<double> P_;     // m x n
    std::vector<double> Ginv_;  // (P P^T)^{-1}, m x m
};

// 量子化前の非線形変換 (仕様 §4.5)。すべて単調・可逆。
enum class PreNonlin : uint8_t { Identity = 0, Power = 1, Asinh = 2, SignedLog = 3 };
double pre_forward(PreNonlin t, double x, double a);
double pre_inverse(PreNonlin t, double y, double a);

// 決定的 PRNG (splitmix64) — ノイズ置換・ランダム投影で符号器/復号器が共有
struct SplitMix64 {
    uint64_t s;
    explicit SplitMix64(uint64_t seed) : s(seed) {}
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    double uniform() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
};

}  // namespace fvc
