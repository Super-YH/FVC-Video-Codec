// FVC: コンテキストミキシング (CM) 確率モデル + 2値 rANS。
//  すべての構文要素は 2値化され、各ビットは CM で確率 p1 (12bit) を得て rANS で符号化される。
//  rANS は LIFO のため符号器は (bit, p1) を全て記録し、ストリーム終了時に逆順で符号化する。
#pragma once
#include <array>
#include <cstdint>
#include <vector>

namespace fvc {

constexpr int kProbBits = 12;
constexpr uint32_t kProbScale = 1u << kProbBits;

// ---------- 2値 rANS ----------
class RansBitEncoder {
public:
    void put(int bit, uint32_t p1) { syms_.push_back({static_cast<uint16_t>(p1), static_cast<uint8_t>(bit)}); }
    std::vector<uint8_t> finish();  // 逆順で符号化し、復号順のバイト列を返す
private:
    struct Sym { uint16_t p1; uint8_t bit; };
    std::vector<Sym> syms_;
};

class RansBitDecoder {
public:
    RansBitDecoder(const uint8_t* data, size_t size);
    int get(uint32_t p1);
private:
    const uint8_t* d_;
    size_t n_, pos_ = 0;
    uint32_t x_ = 0;
    uint8_t byte() { return pos_ < n_ ? d_[pos_++] : 0; }
};

// ---------- ロジスティック領域 ----------
int32_t stretch(int32_t p);  // p∈(0,4096) -> ln(p/(1-p)) * 256, [-2047,2047]
int32_t squash(int32_t s);   // 逆変換

// 適応確率カウンタ (16bit 精度、2 種学習率)
struct BitCounter {
    uint16_t fast = 32768, slow = 32768;
    int32_t p12() const { return (static_cast<int32_t>(fast) + slow) >> 5; }
    void update(int bit) {
        if (bit) { fast += (65535 - fast) >> 4; slow += (65535 - slow) >> 7; }
        else     { fast -= fast >> 4;           slow -= slow >> 7; }
    }
};

// 構文要素ごとのモデル: 複数の文脈 (呼び出し側が与えるハッシュ) の予測を
// ニューラルミキサで混合し、APM(SSE) で補正。
class CMModel {
public:
    static constexpr int kInputs = 4;   // order0 / ctxA / ctxB / ctxA^ctxB
    static constexpr int kTableBits = 14;

    CMModel();
    // ctx_a, ctx_b: 呼び出し側が与える文脈 (近傍シンボル、帯域番号、位置など)
    uint32_t predict(uint32_t node, uint32_t ctx_a, uint32_t ctx_b);
    void update(int bit);

private:
    void init();
    std::vector<BitCounter> tab_;
    std::vector<int32_t> weights_;      // [mixer_ctx][kInputs+1]
    std::vector<uint16_t> apm_;         // [33 * 1024]
    std::array<uint32_t, kInputs> idx_{};
    std::array<int32_t, kInputs + 1> st_{};
    uint32_t mix_ctx_ = 0, apm_idx_ = 0;
    int32_t pmix_ = 2048, apm_w_ = 0;
    uint32_t pfinal_ = 2048;
};

// 符号器/復号器の統一 I/F (同じ 2 値化コードで両側を書けるように)
class EntropyWriter {
public:
    void bit(CMModel& m, uint32_t node, uint32_t a, uint32_t b, int v);
    void bypass(int v) { enc_.put(v, kProbScale / 2); }
    // 符号付き整数: 0 フラグ / 符号 / 指数 (単進) / 仮数 (指数別文脈)
    void sint(CMModel& m, uint32_t ctx_a, uint32_t ctx_b, int32_t v);
    void uint(CMModel& m, uint32_t ctx_a, uint32_t ctx_b, uint32_t v);
    std::vector<uint8_t> finish() { return enc_.finish(); }
private:
    RansBitEncoder enc_;
};

class EntropyReader {
public:
    EntropyReader(const uint8_t* d, size_t n) : dec_(d, n) {}
    int bit(CMModel& m, uint32_t node, uint32_t a, uint32_t b);
    int bypass() { return dec_.get(kProbScale / 2); }
    int32_t sint(CMModel& m, uint32_t ctx_a, uint32_t ctx_b);
    uint32_t uint(CMModel& m, uint32_t ctx_a, uint32_t ctx_b);
private:
    RansBitDecoder dec_;
};

}  // namespace fvc
