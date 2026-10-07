#include "fvc/entropy.hpp"

#include <algorithm>
#include <cmath>

namespace fvc {

// ---------------- rANS ----------------
namespace {
constexpr uint32_t kRansL = 1u << 23;  // 正規化下限、状態 ∈ [L, 256L)
}

std::vector<uint8_t> RansBitEncoder::finish() {
    std::vector<uint8_t> out;
    out.reserve(syms_.size() / 8 + 8);
    uint32_t x = kRansL;
    for (size_t i = syms_.size(); i-- > 0;) {
        const uint32_t p1 = syms_[i].p1;
        const uint32_t start = syms_[i].bit ? 0 : p1;
        const uint32_t freq = syms_[i].bit ? p1 : kProbScale - p1;
        const uint32_t x_max = ((kRansL >> kProbBits) << 8) * freq;
        while (x >= x_max) { out.push_back(static_cast<uint8_t>(x & 0xff)); x >>= 8; }
        x = ((x / freq) << kProbBits) + (x % freq) + start;
    }
    for (int i = 0; i < 4; ++i) { out.push_back(static_cast<uint8_t>(x & 0xff)); x >>= 8; }
    std::reverse(out.begin(), out.end());
    syms_.clear();
    return out;
}

RansBitDecoder::RansBitDecoder(const uint8_t* data, size_t size) : d_(data), n_(size) {
    for (int i = 0; i < 4; ++i) x_ = (x_ << 8) | byte();
}

int RansBitDecoder::get(uint32_t p1) {
    const uint32_t s = x_ & (kProbScale - 1);
    int bit;
    if (s < p1) { bit = 1; x_ = p1 * (x_ >> kProbBits) + s; }
    else        { bit = 0; x_ = (kProbScale - p1) * (x_ >> kProbBits) + s - p1; }
    while (x_ < kRansL) x_ = (x_ << 8) | byte();
    return bit;
}

// ---------------- stretch / squash ----------------
int32_t squash(int32_t s) {
    if (s > 2047) s = 2047;
    if (s < -2047) s = -2047;
    const double p = 4096.0 / (1.0 + std::exp(-s / 256.0));
    return std::clamp(static_cast<int32_t>(p), 1, 4095);
}

int32_t stretch(int32_t p) {
    static std::vector<int16_t> table = [] {
        std::vector<int16_t> t(4096);
        int pi = 0;
        for (int s = -2047; s <= 2047; ++s) {
            const int v = squash(s);
            for (int j = pi; j <= v; ++j) t[j] = static_cast<int16_t>(s);
            pi = v + 1;
        }
        for (int j = pi; j < 4096; ++j) t[j] = 2047;
        return t;
    }();
    return table[std::clamp(p, 0, 4095)];
}

// ---------------- CM モデル ----------------
namespace {
inline uint32_t hash3(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t h = a * 0x9E3779B1u ^ (b + 0x7F4A7C15u) * 0x85EBCA6Bu ^ (c + 0x165667B1u) * 0xC2B2AE35u;
    return h ^ (h >> 15);
}
}  // namespace

CMModel::CMModel() = default;

// テーブルは初回使用時に確保 (使われない構文要素のモデルは 0 コスト)
void CMModel::init() {
    static const std::vector<uint16_t> apm_init = [] {
        std::vector<uint16_t> a(33 * 1024);
        for (size_t c = 0; c < 1024; ++c)
            for (int j = 0; j < 33; ++j) a[c * 33 + j] = static_cast<uint16_t>(squash((j - 16) * 128) * 16);
        return a;
    }();
    tab_.assign(static_cast<size_t>(kInputs) << kTableBits, BitCounter{});
    weights_.assign(static_cast<size_t>(1024) * (kInputs + 1), (1 << 16) / kInputs);
    apm_ = apm_init;
}

uint32_t CMModel::predict(uint32_t node, uint32_t a, uint32_t b) {
    if (tab_.empty()) init();
    const uint32_t mask = (1u << kTableBits) - 1;
    idx_[0] = (hash3(node, 0, 0) & mask);
    idx_[1] = (hash3(node, a, 1) & mask) | (1u << kTableBits);
    idx_[2] = (hash3(node, b, 2) & mask) | (2u << kTableBits);
    idx_[3] = (hash3(node, a, b ^ 0xABCDu) & mask) | (3u << kTableBits);
    for (int i = 0; i < kInputs; ++i) st_[i] = stretch(tab_[idx_[i]].p12());
    st_[kInputs] = 256;  // バイアス入力
    mix_ctx_ = (node * 31 + (a & 0x1f)) & 1023;
    const int32_t* w = &weights_[mix_ctx_ * (kInputs + 1)];
    int64_t dot = 0;
    for (int i = 0; i <= kInputs; ++i) dot += static_cast<int64_t>(w[i]) * st_[i];
    pmix_ = squash(static_cast<int32_t>(dot >> 16));
    // APM: stretch(p) を 33 区間で補間
    const int32_t s = stretch(pmix_) + 2048;  // [1, 4095]
    const int32_t lo = s >> 7;
    apm_w_ = s & 127;
    apm_idx_ = (node & 1023) * 33 + lo;
    const int32_t pa = (apm_[apm_idx_] * (128 - apm_w_) + apm_[apm_idx_ + 1] * apm_w_) >> 11;
    pfinal_ = static_cast<uint32_t>(std::clamp((pmix_ + 3 * pa) >> 2, 1, 4095));
    return pfinal_;
}

void CMModel::update(int bit) {
    // ミキサ: w_i += η * err * st_i
    const int32_t err = ((bit << 12) - pmix_) * 6;
    int32_t* w = &weights_[mix_ctx_ * (kInputs + 1)];
    for (int i = 0; i <= kInputs; ++i) w[i] += (st_[i] * err) >> 10;
    for (int i = 0; i < kInputs; ++i) tab_[idx_[i]].update(bit);
    const int32_t target = bit ? 65535 : 0;
    const uint32_t j = apm_w_ < 64 ? apm_idx_ : apm_idx_ + 1;
    apm_[j] = static_cast<uint16_t>(apm_[j] + ((target - apm_[j]) >> 6));
}

// ---------------- 2値化 ----------------
void EntropyWriter::bit(CMModel& m, uint32_t node, uint32_t a, uint32_t b, int v) {
    enc_.put(v, m.predict(node, a, b));
    m.update(v);
}

int EntropyReader::bit(CMModel& m, uint32_t node, uint32_t a, uint32_t b) {
    const int v = dec_.get(m.predict(node, a, b));
    m.update(v);
    return v;
}

// node 番号: 0=ゼロ, 1=符号, 2..33=指数単進, 64+e*32+i = 仮数ビット
void EntropyWriter::uint(CMModel& m, uint32_t a, uint32_t b, uint32_t v) {
    const uint32_t vp = v + 1;
    int e = 0;
    while ((vp >> (e + 1)) != 0) ++e;  // vp ∈ [2^e, 2^(e+1))
    for (int i = 0; i < e; ++i) bit(m, 2 + i, a, b, 1);
    if (e < 31) bit(m, 2 + e, a, b, 0);
    for (int i = e - 1; i >= 0; --i) {
        const uint32_t node = 64 + static_cast<uint32_t>(e) * 32 + static_cast<uint32_t>(i);
        if (e - 1 - i < 3) bit(m, node, a, b, (vp >> i) & 1);
        else bypass((vp >> i) & 1);  // 下位ビットは均等確率
    }
}

uint32_t EntropyReader::uint(CMModel& m, uint32_t a, uint32_t b) {
    int e = 0;
    while (e < 31 && bit(m, 2 + e, a, b)) ++e;
    uint32_t vp = 1;
    for (int i = e - 1; i >= 0; --i) {
        const uint32_t node = 64 + static_cast<uint32_t>(e) * 32 + static_cast<uint32_t>(i);
        const int v = (e - 1 - i < 3) ? bit(m, node, a, b) : bypass();
        vp = (vp << 1) | static_cast<uint32_t>(v);
    }
    return vp - 1;
}

void EntropyWriter::sint(CMModel& m, uint32_t a, uint32_t b, int32_t v) {
    bit(m, 0, a, b, v != 0);
    if (v == 0) return;
    bit(m, 1, a, b, v < 0);
    uint(m, a, b, static_cast<uint32_t>(v < 0 ? -static_cast<int64_t>(v) : v) - 1);
}

int32_t EntropyReader::sint(CMModel& m, uint32_t a, uint32_t b) {
    if (!bit(m, 0, a, b)) return 0;
    const int neg = bit(m, 1, a, b);
    const int64_t mag = static_cast<int64_t>(uint(m, a, b)) + 1;
    return static_cast<int32_t>(neg ? -mag : mag);
}

}  // namespace fvc
