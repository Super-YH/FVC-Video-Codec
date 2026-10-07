// FVC: 符号器 / 復号器 (段階2: I フレーム, ブロック変換 or PQMF 帯域, L2 ロスレス層)
#pragma once
#include <cstdint>
#include <memory>
#include <vector>

#include "fvc/frame.hpp"

namespace fvc {

enum class Preset : uint8_t { Faster = 0, Fast = 1, Medium = 2, Slow = 3, Placebo = 4 };

struct EncoderConfig {
    int qp = 32;              // 0..63
    bool lossy_layer = true;  // false: 純ロスレス (MED 予測のみ)
    bool l2_lossless = false; // true: 最終残差をロスレス符号化 → 完全可逆
    int pqmf_log2 = 0;        // 0: 帯域分解なし, 1..4: M = 2^n (帯域数 M^2 ≤ 256)
    Preset preset = Preset::Medium;
};

// ユニット種別 (仕様 §12.1)
enum class UnitType : uint8_t { Seq = 0, Frame = 1, Tile = 2, Dict = 3, Sei = 4, Eos = 5 };

struct FrameStats { size_t bytes = 0; double psnr[3] = {0, 0, 0}; };

class Encoder {
public:
    Encoder(const VideoInfo& info, const EncoderConfig& cfg);
    std::vector<uint8_t> sequence_header() const;
    // フレームを符号化し FRAME ユニットを返す。recon に復号器と同一の再構成を返す。
    std::vector<uint8_t> encode(const Frame& f, Frame* recon = nullptr);
    std::vector<uint8_t> end_of_stream() const;
private:
    VideoInfo info_;
    EncoderConfig cfg_;
};

class Decoder {
public:
    // ストリーム全体を逐次復号。unit 境界は内部で解析。
    explicit Decoder(const std::vector<uint8_t>& stream);
    bool ok() const { return ok_; }
    const VideoInfo& info() const { return info_; }
    bool next(Frame& f);  // false: 終端
private:
    std::vector<uint8_t> s_;
    size_t pos_ = 0;
    bool ok_ = false;
    VideoInfo info_;
};

}  // namespace fvc
