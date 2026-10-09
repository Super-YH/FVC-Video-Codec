// FVC: 符号器 / 復号器 (I/P/B/COPY フレーム, ブロック変換 or PQMF 帯域, 図形, 辞書, L2 ロスレス層)
#pragma once
#include <cstdint>
#include <map>
#include <memory>
#include <vector>

#include "fvc/frame.hpp"

namespace fvc {

enum class Preset : uint8_t { Faster = 0, Fast = 1, Medium = 2, Slow = 3, Placebo = 4 };

struct EncoderConfig {
    int qp = 32;              // 0..63 (I フレーム基準。P は +1, B は +3)
    bool lossy_layer = true;  // false: 純ロスレス (MED 予測のみ)
    bool l2_lossless = false; // true: 最終残差をロスレス符号化 → 完全可逆
    int pqmf_log2 = 0;        // 0: 帯域分解なし, 1..4: M = 2^n (帯域数 M^2 ≤ 256)
    Preset preset = Preset::Medium;
    bool psy = false;         // ノイズ補完/帯域ノイズ置換 (PSNR は下がるが質感保持)
    int ibc = -1;             // -1: プリセット依存, 0/1: 強制
    int e8 = -1, tns = -1, cfl = -1, dict = -1, shapes = -1, fir = -1, loop_filter = -1, rect = -1, tmvp = -1, alf = -1, mts = -1;
    int keyint = 0;           // I フレーム間隔 (0: 先頭のみ)
    int bframes = -1;         // アンカー間の B 枚数 (階層 B, 0..15)。-1: プリセット依存
    int refs = -1;            // P の参照枚数 (1..4), -1: プリセット依存
    bool copy_frames = true;  // 静止区間を COPY フレームで符号化
    int aqp = -1;             // CTU 適応 QP (静止領域の QP を下げる)。-1: プリセット依存
    bool keep_recon = false;  // 再構成を表示順で保持 (テスト/PSNR 用)
    int tile_cols = 0, tile_rows = 0;  // 0: プリセット依存
    int threads = 0;          // 0: ハードウェアスレッド数
    int chroma_qp_offset = 0; // 色差 QP オフセット (-12..12)
    double target_ssim = 0;   // >0: フレームごとに輝度 SSIM がこの値以上となる最大 QP を探索
};

// ユニット種別 (仕様 §12.1)
enum class UnitType : uint8_t { Seq = 0, Frame = 1, Tile = 2, Dict = 3, Sei = 4, Eos = 5 };
enum class FrameType : uint8_t { I = 0, P = 1, B = 2, Copy = 3 };

// ブロック使用統計 (輝度の面積 [画素] で集計、CfL のみ色差)
struct BlockUsage {
    uint64_t intra = 0, inter = 0, ibc = 0, dict = 0;  // 予測種別
    uint64_t merge = 0, skip = 0, rect = 0, bi = 0;    // インターの内訳 (skip = マージかつ残差なし)
    uint64_t size[7] = {0, 0, 0, 0, 0, 0, 0};          // 4x4 .. 256x256 (log2 2..8)
    uint64_t tns = 0, e8 = 0, cfl = 0, leaves = 0;
    void add(const BlockUsage& o) {
        intra += o.intra; inter += o.inter; ibc += o.ibc; dict += o.dict;
        merge += o.merge; skip += o.skip; rect += o.rect; bi += o.bi;
        for (int i = 0; i < 7; ++i) size[i] += o.size[i];
        tns += o.tns; e8 += o.e8; cfl += o.cfl; leaves += o.leaves;
    }
};

struct FrameStats {
    int poc = 0;
    FrameType type = FrameType::I;
    size_t bytes = 0;
    int qp = 0;
    int trials = 1;         // 目標 SSIM 探索での符号化回数
    double psnr[3] = {0, 0, 0};
    double ssim = 0;        // 輝度 SSIM
    BlockUsage usage;
};

struct CodecState;  // DPB と辞書 (符号器・復号器で同一に更新)

class Encoder {
public:
    Encoder(const VideoInfo& info, const EncoderConfig& cfg);
    ~Encoder();
    std::vector<uint8_t> sequence_header() const;
    // 表示順に 1 フレーム入力。B フレームの並べ替えにより 0 個以上の FRAME ユニットを返す。
    std::vector<uint8_t> encode(const Frame& f);
    // 保留フレームを全て符号化し EOS を付ける
    std::vector<uint8_t> flush();
    const std::vector<FrameStats>& stats() const { return stats_; }
    const std::map<int, Frame>& recon() const { return recon_; }

private:
    VideoInfo info_;
    EncoderConfig cfg_;
    std::unique_ptr<CodecState> st_;
    int next_poc_ = 0;
    std::vector<std::pair<int, Frame>> pend_;  // 未符号化フレーム (表示順)
    std::vector<uint8_t> flush_pending();
    void encode_b_range(const std::vector<std::pair<int, Frame>>& v, int a, int b, int depth, std::vector<uint8_t>& out);
    std::vector<FrameStats> stats_;
    std::map<int, Frame> recon_;
    std::vector<uint8_t> encode_picture(const Frame& f, int poc, FrameType type, int depth = 0,
                                        const std::vector<const Frame*>& look = {});
    bool key_pending_ = false;
    int last_q_[4] = {-1, -1, -1, -1};  // 目標 SSIM: フレーム種別ごとの直前 QP
};

class Decoder {
public:
    explicit Decoder(const std::vector<uint8_t>& stream, int threads = 0);
    ~Decoder();
    bool ok() const { return ok_; }
    const VideoInfo& info() const { return info_; }
    bool next(Frame& f);  // 表示順に返す。false: 終端
private:
    std::vector<uint8_t> s_;
    size_t pos_ = 0;
    bool ok_ = false;
    VideoInfo info_;
    std::unique_ptr<CodecState> st_;
    std::map<int, Frame> out_;
    int next_out_ = 0;
    bool eos_ = false;
    int threads_ = 1;
    bool decode_unit();
};

}  // namespace fvc
