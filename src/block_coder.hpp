// FVC 内部: ブロック単位のイントラ符号化 (仕様 §5, §6, §2.3)
#pragma once
#include <cmath>
#include <cstdint>
#include <vector>

#include "fvc/entropy.hpp"
#include "fvc/frame.hpp"
#include "fvc/transform.hpp"
#include "inter.hpp"
#include "fvc/codec.hpp"

namespace fvc { class Dictionary; }

namespace fvc {

// 符号器/復号器共有の構文 I/O。w != nullptr なら符号化 (値をそのまま返す)、そうでなければ復号値を返す。
struct SymIO {
    EntropyWriter* w = nullptr;
    EntropyReader* r = nullptr;
    double* cost = nullptr;  // レート推定モード: モデルを更新せず -log2 p を積算
    bool enc = false;        // 値が既知 (符号化 or レート推定)
    static double bits(uint32_t p12) {
        static const std::vector<double> t = [] {
            std::vector<double> v(4097);
            for (int i = 1; i <= 4096; ++i) v[i] = -std::log2(i / 4096.0);
            v[0] = 12.0;
            return v;
        }();
        return t[p12];
    }
    int bit(CMModel& m, uint32_t node, uint32_t a, uint32_t b, int v) {
        if (cost) { const uint32_t p = m.predict(node, a, b); *cost += bits(v ? p : 4096 - p); return v; }
        if (w) { w->bit(m, node, a, b, v); return v; }
        return r->bit(m, node, a, b);
    }
    // 2 値化は EntropyWriter/Reader と同一 (entropy.cpp)
    uint32_t uint(CMModel& m, uint32_t a, uint32_t b, uint32_t v) {
        if (cost) {
            const uint32_t vp = v + 1;
            int e = 0;
            while ((vp >> (e + 1)) != 0) ++e;
            for (int i = 0; i < e; ++i) bit(m, 2 + i, a, b, 1);
            if (e < 31) bit(m, 2 + e, a, b, 0);
            for (int i = e - 1; i >= 0; --i) {
                if (e - 1 - i < 3) bit(m, 64 + static_cast<uint32_t>(e) * 32 + static_cast<uint32_t>(i), a, b, (vp >> i) & 1);
                else *cost += 1.0;
            }
            return v;
        }
        if (w) { w->uint(m, a, b, v); return v; }
        return r->uint(m, a, b);
    }
    int32_t sint(CMModel& m, uint32_t a, uint32_t b, int32_t v) {
        if (cost) {
            bit(m, 0, a, b, v != 0);
            if (v) { bit(m, 1, a, b, v < 0); uint(m, a, b, static_cast<uint32_t>(v < 0 ? -static_cast<int64_t>(v) : v) - 1); }
            return v;
        }
        if (w) { w->sint(m, a, b, v); return v; }
        return r->sint(m, a, b);
    }
};

struct Models {
    CMModel split, mode, cbf, last, coef_y, coef_c, band_ll, band_hi, lossless, l2, ibc, cfl, tns, e8, nf;
    CMModel inter, mvd, dict, shape, global, band_mode, lf, aqp;
};

const std::vector<int>& diag_scan(int log2s);
double qp_step(int qp, int bit_depth);
int32_t quant_dz(double c, double step, double rnd);

// イントラモード: 0=DC, 1=Planar, 2..34=角度 (HEVC 番号互換), 35=CfL (色差のみ)
constexpr int kModeDC = 0, kModePlanar = 1, kModeHor = 10, kModeVer = 26, kNumIntra = 35, kModeCfl = 35;

// フレーム単位のツール有効化 (フレームヘッダで伝送)
struct Tools {
    bool ibc = false, tns = false, e8 = false, cfl = false, nf = false;
    bool all_angular = true;  // false: {DC, Planar, H, V} のみ
    bool dict = false;        // 辞書予測 (§10)
    bool fir = true;          // FIR 動き平滑 (§7.3)
    bool rect = false;        // 長方形予測分割 (2NxN / Nx2N)
    bool tmvp = false;        // 時間方向動きベクトル候補
    bool mts = false;         // 複数変換選択 (DCT2/DST7/DCT8)
    bool pred_only = false;   // 予測のみ (残差は別経路: 動画 PQMF)
};

// インター予測の文脈 (プレーン単位)
struct InterCtx {
    bool enabled = false;
    bool bframe = false;
    int nref[2] = {0, 0};
    RefPlane l0[4], l1[4];
    MotionField* mf = nullptr;  // 輝度が書き、色差は読み取り専用
    int chroma_shift = 0;       // 色差プレーンの縮小 (420: 1)
    int gmv_x = 0, gmv_y = 0;   // グローバル動き (1/4 輝度画素)
    // 時間方向候補: 同位置ピクチャ (L1[0] があればそれ、なければ L0[0]) の動きベクトル場
    const MotionField* col = nullptr;
    int col_poc = 0, cur_poc = 0;
    int col_ref_poc[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
    int ref_poc[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
};

// 符号器探索パラメータ (ビットストリームに影響しない)
struct Search {
    int rd_modes = 4;       // RD 評価するイントラ候補数
    int ibc_range = 64;     // IBC 探索範囲 (画素)
    bool try_e8 = false, try_tns = false;
    int me_range = 16;      // 整数動き探索範囲
    bool me_bi = true;      // 双予測探索
    bool try_fir = true;
    bool qpel = true;              // 1/4 画素精密化 (false: 1/2 まで)
    bool inter_skip_intra = false; // インター候補が十分よければイントラを評価しない
    int max_rd_cands = 0;          // >0: SAD 上位 K 候補のみ RD 評価
    bool skip_split_on_skip = false; // 残差なしインター葉なら分割を試さない
    bool approx_subpel = false;    // サブ画素探索コストを双線形で近似
    bool rdoq = false;             // レート歪み最適化量子化
    bool try_mts = false;          // MTS の RD 探索
    double psy = 0;                // 心理視覚歪みの強さ (0: SSE)
    double chroma_weight = 1.0;    // 色差歪みの重み
};

class BlockCoder {
public:
    BlockCoder(Plane* rec, const Plane* org, const Plane* luma, int plane, int32_t lo, int32_t hi, double step,
               double lambda, int min_log2, int max_log2, const Tools& tools, const Search& search,
               const InterCtx* inter = nullptr, Dictionary* dict = nullptr, int tx0 = 0, int ty0 = 0, int tw = 0,
               int th = 0, int32_t leaf_base = 0);
    // CTU (cx,cy,size) を符号化/復号 (io.w があれば RD 探索も行う)
    void code_ctu(SymIO& io, Models& md, int cx, int cy, int ctu);
    // 符号器: RD 探索のみ行い CTU の RD コストを返す (再構成は rec に残る)
    double rd_ctu(int cx, int cy, int ctu);
    // CTU 単位の適応 QP (§13.1)。base_qp と、符号器では CTU ごとの dQP 決定関数を与える
    void enable_aqp(int base_qp, int bit_depth, int (*)(void*, int, int), void* ctx) ;
    // ループフィルタ用 4x4 ブロック情報
    const std::vector<int32_t>& leaf_ids() const { return leaf4_; }
    const BlockUsage& usage() const { return usage_; }
    // 帯域符号化: 符号化済みの隣接帯域 (鏡像補正済み, 同じ座標系) を文脈・予測に使う
    void set_xband(const Plane* p) { xband_ = p; }
    // 帯域符号化 (P/B): 動き補償予測の同帯域 (正規化済み) を時間方向パラメトリック予測に使う
    void set_tband(const Plane* p) { tband_ = p; }
    const std::vector<uint8_t>& leaf_flags() const { return flags4_; }

private:
    struct Leaf {
        int pt = 0;  // 0: イントラ, 1: IBC, 2: インター, 3: 辞書, 4: 帯域間予測
        MotionInfo mi;       // pt==2
        int merge = -1;      // pt==2: マージ候補番号 (-1: 明示)
        int part = 0;        // pt==2 (輝度): 0 = 2Nx2N, 1 = 2NxN (上下), 2 = Nx2N (左右)
        MotionInfo mi2;      // 第 2 区画の動き
        int merge2 = -1;
        int dict_idx = 0, dict_gain = 0;
        int xgain = 0;
        int xsrc = 0;        // pt==4: 0 = 帯域間 (ゲイン ±1..4 /4), 1 = 時間方向 (ゲイン 1..5 /4)       // pt==4: 帯域間予測ゲイン (±1..4)/4  // pt==3: サイズ別リスト内の位置, ゲイン (1/16)
        int mode = kModePlanar, alpha = 0, bvx = 0, bvy = 0;
        int qmode = 0;  // 0: デッドゾーンスカラ, 1: E8 格子 VQ
        int mts = 0;    // 変換の組 (輝度, 4..32)
        bool tns_on = false;
        TnsFilter tns;
        int nf = 0;     // ノイズ補完レベル 0..7
        int last = -1;  // qmode0: 走査順最終非ゼロ, qmode1: チャンク数-1
        std::vector<int32_t> q;  // qmode0: ラスタ順レベル, qmode1: 走査順 2y (E8 半整数を整数化)
    };

    Plane* rec_;
    const Plane* org_;
    const Plane* luma_;
    int plane_;
    int32_t lo_, hi_, mid_;
    double step_, lambda_;
    int min_log2_, max_log2_;
    Tools tools_;
    Search search_;
    InterCtx inter_;
    Dictionary* dict_;
    int cx_ = 0, cy_ = 0, ctu_ = 64;
    int bv_px_ = 0, bv_py_ = 0;  // BV 予測子
    std::vector<int8_t> modes4_;  // 4x4 単位のモード (MPM 用)
    std::vector<std::vector<int8_t>> split_map_;
    std::vector<std::vector<Leaf>> leaf_map_;  // [log2] -> grid (符号器の決定)

    // タイル (§12.4): 予測参照はタイル内に限定 → タイル単位で並列符号化/復号できる
    int tx0_ = 0, ty0_ = 0, tx1_ = 0, ty1_ = 0;
    int grid_w(int l) const { return (tx1_ - tx0_) >> l; }
    size_t gidx(int x, int y, int l) const { return static_cast<size_t>((y - ty0_) >> l) * grid_w(l) + ((x - tx0_) >> l); }
    int8_t& split_at(int x, int y, int l) { return split_map_[l][gidx(x, y, l)]; }
    Leaf& leaf_at(int x, int y, int l) { return leaf_map_[l][gidx(x, y, l)]; }
    TxType tx_for(int s) const { return s == 4 ? TxType::DST7 : TxType::DCT2; }
    // MTS (§5.2): 0 = 既定, 1..4 = (水平, 垂直) ∈ {DST7,DST7} {DCT8,DST7} {DST7,DCT8} {DCT8,DCT8}
    void tx_pair(int mts, int s, TxType& th, TxType& tv) const {
        static constexpr TxType kH[5] = {TxType::DCT2, TxType::DST7, TxType::DCT8, TxType::DST7, TxType::DCT8};
        static constexpr TxType kV[5] = {TxType::DCT2, TxType::DST7, TxType::DST7, TxType::DCT8, TxType::DCT8};
        if (mts == 0) { th = tv = tx_for(s); return; }
        th = kH[mts]; tv = kV[mts];
    }

    void mpm(int x0, int y0, int& m0, int& m1) const;
    void set_modes4(int x0, int y0, int s, int mode);
    void predict(const Leaf& lf, int x0, int y0, int l, int32_t* pred) const;
    void intra_angular(int x0, int y0, int s, int mode, int32_t* pred) const;
    void intra_cfl(int x0, int y0, int s, int alpha, int32_t* pred) const;
    int fit_cfl_alpha(int x0, int y0, int s) const;
    bool ibc_valid(int rx, int ry, int s) const;
    void ibc_search(int x0, int y0, int s, int& bx, int& by) const;
    bool is_luma() const { return plane_ == 0; }
    bool luma_motion_uniform(int x0, int y0, int s) const;
    static constexpr int kMaxMerge = 5;
    int merge_list(int x0, int y0, int w, int h, MotionInfo* out) const;
    bool temporal_cand(int x, int y, MotionInfo& out) const;
    void code_motion(SymIO& io, Models& md, uint32_t L, int x0, int y0, int w, int h, MotionInfo& m, int& merge);
    void fill_mf(const Leaf& lf, int x0, int y0, int s);
    void rect_search(int x0, int y0, int s, std::vector<Leaf>& cands);
    void mv_pred(int x0, int y0, int list, int ref, int& px, int& py) const;
    int64_t me_cost(int x0, int y0, int w, int h, const MotionInfo& mi) const;
    void motion_search(int x0, int y0, int w, int h, std::vector<Leaf>& cands) const;
    void dict_search(int x0, int y0, int s, std::vector<Leaf>& cands) const;
    void rdoq(Leaf& lf, int l, const std::vector<double>& e, uint32_t xb) const;
    uint32_t xb_ctx(int x0, int y0, int s) const;
    const Plane* xband_ = nullptr;
    const Plane* tband_ = nullptr;
    void quantize(Leaf& lf, int x0, int y0, int l, const int32_t* pred) const;
    void reconstruct(const Leaf& lf, int x0, int y0, int l, const int32_t* pred);
    double leaf_bits(const Leaf& lf, int x0, int y0, int l) const;
    double sse(int x0, int y0, int s) const;
    void save(int x0, int y0, int s, std::vector<int32_t>& b) const;
    void restore(int x0, int y0, int s, const std::vector<int32_t>& b);
    double rd_node(int x0, int y0, int l);
    void code_node(SymIO& io, Models& md, int x0, int y0, int l);
    void code_leaf(SymIO& io, Models& md, Leaf& lf, int x0, int y0, int l);
    void leaf_syntax(SymIO& io, Models& md, Leaf& lf, int x0, int y0, int l);
    double leaf_rate(Leaf& lf, int x0, int y0, int l);
    double split_rate(int x0, int y0, int l, int split);
    Models* md_ = nullptr;
    BlockUsage usage_;
    bool aqp_ = false;
    int base_qp_ = 0, bit_depth_ = 8, prev_dqp_ = 0;
    int (*dqp_fn_)(void*, int, int) = nullptr;
    void* dqp_ctx_ = nullptr;
    std::vector<int32_t> leaf4_;
    std::vector<uint8_t> flags4_;
    int32_t leaf_counter_ = 0;
    Models own_md_;
};

}  // namespace fvc
