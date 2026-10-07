// FVC 内部: ブロック単位のイントラ符号化 (仕様 §5, §6, §2.3)
#pragma once
#include <cmath>
#include <cstdint>
#include <vector>

#include "fvc/entropy.hpp"
#include "fvc/frame.hpp"
#include "fvc/transform.hpp"
#include "inter.hpp"

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
    CMModel inter, mvd, dict, shape, global, band_mode;
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
};

// 符号器探索パラメータ (ビットストリームに影響しない)
struct Search {
    int rd_modes = 4;       // RD 評価するイントラ候補数
    int ibc_range = 64;     // IBC 探索範囲 (画素)
    bool try_e8 = false, try_tns = false;
    int me_range = 16;      // 整数動き探索範囲
    bool me_bi = true;      // 双予測探索
    bool try_fir = true;
};

class BlockCoder {
public:
    BlockCoder(Plane* rec, const Plane* org, const Plane* luma, int plane, int32_t lo, int32_t hi, double step,
               double lambda, int min_log2, int max_log2, const Tools& tools, const Search& search,
               const InterCtx* inter = nullptr, Dictionary* dict = nullptr);
    // CTU (cx,cy,size) を符号化/復号 (io.w があれば RD 探索も行う)
    void code_ctu(SymIO& io, Models& md, int cx, int cy, int ctu);
    // 符号器: RD 探索のみ行い CTU の RD コストを返す (再構成は rec に残る)
    double rd_ctu(int cx, int cy, int ctu);

private:
    struct Leaf {
        int pt = 0;  // 0: イントラ, 1: IBC, 2: インター, 3: 辞書
        MotionInfo mi;       // pt==2
        int merge = -1;      // pt==2: マージ候補番号 (-1: 明示)
        int dict_idx = 0, dict_gain = 0;  // pt==3: サイズ別リスト内の位置, ゲイン (1/16)
        int mode = kModePlanar, alpha = 0, bvx = 0, bvy = 0;
        int qmode = 0;  // 0: デッドゾーンスカラ, 1: E8 格子 VQ
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

    int grid_w(int l) const { return rec_->w >> l; }
    int8_t& split_at(int x, int y, int l) { return split_map_[l][(y >> l) * grid_w(l) + (x >> l)]; }
    Leaf& leaf_at(int x, int y, int l) { return leaf_map_[l][(y >> l) * grid_w(l) + (x >> l)]; }
    TxType tx_for(int s) const { return s == 4 ? TxType::DST7 : TxType::DCT2; }

    void mpm(int x0, int y0, int& m0, int& m1) const;
    void set_modes4(int x0, int y0, int s, int mode);
    void predict(const Leaf& lf, int x0, int y0, int l, int32_t* pred) const;
    void intra_angular(int x0, int y0, int s, int mode, int32_t* pred) const;
    void intra_cfl(int x0, int y0, int s, int alpha, int32_t* pred) const;
    int fit_cfl_alpha(int x0, int y0, int s) const;
    bool ibc_valid(int rx, int ry, int s) const;
    void ibc_search(int x0, int y0, int s, int& bx, int& by) const;
    bool is_luma() const { return plane_ == 0; }
    int merge_list(int x0, int y0, MotionInfo* out) const;
    void mv_pred(int x0, int y0, int list, int ref, int& px, int& py) const;
    int64_t me_cost(int x0, int y0, int s, const MotionInfo& mi) const;
    void motion_search(int x0, int y0, int s, std::vector<Leaf>& cands) const;
    void dict_search(int x0, int y0, int s, std::vector<Leaf>& cands) const;
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
    Models own_md_;
};

}  // namespace fvc
