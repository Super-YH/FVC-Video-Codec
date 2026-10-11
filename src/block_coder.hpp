// FVC 内部: ブロック単位のイントラ符号化 (仕様 §5, §6, §2.3)
#pragma once
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <vector>
#include <unordered_map>

#include "fvc/entropy.hpp"
#include "fvc/frame.hpp"
#include "fvc/transform.hpp"
#include "inter.hpp"
#include "gs_coder.hpp"
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
    uint32_t uintc(CMModel& m, uint32_t a, uint32_t b, uint32_t v) {
        if (cost) {
            const uint32_t vp = v + 1;
            int e = 0;
            while ((vp >> (e + 1)) != 0) ++e;
            for (int i = 0; i < e; ++i) bit(m, 2 + i, a, b, 1);
            if (e < 31) bit(m, 2 + e, a, b, 0);
            uint32_t tree = 1;
            for (int i = e - 1, k = 0; i >= 0; --i, ++k) {
                const int bt = (vp >> i) & 1;
                const uint32_t node = k < 5 ? 3000u + static_cast<uint32_t>(e) * 64u + tree : 5000u + static_cast<uint32_t>(e) * 32u + static_cast<uint32_t>(k);
                bit(m, node, a, b, bt);
                if (k < 5) tree = (tree << 1) | static_cast<uint32_t>(bt);
            }
            return v;
        }
        if (w) { w->uintc(m, a, b, v); return v; }
        return r->uintc(m, a, b);
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
    CMModel inter, mvd, dict, shape, global, band_mode, lf, aqp, gs, line;
};

const std::vector<int>& diag_scan(int log2s);
double qp_step(int qp, int bit_depth);
int32_t quant_dz(double c, double step, double rnd);

// イントラモード: 0=DC, 1=Planar, 2..34=角度 (HEVC 番号互換), 35=CfL (色差のみ)
// 36 = Smooth, 37 = Paeth (AV1 型)。35 (CfL) は通常モードの番号空間に含めない
constexpr int kModeDC = 0, kModePlanar = 1, kModeHor = 10, kModeVer = 26, kNumIntra = 38, kModeCfl = 35;
constexpr int kModeSmooth = 36, kModePaeth = 37;
// 特許安全イントラ (Tools::safe_intra): 64..95 = 線方向 d (角度 d·180°/32)。線の両端の参照を距離で重み付け補間。
// このとき kModePlanar は参照への最小二乗平面 a + bx + cy を表す
constexpr int kModeLine0 = 64, kNumLine = 32;

// フレーム単位のツール有効化 (フレームヘッダで伝送)
struct Tools {
    bool ibc = false, tns = false, e8 = false, cfl = false, nf = false;
    bool all_angular = true;  // false: {DC, Planar, H, V} のみ
    bool dict = false;        // 辞書予測 (§10)
    bool fir = true;          // FIR 動き平滑 (§7.3)
    bool rect = false;        // 長方形予測分割 (2NxN / Nx2N)
    bool tmvp = false;        // 時間方向動きベクトル候補
    bool no_merge = false;    // マージ候補を使わない (特許安全モード)
    bool no_obmc = false;     // OBMC を使わない (特許安全モード)
    bool no_sp = false;       // Smooth/Paeth イントラを使わない (特許安全モード)
    bool line_mv = false;     // 線に基づく動き場のモード (lines.hpp)
    bool safe_intra = false;  // 特許安全イントラ: DC / 最小二乗平面 / 32 方向の両側線補間 (モードは近傍を文脈にして直接符号化)
    bool sr = false;          // ブロック超解像: 残差を 1/2 解像度で符号化し予測に導かれた補間で戻す
    bool mts = false;         // 複数変換選択 (DCT2/DST7/DCT8)
    bool pred_only = false;   // 予測のみ (残差は別経路: 動画 PQMF)
    bool band_ns = false;     // 帯域の時間差分ノイズ置換 (心理視覚)
    bool inter_ns = false;    // インター残差なしブロックの高域ノイズ置換 (心理視覚)
    bool firb = false;        // FIR 動き基底 ψ1..4 (§7.3)
    bool gs = false;          // 利得形状分離・バンド/パーティション符号化 (§5.3–5.7)
};

// インター予測の文脈 (プレーン単位)
struct InterCtx {
    bool enabled = false;
    bool bframe = false;
    int nref[2] = {0, 0};
    RefPlane l0[4], l1[4];
    MotionField* mf = nullptr;  // 輝度が書き、色差は読み取り専用
    int chroma_shift = 0;       // 色差プレーンの縮小 (420: 1)
    int gmv_x = 0, gmv_y = 0;   // グローバル動き (1/4 輝度画素, 画面中心での値)
    GlobalModel gm;             // グローバル動きモデル (§7.2)。位置ごとの MV は gm.mv_at
    void gmv_at(int x, int y, int& mx, int& my) const { gm.mv_at(x, y, mx, my); }
    // 時間方向候補: 同位置ピクチャ (L1[0] があればそれ、なければ L0[0]) の動きベクトル場
    const MotionField* col = nullptr;
    int col_poc = 0, cur_poc = 0;
    int col_ref_poc[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
    int ref_poc[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
    // 線に基づく動き場 (輝度 4x4 単位の (mvx, mvy), L0 先頭参照)。nullptr なら無効
    const std::vector<int16_t>* lfield = nullptr;
    int lf_w4 = 0, lf_h4 = 0;
};

// 符号器探索パラメータ (ビットストリームに影響しない)
struct Search {
    int rd_modes = 4;       // RD 評価するイントラ候補数
    int ibc_range = 64;     // IBC 探索範囲 (画素)
    bool try_e8 = false, try_tns = false;
    int me_range = 16;      // 整数動き探索範囲
    bool adam_me = false;   // 整数探索を Adam (連続 MV の勾配最適化) に置き換える
    bool me_bi = true;      // 双予測探索
    bool try_fir = true;
    bool qpel = true;              // 1/4 画素精密化 (false: 1/2 まで)
    bool inter_skip_intra = false; // インター候補が十分よければイントラを評価しない
    int max_rd_cands = 0;          // >0: SAD 上位 K 候補のみ RD 評価
    bool requant = false;          // 符号化時に再量子化する (従来動作、比較用)
    int dict_age = 48;             // 学習原子をこの根ブロック数使われなければ捨てる (符号器の方針、実測で 4〜48 は誤差内)
    double dict_bias = 64.0;       // 辞書を使う葉の RD 罰則 (ビット)
    int refine_top = 2;            // MTS/E8/利得形状を試す上位候補数 (1 段目の J の順)
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
    // 変化マスク (LL 帯域の |再構成 - 時間方向予測|, 正規化済み) を cbf/係数の文脈に使う
    void set_cband(const Plane* p) { cband_ = p; }
    void set_noise_seed(uint64_t s) { noise_seed_ = s; }
    const std::vector<uint8_t>& leaf_flags() const { return flags4_; }

private:
    struct Leaf {
        int pt = 0;  // 0: イントラ, 1: IBC, 2: インター, 3: 辞書, 4: 帯域間予測
        MotionInfo mi;       // pt==2
        int merge = -1;      // pt==2: マージ候補番号 (-1: 明示)
        int part = 0;        // pt==2 (輝度): 0 = 2Nx2N, 1 = 2NxN (上下), 2 = Nx2N (左右)
        MotionInfo mi2;      // 第 2 区画の動き
        int merge2 = -1;
        int lmode = 0;       // pt==2 (輝度, 2Nx2N): 1 = 線の動き場で 4x4 ごとに予測 (動き情報は送らない)
        int sr = 0;          // 残差の超解像: 0 = なし, 1 = 双線形, 2 = 予測の勾配に沿う補間 (係数は (s/2)^2)
        int fir[4] = {0, 0, 0, 0};
        int cg[3] = {8, 8, 8};      // pt==2 輝度: 予測係数のバンド別利得 a_b/8 (低/中/高, §7.6-2 係数コピー)  // pt==2 輝度: FIR 動き基底係数 ψ (∂x, ∂y, ∂xx, ∂yy)/16 (§7.3)
        int dict_idx = 0, dict_gain = 0;
        // pt==3: 原子の重み付き和 (最大 kMaxAtoms 個)。予測 = DC + Σ gain_k/16 · 原子_k。辞書の葉は残差を持たない
        static constexpr int kMaxAtoms = 16;
        int nat = 0;
        int16_t at_idx[kMaxAtoms] = {}, at_gain[kMaxAtoms] = {};
        int8_t at_dx[kMaxAtoms] = {}, at_dy[kMaxAtoms] = {};  // 原子の平行移動 (画素, ±kAtomShift)
        int dres = -1, dres_g = 0;
        double rd_dist = -1;  // 符号器: RD 評価時の歪み (診断用)  // 残差の辞書原子 (任意の予測に g/16 · 原子を加える)。-1: なし
        int xgain = 0;
        int xsrc = 0;
        int ns = 0;          // pt==4 時間方向・残差なし: ノイズ置換レベル 0..7 (包絡は参照帯域の振幅)        // pt==4: 0 = 帯域間 (ゲイン ±1..4 /4), 1 = 時間方向 (ゲイン 1..5 /4)       // pt==4: 帯域間予測ゲイン (±1..4)/4  // pt==3: サイズ別リスト内の位置, ゲイン (1/16)
        int mode = kModePlanar, alpha = 0, bvx = 0, bvy = 0;
        int qmode = 0;  // 0: デッドゾーンスカラ, 1: E8 格子 VQ, 2: 利得形状 (GsBlock)
        GsBlock gs;
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
        static constexpr TxType kH[9] = {TxType::DCT2, TxType::DST7, TxType::DCT8, TxType::DST7, TxType::DCT8,
                                         TxType::DST7, TxType::DCT2, TxType::IDTX, TxType::DCT2};
        static constexpr TxType kV[9] = {TxType::DCT2, TxType::DST7, TxType::DST7, TxType::DCT8, TxType::DCT8,
                                         TxType::DCT2, TxType::DST7, TxType::IDTX, TxType::DCT8};
        if (mts == 0) { th = tv = tx_for(s); return; }
        th = kH[mts]; tv = kV[mts];
    }

    void mpm(int x0, int y0, int& m0, int& m1) const;
    void mpm6(int x0, int y0, int* out) const;  // 6 候補 MPM (全角度モード時)
    void set_modes4(int x0, int y0, int s, int mode);
    void predict(const Leaf& lf, int x0, int y0, int l, int32_t* pred) const;
    void predict_base(const Leaf& lf, int x0, int y0, int l, int32_t* pred) const;
    // 辞書 (§10): ブロック一辺 s 用の項目リストと、リスト内 k 番目の項目を s x s に整数双線形で再標本化したもの
    int dict_count(int s) const;  // 大域辞書 + タイル内学習原子の数
    const std::vector<int32_t>& dict_atom(int k, int s) const;
    static constexpr int kAtomShift = 3;
    // 平行移動した原子の値 (範囲外は 0)
    static int32_t atom_at(const std::vector<int32_t>& d, int s, int x, int y, int dx, int dy) {
        const int sx = x - dx, sy = y - dy;
        return (sx < 0 || sy < 0 || sx >= s || sy >= s) ? 0 : d[sy * s + sx];
    }
    // 残差の辞書原子: 予測 pred に対する残差に最も合う原子 (リスト番号, 利得) を探す。見つからなければ false
    bool dict_best_atom(int x0, int y0, int s, const int32_t* pred, int& k, int& g) const;
    mutable std::unordered_map<uint64_t, std::vector<int32_t>> atom_cache_;
    // フレーム内学習原子 (タイルごと): CTU を符号化するたびに再構成の高分散 16x16 / 8x8 を平均除去して登録する。
    // 規則は再構成だけで決まるので構文は不要 (符号器・復号器で同一)。番号 serial は再標本化キャッシュ用
    struct LocalAtom { int n; uint64_t serial; std::vector<int32_t> d; int last_used = 0; };  // last_used: 符号器の管理用 (根番号)
    std::vector<LocalAtom> local_[2];  // [0]: 8x8, [1]: 16x16
    uint64_t local_serial_ = 0;
    static constexpr size_t kLocalCap = 256;
    void learn_atoms(int cx, int cy, int ctu);
    // 根ブロック単位の辞書使用フラグ: 0 なら配下の葉は辞書関連のフラグを持たない (未使用時のフラグ代を省く)
    bool dict_on_ = true;
    bool root_uses_dict(int x0, int y0, int l);
    int root_no_ = 0;                 // 符号化した根ブロックの番号
    std::vector<int> used_global_;
    void mark_atom_used(int k, int s); // 符号器: リスト番号 k (サイズ s) の原子を使用済みにする
    void code_discard(SymIO& io, Models& md);  // 根ブロックごとの「捨てる」指示 (いつでも立てられる)

    void obmc(const MotionInfo& cur, int x0, int y0, int s, int32_t* pred) const;
    // 線の動き場: 輝度位置 (x, y) の動き (L0 先頭参照)
    MotionInfo line_mi(int x, int y) const {
        MotionInfo m;
        m.dir = 1;
        const int bx = std::clamp(x / 4, 0, inter_.lf_w4 - 1), by = std::clamp(y / 4, 0, inter_.lf_h4 - 1);
        const size_t i = (static_cast<size_t>(by) * inter_.lf_w4 + bx) * 2;
        m.mvx[0] = (*inter_.lfield)[i];
        m.mvy[0] = (*inter_.lfield)[i + 1];
        return m;
    }
    bool line_ok() const { return tools_.line_mv && inter_.lfield && is_luma(); }
    // 超解像: 1/2 解像度の残差 rl (h x h) を予測 pred (s x s) に導かれた補間で s x s に戻す
    void sr_upsample(int mode, const double* rl, const int32_t* pred, int s, double* r) const;
    // Adam による動き推定: 開始点 (sx, sy) [画素] から連続 MV を最適化し、整数画素に丸めて返す
    void adam_motion(const Plane& rp, int x0, int y0, int w, int h, double sx, double sy, int& ox, int& oy) const;
    void fir_apply(const int* psi, int s, int32_t* pred) const;
    bool fir_fit(const Leaf& lf, int x0, int y0, int s, int* psi) const;
    void cg_apply(const int* cg, int s, int32_t* pred) const;
    bool cg_fit(const Leaf& lf, int x0, int y0, int s, int* cg) const;
    void intra_angular(int x0, int y0, int s, int mode, int32_t* pred) const;
    void intra_safe(int s, int mode, std::vector<int32_t> top, std::vector<int32_t> left, int32_t corner, int32_t* pred) const;
    bool coded_before(int px, int py, int x0, int y0) const;
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
    void rdoq(Leaf& lf, int l, const std::vector<double>& e, uint32_t xb, uint32_t cb) const;
    uint32_t xb_ctx(int x0, int y0, int s) const;
    const Plane* xband_ = nullptr;
    const Plane* tband_ = nullptr;
    const Plane* cband_ = nullptr;
    uint64_t noise_seed_ = 0;
    int vw_ = 1 << 30, vh_ = 1 << 30;  // 有効領域 (これより外のブロックは符号化しない)
public:
    void set_valid(int vw, int vh) { vw_ = vw; vh_ = vh; }
    // 符号器: このフレームで使われた大域辞書の項目 ID (捨てる判断用)
    const std::vector<int>& used_global() const { return used_global_; }
    // 符号化しない (画面外の) ブロックの動き情報は既定値 (イントラ扱い) に戻す。符号器の試行符号化の残りが
    // 色差の動き導出や時間方向候補に漏れて復号器とずれないように、符号器・復号器で同じく行う
    void clear_outside_mf(int x0, int y0, int s) {
        if (!inter_.enabled || !is_luma() || !inter_.mf) return;
        MotionField& mf = *inter_.mf;
        for (int y = y0 / 4; y < std::min((y0 + s) / 4, mf.h4); ++y)
            for (int x = x0 / 4; x < std::min((x0 + s) / 4, mf.w4); ++x) mf.mi[static_cast<size_t>(y) * mf.w4 + x] = MotionInfo{};
    }
    // 有効領域の外を端の画素の複製で埋める (ループフィルタ等が参照する前に、符号器・復号器で同じく行う)
    static void fill_outside(Plane& R, int vw, int vh) {
        for (int y = 0; y < R.h; ++y)
            for (int x = 0; x < R.w; ++x)
                if (x >= vw || y >= vh) R.at(x, y) = R.at(std::min(x, vw - 1), std::min(y, vh - 1));
    }
private:
    GsMem gs_mem_;  // 利得形状ブロック間の利得予測 (§5.3 連結)
    uint64_t gs_seed(int x0, int y0) const {
        return (noise_seed_ + 0x632BE59BD9B4E019ull) ^ (static_cast<uint64_t>(x0) << 32 | static_cast<uint64_t>(y0) << 4 | static_cast<uint64_t>(plane_));
    }
    uint32_t cb_ctx(int x0, int y0, int s) const;
    bool ns_allowed(const Leaf& lf) const {
        return (tools_.band_ns && lf.pt == 4 && lf.xsrc == 1) || (tools_.inter_ns && lf.pt == 2 && plane_ == 0);
    }
    // coefs: 同じ (予測, 変換) の順変換係数を再利用する場合に与える (ラスタ順)。nullptr なら計算する
    void quantize(Leaf& lf, int x0, int y0, int l, const int32_t* pred, const std::vector<double>* coefs = nullptr) const;
    void fwd_coefs(int mts, int x0, int y0, int l, const int32_t* pred, std::vector<double>& c) const;
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
