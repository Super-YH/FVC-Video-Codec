#include "fvc/codec.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <atomic>
#include <string>
#include <thread>
#include <stdexcept>

#include "fvc/entropy.hpp"
#include "fvc/pqmf.hpp"
#include "fvc/transform.hpp"
#include "block_coder.hpp"
#include "dictionary.hpp"
#include "inter.hpp"
#include "alf.hpp"
#include "cdef.hpp"
#include "loop_filter.hpp"
#include "shapes.hpp"
#include "fvc/quant.hpp"

namespace fvc {

namespace {

constexpr int kCtuLog2 = 6;  // 64x64 CTU (段階2)
constexpr int kCtu = 1 << kCtuLog2;
constexpr uint32_t kMagic = 0x46564331;  // 'FVC1'

// ---------------- バイト列ユーティリティ ----------------
void put_u8(std::vector<uint8_t>& o, uint32_t v) { o.push_back(static_cast<uint8_t>(v)); }
void put_u16(std::vector<uint8_t>& o, uint32_t v) { put_u8(o, v >> 8); put_u8(o, v); }
void put_u32(std::vector<uint8_t>& o, uint32_t v) { put_u16(o, v >> 16); put_u16(o, v); }
// 可変長整数 (LEB128: 下位 7 ビットずつ、最上位ビット = 継続)
void put_uv(std::vector<uint8_t>& o, uint32_t v) {
    while (v >= 0x80) { o.push_back(static_cast<uint8_t>(v | 0x80)); v >>= 7; }
    o.push_back(static_cast<uint8_t>(v));
}

struct ByteReader {
    const uint8_t* d; size_t n, pos = 0;
    bool fail = false;
    uint32_t u8() { if (pos >= n) { fail = true; return 0; } return d[pos++]; }
    uint32_t u16() { uint32_t a = u8(); return (a << 8) | u8(); }
    uint32_t u32() { uint32_t a = u16(); return (a << 16) | u16(); }
    uint32_t uv() {
        uint32_t v = 0;
        for (int sh = 0; sh < 35; sh += 7) {
            const uint32_t b = u8();
            v |= (b & 0x7f) << sh;
            if (!(b & 0x80)) return v;
        }
        fail = true;
        return 0;
    }
};

void put_unit(std::vector<uint8_t>& out, UnitType t, const std::vector<uint8_t>& payload) {
    put_u8(out, static_cast<uint8_t>(t));
    put_u32(out, static_cast<uint32_t>(payload.size()));
    out.insert(out.end(), payload.begin(), payload.end());
}

// PQMF の構成 (§3.2): 帯域数 2^lx x 2^ly とプロトタイプフィルタ (遅延 m, Kaiser β) の選択
struct PqmfCfg { int lx = 0, ly = 0, filter = 0; bool ll_split = false; };  // ll_split: LL 帯域をさらに 2x2 分解 (§3.2 部分木)
struct PqmfFilter { int m; double beta; };
constexpr PqmfFilter kPqmfFilters[4] = {{6, 9.0}, {4, 7.0}, {8, 10.0}, {3, 5.0}};

struct FrameParams;
PqmfCfg pqmf_cfg(const FrameParams& fp, bool ll_split = false);

// ---------------- PQMF 帯域符号化 (§3, 帯域間差分 §3.3, ノイズ置換 §5.6) ----------------
// band_mode: 0 = 通常 (量子化値を符号化), 1 = 帯域間差分 (左/上の隣接帯域を鏡像参照, ゲイン a/64), 2 = ノイズ置換 (RMS のみ)
void code_bands(SymIO& io, Models& md, const PqmfCfg& pq, const Plane* org, Plane& rec, double step, double rnd,
                int32_t lo, int32_t hi, bool band_tools, bool psy, uint64_t seed) {
    const int Mx = 1 << pq.lx, My = 1 << pq.ly, W = rec.w, H = rec.h;
    Pqmf2D fb(Mx, My, kPqmfFilters[pq.filter].m, kPqmfFilters[pq.filter].beta);
    const int32_t off = (lo + hi + 1) / 2;
    std::vector<std::vector<double>> bands, src;
    if (io.enc) {
        std::vector<double> x(static_cast<size_t>(W) * H);
        for (size_t i = 0; i < x.size(); ++i) x[i] = org->v[i] - off;
        fb.analyze(x, W, H, bands);
        src = bands;
    } else {
        bands.assign(Mx * My, std::vector<double>(static_cast<size_t>(W / Mx) * (H / My), 0.0));
    }
    const int bw = W / Mx, bh = H / My;
    constexpr int kPB = 8;  // 帯域間予測のブロック (帯域画像の画素)
    std::vector<std::vector<int32_t>> qs(Mx * My, std::vector<int32_t>(static_cast<size_t>(bw) * bh, 0));
    double ll_step = 1.0;
    for (int ky = 0; ky < My; ++ky)
        for (int kx = 0; kx < Mx; ++kx) {
            const int k = ky * Mx + kx;
            auto& b = bands[k];
            auto& q = qs[k];
            const double st = step / fb.band_norm(kx, ky) * (1.0 + 0.08 * (kx + ky));
            const bool ll = kx == 0 && ky == 0;
            if (ll) ll_step = st;
            // 参照帯域 (再構成済み): 左隣 or 上隣。奇数帯域のスペクトル鏡像は (-1)^n 変調で補正
            const int rk = kx > 0 ? k - 1 : (ky > 0 ? k - Mx : -1);
            const bool horiz = kx > 0;
            auto refv = [&](int x, int y) {
                const double v = bands[rk][static_cast<size_t>(y) * bw + x];
                return ((horiz ? x : y) & 1) ? -v : v;
            };
            // 帯域モード: 0 = 通常 (ブロック単位の帯域間予測つき), 2 = ノイズ置換
            int mode = 0, rms_q = 0;
            if (io.enc && band_tools && !ll && psy) {
                double e = 0;
                for (double v : src[k]) e += v * v;
                const double rms = std::sqrt(e / src[k].size());
                if (rms < 0.5 * st && rms > 0.15 * st) { mode = 2; rms_q = std::clamp(static_cast<int>(std::lround(rms / st * 32.0)), 1, 63); }
            }
            if (band_tools && !ll) {
                mode = static_cast<int>(io.uint(md.band_mode, 0, static_cast<uint32_t>(std::min(kx + ky, 15)), static_cast<uint32_t>(mode)));
                if (mode != 0 && mode != 2) throw std::runtime_error("corrupt stream: band mode");
                if (mode == 2) rms_q = static_cast<int>(io.uint(md.band_mode, 2, 0, static_cast<uint32_t>(rms_q)));
                if (rms_q > 63) throw std::runtime_error("corrupt stream: band params");
            }
            if (mode == 2) {
                SplitMix64 rng(seed * 1315423911ull + static_cast<uint64_t>(k));
                const double amp = rms_q / 32.0 * st;
                for (auto& v : b) v = (rng.next() >> 63) ? amp : -amp;
                continue;
            }
            // ブロック単位の帯域間予測ゲイン g ∈ {-4..4}/4 (常時有効、g=0 は予測なし)
            const int nbx = (bw + kPB - 1) / kPB, nby = (bh + kPB - 1) / kPB;
            std::vector<int> gain(static_cast<size_t>(nbx) * nby, 0);
            std::vector<double> pr(b.size(), 0.0);
            if (!ll && rk >= 0) {
                int prevg = 0;
                for (int by = 0; by < nby; ++by)
                    for (int bx = 0; bx < nbx; ++bx) {
                        int g = 0;
                        if (io.enc) {
                            double num = 0, den = 0, e0 = 0;
                            for (int y = by * kPB; y < std::min(bh, (by + 1) * kPB); ++y)
                                for (int x = bx * kPB; x < std::min(bw, (bx + 1) * kPB); ++x) {
                                    const double r = refv(x, y), t = src[k][static_cast<size_t>(y) * bw + x];
                                    num += r * t; den += r * r; e0 += t * t;
                                }
                            if (den > 0) {
                                const int gq = std::clamp(static_cast<int>(std::lround(num / den * 4.0)), -4, 4);
                                const double a = gq / 4.0;
                                const double e1 = e0 - 2 * a * num + a * a * den;
                                // 予測で残差エネルギーが量子化ステップ相当以上減る場合のみ (副情報 ~3 ビット)
                                if (gq != 0 && e0 - e1 > 3.0 * st * st) g = gq;
                            }
                        }
                        g = prevg + io.sint(md.band_mode, 3, static_cast<uint32_t>(std::min(kx + ky, 15)), g - prevg);
                        if (g < -4 || g > 4) throw std::runtime_error("corrupt stream: band gain");
                        gain[by * nbx + bx] = g;
                        prevg = g;
                        for (int y = by * kPB; y < std::min(bh, (by + 1) * kPB); ++y)
                            for (int x = bx * kPB; x < std::min(bw, (bx + 1) * kPB); ++x)
                                pr[static_cast<size_t>(y) * bw + x] = g / 4.0 * refv(x, y);
                    }
            }
            // 高域ほどデッドゾーンを広げる
            const double r_band = std::max(0.18, rnd - 0.02 * (kx + ky));
            const auto& LL = qs[0];
            for (int y = 0; y < bh; ++y)
                for (int x = 0; x < bw; ++x) {
                    const size_t i = static_cast<size_t>(y) * bw + x;
                    const int32_t qw = x ? q[i - 1] : 0, qn = y ? q[i - bw] : 0, qnw = (x && y) ? q[i - bw - 1] : 0;
                    const int32_t qne = (y && x + 1 < bw) ? q[i - bw + 1] : 0;
                    if (ll) {
                        int32_t p;
                        if (!x && !y) p = 0; else if (!y) p = qw; else if (!x) p = qn;
                        else { const int32_t mx = std::max(qw, qn), mn = std::min(qw, qn); p = qnw >= mx ? mn : qnw <= mn ? mx : qw + qn - qnw; }
                        const uint32_t act = static_cast<uint32_t>(std::min(std::abs(qw - qnw) + std::abs(qn - qnw), 31));
                        const int32_t d = io.sint(md.band_ll, act, 0, io.enc ? quant_dz(src[k][i], st, 0.5) - p : 0);
                        q[i] = p + d;
                    } else {
                        // 文脈 (帯域間相関の常時利用): 空間近傍 + 隣接帯域の同位置 + LL 帯域の局所勾配
                        int cross = 0;
                        if (kx > 0) cross += std::abs(qs[k - 1][i]);
                        if (ky > 0) cross += std::abs(qs[k - Mx][i]);
                        if (kx > 0 && ky > 0) cross += std::abs(qs[k - Mx - 1][i]);
                        const int xl = std::max(x - 1, 0), xr = std::min(x + 1, bw - 1), yu = std::max(y - 1, 0), yd = std::min(y + 1, bh - 1);
                        const int32_t grad = std::abs(LL[static_cast<size_t>(y) * bw + xr] - LL[static_cast<size_t>(y) * bw + xl]) +
                                             std::abs(LL[static_cast<size_t>(yd) * bw + x] - LL[static_cast<size_t>(yu) * bw + x]);
                        const double gs = grad * ll_step / st;  // 当該帯域のステップ単位
                        const int gb = gs < 0.5 ? 0 : gs < 2 ? 1 : gs < 6 ? 2 : gs < 16 ? 3 : 4;
                        const uint32_t a = static_cast<uint32_t>(std::min(std::abs(qw) + std::abs(qn) + std::abs(qnw) + std::abs(qne), 15)) |
                                           (static_cast<uint32_t>(std::min(cross, 7)) << 4);
                        const uint32_t bctx = static_cast<uint32_t>(std::min(kx + ky, 15)) | (static_cast<uint32_t>(gb) << 4) |
                                              (pr[i] != 0.0 ? 128u : 0u);
                        q[i] = io.sint(md.band_hi, a, bctx, io.enc ? quant_dz(src[k][i] - pr[i], st, r_band) : 0);
                    }
                    b[i] = pr[i] + q[i] * st;
                }
        }
    std::vector<double> y;
    fb.synthesize(bands, W, H, y);
    for (size_t i = 0; i < y.size(); ++i) rec.v[i] = std::clamp(static_cast<int32_t>(std::lround(y[i])) + off, lo, hi);
}

// ---------------- 動画の帯域符号化 (§3, §7.5) ----------------
// 予測画像 P の帯域 P_k を時間方向予測として、帯域ブロック (8x8 帯域標本) ごとに
//   pred = g_t · P_k + g_c · T_s(R_j - P_j)     (R_j: 現フレームで符号化済みの隣接帯域 j の再構成)
// を探索する。g_t ∈ {0, .5, .75, 1, 1.25}, g_c ∈ {-4..4}/4, T_s は鏡像 + シフト (±1 帯域標本, 9 通り)。
// 残差は RD でスキップ (係数なし) または帯域間文脈つきで符号化する。
void code_bands_video(SymIO& io, Models& md, const PqmfCfg& pq, const Plane* org, const Plane& P, Plane& rec, double step,
                      double lambda, int32_t lo, int32_t hi) {
    const int Mx = 1 << pq.lx, My = 1 << pq.ly, W = rec.w, H = rec.h, bw = W / Mx, bh = H / My;
    Pqmf2D fb(Mx, My, kPqmfFilters[pq.filter].m, kPqmfFilters[pq.filter].beta);
    const int32_t off = (lo + hi + 1) / 2;
    std::vector<std::vector<double>> pb, xb;
    {
        std::vector<double> t(static_cast<size_t>(W) * H);
        for (size_t i = 0; i < t.size(); ++i) t[i] = P.v[i] - off;
        fb.analyze(t, W, H, pb);
        if (io.enc) {
            for (size_t i = 0; i < t.size(); ++i) t[i] = org->v[i] - off;
            fb.analyze(t, W, H, xb);
        }
    }
    std::vector<std::vector<double>> rb(Mx * My, std::vector<double>(static_cast<size_t>(bw) * bh, 0.0));
    std::vector<std::vector<int32_t>> qs(Mx * My, std::vector<int32_t>(static_cast<size_t>(bw) * bh, 0));
    constexpr int kB = 8;
    constexpr double kGt[5] = {0.0, 0.5, 0.75, 1.0, 1.25};
    const int nbx = (bw + kB - 1) / kB, nby = (bh + kB - 1) / kB;
    for (int ky = 0; ky < My; ++ky)
        for (int kx = 0; kx < Mx; ++kx) {
            const int k = ky * Mx + kx;
            const double nrm = fb.band_norm(kx, ky);
            const double st = step / nrm * (1.0 + 0.08 * (kx + ky));
            const double lam = lambda / (nrm * nrm);  // 帯域領域の SSE と画素領域の SSE の換算
            const int rk = kx > 0 ? k - 1 : (ky > 0 ? k - Mx : -1);
            const bool horiz = kx > 0;
            // 参照帯域の再構成残差 (鏡像補正つき)、シフト s ∈ [0,9)
            auto cref = [&](int x, int y, int sh) {
                const int xx = std::clamp(x + (sh % 3) - 1, 0, bw - 1), yy = std::clamp(y + (sh / 3) - 1, 0, bh - 1);
                const size_t i = static_cast<size_t>(yy) * bw + xx;
                const double v = rb[rk][i] - pb[rk][i];
                return ((horiz ? xx : yy) & 1) ? -v : v;
            };
            const double rnd = std::max(0.18, 1.0 / 3.0 - 0.02 * (kx + ky));
            int prev_gt = 3, prev_gc = 0;
            for (int by = 0; by < nby; ++by)
                for (int bx = 0; bx < nbx; ++bx) {
                    const int x0 = bx * kB, y0 = by * kB, x1 = std::min(bw, x0 + kB), y1 = std::min(bh, y0 + kB);
                    int gt = 3, gc = 0, sh = 4;
                    std::vector<double> pr(static_cast<size_t>(kB) * kB, 0.0);
                    auto build = [&](int g1, int g2, int s2) {
                        for (int y = y0; y < y1; ++y)
                            for (int x = x0; x < x1; ++x) {
                                double v = kGt[g1] * pb[k][static_cast<size_t>(y) * bw + x];
                                if (g2 && rk >= 0) v += g2 / 4.0 * cref(x, y, s2);
                                pr[(y - y0) * kB + (x - x0)] = v;
                            }
                    };
                    if (io.enc) {
                        // 予測パラメータの探索 (残差エネルギー + λ·副情報)
                        double best = 1e300;
                        for (int g1 = 0; g1 < 5; ++g1)
                            for (int s2 = 0; s2 < (rk >= 0 ? 9 : 1); ++s2)
                                for (int g2 = (rk >= 0 ? -4 : 0); g2 <= (rk >= 0 ? 4 : 0); ++g2) {
                                    if (g2 == 0 && s2 != 4) continue;
                                    build(g1, g2, s2);
                                    double e = 0;
                                    for (int y = y0; y < y1; ++y)
                                        for (int x = x0; x < x1; ++x) {
                                            const double d = xb[k][static_cast<size_t>(y) * bw + x] - pr[(y - y0) * kB + (x - x0)];
                                            e += d * d;
                                        }
                                    const double side = (g1 == prev_gt ? 1.0 : 3.0) + (g2 == prev_gc ? 1.0 : 4.0) + (g2 ? 3.0 : 0.0);
                                    const double j = e + lam * side;
                                    if (j < best) { best = j; gt = g1; gc = g2; sh = s2; }
                                }
                    }
                    const uint32_t bc = static_cast<uint32_t>(std::min(kx + ky, 15));
                    gt = static_cast<int>(io.uint(md.band_mode, 4, bc | (static_cast<uint32_t>(prev_gt) << 4), static_cast<uint32_t>(gt)));
                    if (gt > 4) throw std::runtime_error("corrupt stream: band gt");
                    if (rk >= 0) {
                        gc = prev_gc + io.sint(md.band_mode, 5, bc, gc - prev_gc);
                        if (gc < -4 || gc > 4) throw std::runtime_error("corrupt stream: band gc");
                        if (gc) {
                            sh = static_cast<int>(io.uint(md.band_mode, 6, bc, static_cast<uint32_t>(sh)));
                            if (sh > 8) throw std::runtime_error("corrupt stream: band shift");
                        } else {
                            sh = 4;
                        }
                    } else {
                        gc = 0;
                    }
                    prev_gt = gt; prev_gc = gc;
                    build(gt, gc, sh);
                    // 残差の量子化 (符号器) とスキップ判定
                    std::vector<int32_t> q(static_cast<size_t>(kB) * kB, 0);
                    int skip = 1;
                    if (io.enc) {
                        double e_skip = 0, e_code = 0, bits = 2.0;
                        for (int y = y0; y < y1; ++y)
                            for (int x = x0; x < x1; ++x) {
                                const int li = (y - y0) * kB + (x - x0);
                                const double t = xb[k][static_cast<size_t>(y) * bw + x] - pr[li];
                                q[li] = quant_dz(t, st, rnd);
                                const double d = t - q[li] * st;
                                e_skip += t * t;
                                e_code += d * d;
                                bits += q[li] ? 2.0 + 2.0 * std::log2(1.0 + std::abs(q[li])) : 0.4;
                            }
                        bool any = false;
                        for (int32_t v : q) any |= v != 0;
                        skip = !any || e_skip <= e_code + lam * bits;
                    }
                    skip = io.bit(md.band_mode, 7, bc, static_cast<uint32_t>(gt), skip);
                    for (int y = y0; y < y1; ++y)
                        for (int x = x0; x < x1; ++x) {
                            const size_t i = static_cast<size_t>(y) * bw + x;
                            const int li = (y - y0) * kB + (x - x0);
                            int32_t v = 0;
                            if (!skip) {
                                const auto& qk = qs[k];
                                const int32_t qw = x ? qk[i - 1] : 0, qn = y ? qk[i - bw] : 0, qnw = (x && y) ? qk[i - bw - 1] : 0;
                                const int32_t qne = (y && x + 1 < bw) ? qk[i - bw + 1] : 0;
                                int cross = 0;
                                if (kx > 0) cross += std::abs(qs[k - 1][i]);
                                if (ky > 0) cross += std::abs(qs[k - Mx][i]);
                                const uint32_t a = static_cast<uint32_t>(std::min(std::abs(qw) + std::abs(qn) + std::abs(qnw) + std::abs(qne), 15)) |
                                                   (static_cast<uint32_t>(std::min(cross, 7)) << 4);
                                v = io.sint(md.band_hi, a, bc | 64u, io.enc ? q[li] : 0);
                            }
                            qs[k][i] = v;
                            rb[k][i] = pr[li] + v * st;
                        }
                }
        }
    std::vector<double> y;
    fb.synthesize(rb, W, H, y);
    for (size_t i = 0; i < y.size(); ++i) rec.v[i] = std::clamp(static_cast<int32_t>(std::lround(y[i])) + off, lo, hi);
}

// ---------------- 帯域画像のブロック符号化 (§3 + §5) ----------------
// 各帯域画像を合成利得で正規化した整数プレーンにし、既存のブロック符号化器
// (分割・イントラ予測・変換・RDOQ・CM) で符号化する。P != nullptr (P/B) のときは
// 帯域ごとの残差 X_k - P_k を符号化する。帯域ごとに量子化ステップを高域ほど粗くする。
void code_bands_blocks(SymIO& io, Models& md, const PqmfCfg& pq, const Plane* org, const Plane* P, Plane& rec, double step,
                       double lambda, int32_t lo, int32_t hi, int plane, const Tools& tools, const Search& search,
                       int min_log2, int max_log2, bool band_ns, uint64_t seed) {
    const int Mx = 1 << pq.lx, My = 1 << pq.ly, W = rec.w, H = rec.h, bw = W / Mx, bh = H / My;
    const int BW = (bw + kCtu - 1) / kCtu * kCtu, BH = (bh + kCtu - 1) / kCtu * kCtu;
    Pqmf2D fb(Mx, My, kPqmfFilters[pq.filter].m, kPqmfFilters[pq.filter].beta);
    const int32_t off = (lo + hi + 1) / 2;
    std::vector<std::vector<double>> pb, xb;
    {
        std::vector<double> t(static_cast<size_t>(W) * H);
        if (P) {
            for (size_t i = 0; i < t.size(); ++i) t[i] = P->v[i] - off;
            fb.analyze(t, W, H, pb);
        }
        if (io.enc) {
            for (size_t i = 0; i < t.size(); ++i) t[i] = org->v[i] - off;
            fb.analyze(t, W, H, xb);
        }
    }
    std::vector<std::vector<double>> rb(Mx * My, std::vector<double>(static_cast<size_t>(bw) * bh, 0.0));
    Tools bt = tools;
    bt.cfl = false; bt.ibc = false; bt.dict = false; bt.pred_only = false; bt.rect = false; bt.tmvp = false;
    Search bs = search;
    const int32_t range = 8 * (hi - lo + 1) * std::max(Mx, My);  // 帯域値の取りうる範囲 (正規化後) に十分な余裕
    std::vector<Plane> rk_plane(Mx * My);           // 各帯域の再構成 (正規化整数, パディング込み)
    Plane tp0, chg;                               // LL 帯域の時間方向予測と変化マスク
    for (int ky = 0; ky < My; ++ky)
        for (int kx = 0; kx < Mx; ++kx) {
            const int k = ky * Mx + kx;
            const double nrm = fb.band_norm(kx, ky);
            constexpr double kBw = 0.08;  // 実測: 0/0.08/0.2/0.4 で同等、0.08 が僅かに最良
            const double bstep = step * (1.0 + kBw * (kx + ky));
            Plane target, R(BW, BH, 0);
            if (io.enc) {
                target = Plane(BW, BH);
                for (int y = 0; y < BH; ++y)
                    for (int x = 0; x < BW; ++x) {
                        const size_t i = static_cast<size_t>(std::min(y, bh - 1)) * bw + std::min(x, bw - 1);
                        const double v = xb[k][i];  // P/B でも入力帯域そのもの (時間方向予測はモードで選ぶ)
                        target.at(x, y) = static_cast<int32_t>(std::lround(v * nrm));
                    }
            }
            bt.band_ns = band_ns && P && (kx + ky) >= 2;
            BlockCoder bc(&R, io.enc ? &target : nullptr, nullptr, plane, -range, range, bstep, lambda,
                          std::min(min_log2, 3), max_log2, bt, bs);
            // 帯域間相関: 左 (なければ上) の符号化済み帯域を鏡像補正して文脈・予測に使う
            Plane xp;
            const int rk = kx > 0 ? k - 1 : (ky > 0 ? k - Mx : -1);
            if (rk >= 0) {
                xp = rk_plane[rk];
                const bool horiz = kx > 0;
                for (int y = 0; y < BH; ++y)
                    for (int x = 0; x < BW; ++x)
                        if ((horiz ? x : y) & 1) xp.at(x, y) = -xp.at(x, y);
                bc.set_xband(&xp);
            }
            Plane tp;
            if (P) {
                tp = Plane(BW, BH);
                for (int y = 0; y < BH; ++y)
                    for (int x = 0; x < BW; ++x)
                        tp.at(x, y) = static_cast<int32_t>(std::lround(pb[k][static_cast<size_t>(std::min(y, bh - 1)) * bw + std::min(x, bw - 1)] * nrm));
                bc.set_tband(&tp);
                if (k > 0 && chg.w) bc.set_cband(&chg);
            }
            bc.set_noise_seed(seed * 1315423911ull + static_cast<uint64_t>(k));
            if (k == 0 && pq.ll_split && bw % 2 == 0 && bh % 2 == 0) {
                // 部分木分解 (§3.2 band_split): LL 帯域を同じ方式で 2x2 帯域に再分解して符号化する
                Plane llo, llp, llr(bw, bh, 0);
                if (io.enc) {
                    llo = Plane(bw, bh);
                    for (int y = 0; y < bh; ++y) for (int x = 0; x < bw; ++x) llo.at(x, y) = target.at(x, y);
                }
                if (P) {
                    llp = Plane(bw, bh);
                    for (int y = 0; y < bh; ++y) for (int x = 0; x < bw; ++x) llp.at(x, y) = tp.at(x, y);
                }
                PqmfCfg sub;
                sub.lx = sub.ly = 1;
                sub.filter = pq.filter;
                code_bands_blocks(io, md, sub, io.enc ? &llo : nullptr, P ? &llp : nullptr, llr, bstep, lambda, -range, range, plane,
                                  tools, search, min_log2, max_log2, false, seed * 31 + 7);
                for (int y = 0; y < BH; ++y)
                    for (int x = 0; x < BW; ++x) R.at(x, y) = llr.at(std::min(x, bw - 1), std::min(y, bh - 1));
            } else {
                for (int cy = 0; cy < BH; cy += kCtu)
                    for (int cx = 0; cx < BW; cx += kCtu) bc.code_ctu(io, md, cx, cy, kCtu);
            }
            for (int y = 0; y < bh; ++y)
                for (int x = 0; x < bw; ++x) {
                    const size_t i = static_cast<size_t>(y) * bw + x;
                    rb[k][i] = R.at(x, y) / nrm;
                }
            if (k == 0 && P) {
                // 変化マスク: |LL 再構成 - LL 時間方向予測| (動いた場所は全帯域で共通)
                chg = Plane(BW, BH);
                for (size_t i = 0; i < chg.v.size(); ++i) chg.v[i] = std::abs(R.v[i] - tp.v[i]);
            }
            rk_plane[k] = std::move(R);
        }
    std::vector<double> y;
    fb.synthesize(rb, W, H, y);
    for (size_t i = 0; i < y.size(); ++i) rec.v[i] = std::clamp(static_cast<int32_t>(std::lround(y[i])) + off, lo, hi);
}

// ---------------- ロスレス層 ----------------
// lossy=false: 原画を MED 予測で直接符号化。lossy=true: e = X - R (L2 残差) を符号化。
void code_lossless(SymIO& io, Models& md, int plane, int w, int h, const Plane* org, Plane& rec, bool lossy, int32_t mid) {
    Plane e(w, h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            if (!lossy) {
                const int32_t W_ = x ? rec.at(x - 1, y) : (y ? rec.at(x, y - 1) : mid);
                const int32_t N_ = y ? rec.at(x, y - 1) : W_;
                const int32_t NW = (x && y) ? rec.at(x - 1, y - 1) : N_;
                const int32_t NE = (y && x + 1 < w) ? rec.at(x + 1, y - 1) : N_;
                const int32_t mx = std::max(W_, N_), mn = std::min(W_, N_);
                const int32_t p = NW >= mx ? mn : NW <= mn ? mx : W_ + N_ - NW;
                const int32_t act = std::abs(W_ - NW) + std::abs(N_ - NW) + std::abs(NE - N_);
                int ab = 0;
                while (ab < 12 && (1 << ab) <= act) ++ab;
                const int32_t eW = x ? e.at(x - 1, y) : 0, eN = y ? e.at(x, y - 1) : 0;
                const uint32_t a = static_cast<uint32_t>(ab) | (static_cast<uint32_t>(std::min(std::abs(eW) + std::abs(eN), 15)) << 4);
                const int32_t d = io.sint(md.lossless, a, static_cast<uint32_t>(plane), io.w ? org->at(x, y) - p : 0);
                e.at(x, y) = d;
                rec.at(x, y) = p + d;
            } else {
                const int32_t eW = x ? e.at(x - 1, y) : 0, eN = y ? e.at(x, y - 1) : 0;
                const int32_t eNW = (x && y) ? e.at(x - 1, y - 1) : 0;
                const int32_t gx = x ? std::abs(rec.at(x, y) - rec.at(x - 1, y)) : 0;
                const int32_t gy = y ? std::abs(rec.at(x, y) - rec.at(x, y - 1)) : 0;
                int gb = 0;
                while (gb < 7 && (2 << gb) <= gx + gy) ++gb;
                const uint32_t a = static_cast<uint32_t>(std::min(std::abs(eW) + std::abs(eN) + std::abs(eNW), 31)) | (static_cast<uint32_t>(gb) << 5);
                const int32_t d = io.sint(md.l2, a, static_cast<uint32_t>(plane), io.w ? org->at(x, y) - rec.at(x, y) : 0);
                e.at(x, y) = d;
                rec.at(x, y) += d;
            }
        }
}

Plane pad_plane(const Plane& p, int W, int H) {
    Plane o(W, H);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) o.at(x, y) = p.at(std::min(x, p.w - 1), std::min(y, p.h - 1));
    return o;
}

// 色差の CfL 用: 輝度再構成を色差解像度に縮小 (パディング込み)
Plane luma_at_chroma(const Plane& luma, const VideoInfo& info, int W, int H) {
    Plane o(W, H);
    const bool sub = info.chroma == ChromaFormat::C420;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            if (!sub) { o.at(x, y) = luma.at(std::min(x, luma.w - 1), std::min(y, luma.h - 1)); continue; }
            const int x0 = std::min(2 * x, luma.w - 1), x1 = std::min(2 * x + 1, luma.w - 1);
            const int y0 = std::min(2 * y, luma.h - 1), y1 = std::min(2 * y + 1, luma.h - 1);
            o.at(x, y) = (luma.at(x0, y0) + luma.at(x1, y0) + luma.at(x0, y1) + luma.at(x1, y1) + 2) >> 2;
        }
    return o;
}

struct FrameParams {
    FrameType type = FrameType::I;
    int poc = 0;
    int qp = 32;
    bool lossy = true, l2 = false;
    int pqmf_log2 = 0;
    int pqmf_log2y = 0;  // 縦の帯域数 (log2)。符号器が §3.2 の選択で決める (0: pqmf_log2 と同じ)
    int pqmf_filter = 0; // プロトタイプ (kPqmfFilters の番号)
    int min_log2 = 2, max_log2 = kCtuLog2;
    Tools tools;               // ビットストリームで伝送
    bool shapes = false;       // 図形レイヤ (I のみ)
    bool band_tools = false;   // 帯域間差分/ノイズ置換
    bool dict_reset = false;
    bool dict_add = true;                              // 符号器のみ: 辞書追加 (将来フレーム用) を送るか
    bool lf = true, lf_freq = false, lf_map = false;  // ループフィルタ (§9)
    int tile_cols = 1, tile_rows = 1;                  // タイル分割 (CTU 単位で均等)
    int cqp_off = 0;                                   // 色差 QP オフセット
    bool aqp = false;                                  // CTU 単位の適応 QP
    bool alf = false;                                  // 適応ウィーナーフィルタ (§9.4)
    bool band_blocks = true;                           // 帯域画像をブロック符号化 (false: 標本単位)
    bool band_ns = false;                              // 帯域の時間差分ノイズ置換 (心理視覚)
    bool cdef = false;                                 // 方向性デリンギングフィルタ (§9.5)
    // 符号器: 輝度 64x64 CTU ごとの dQP (先読みによる静止度から決定)
    std::vector<int8_t> aqp_map;
    int aqp_w = 0;
    int threads = 1;                                   // 符号器/復号器のスレッド数 (ビットストリームに影響しない)
    // 符号器: グローバルパラメータのキャッシュ (COPY 判定と本符号化で共有)
    mutable bool gm_valid = false;
    mutable int gm_x = 0, gm_y = 0, gm_gain[3] = {64, 64, 64}, gm_off[3] = {0, 0, 0};
    mutable GlobalModel gm_model;
    mutable int gm_hg[3] = {64, 64, 64};
    std::vector<int> ref_poc[2];
    Search search;             // 符号器のみ
    bool psy = false;          // 符号器のみ
};

uint8_t tools_byte(const Tools& t) {
    return static_cast<uint8_t>((t.ibc ? 1 : 0) | (t.tns ? 2 : 0) | (t.e8 ? 4 : 0) | (t.cfl ? 8 : 0) | (t.nf ? 16 : 0) |
                                (t.all_angular ? 32 : 0) | (t.dict ? 64 : 0) | (t.fir ? 128 : 0));
}
Tools tools_from_byte(uint8_t b) {
    Tools t;
    t.ibc = b & 1; t.tns = b & 2; t.e8 = b & 4; t.cfl = b & 8; t.nf = b & 16; t.all_angular = b & 32;
    t.dict = b & 64; t.fir = b & 128;
    return t;
}

struct Picture {
    int poc = 0;
    Frame f;
    bool anchor = true;
    std::shared_ptr<MotionField> mf;  // 輝度 4x4 動き (時間方向候補用)
    std::vector<int> ref_poc[2];
};

// タイルごとの独立 rANS ストリーム (符号器は出力を蓄積、復号器は順に消費)
struct TileStreams {
    std::vector<std::vector<uint8_t>> out;
    std::vector<std::pair<const uint8_t*, size_t>> in;
    size_t next = 0;
};

}  // namespace

// DPB と辞書 (符号器・復号器で同一規則で更新)
struct CodecState {
    std::deque<Picture> dpb;  // 復号順、最大 kMaxDpb
    Dictionary dict;
    static constexpr size_t kMaxDpb = 12;
    const Frame* find(int poc) const {
        for (const auto& p : dpb) if (p.poc == poc) return &p.f;
        return nullptr;
    }
    const Picture* find_pic(int poc) const {
        for (const auto& p : dpb) if (p.poc == poc) return &p;
        return nullptr;
    }
    // 直前に符号化/復号したフレームの動き (code_frame が設定)
    std::shared_ptr<MotionField> last_mf;
    std::vector<int> last_ref_poc[2];
    std::vector<Shape> last_shapes;  // 直前フレームの図形 (時間方向の図形予測 §7.4)
    void push(int poc, const Frame& f, bool anchor) {
        Picture pic;
        pic.poc = poc; pic.f = f; pic.anchor = anchor;
        pic.mf = last_mf;
        pic.ref_poc[0] = last_ref_poc[0];
        pic.ref_poc[1] = last_ref_poc[1];
        dpb.push_back(std::move(pic));
        while (dpb.size() > kMaxDpb) dpb.pop_front();
    }
};

namespace {

// グローバル動き推定 (符号器): 1/4 縮小で全探索 → 原解像度で ±2 画素精密化。戻り値は 1/4 画素単位。
void estimate_global_motion(const Plane& cur, const Plane& ref, int& gx, int& gy) {
    auto down = [](const Plane& p) {
        Plane o(std::max(1, p.w / 4), std::max(1, p.h / 4));
        for (int y = 0; y < o.h; ++y)
            for (int x = 0; x < o.w; ++x) {
                int s = 0;
                for (int j = 0; j < 4; ++j) for (int i = 0; i < 4; ++i) s += p.at(std::min(4 * x + i, p.w - 1), std::min(4 * y + j, p.h - 1));
                o.at(x, y) = s / 16;
            }
        return o;
    };
    const Plane c = down(cur), r = down(ref);
    auto sad = [](const Plane& a, const Plane& b, int dx, int dy, int stride) {
        int64_t s = 0, n = 0;
        for (int y = 0; y < a.h; y += stride)
            for (int x = 0; x < a.w; x += stride) {
                const int rx = x + dx, ry = y + dy;
                if (rx < 0 || ry < 0 || rx >= b.w || ry >= b.h) continue;
                s += std::abs(a.at(x, y) - b.at(rx, ry));
                ++n;
            }
        return n ? static_cast<double>(s) / n : 1e30;
    };
    int bx = 0, by = 0;
    double best = sad(c, r, 0, 0, 2);
    for (int dy = -16; dy <= 16; dy += 2)
        for (int dx = -16; dx <= 16; dx += 2) {
            const double v = sad(c, r, dx, dy, 2) + 0.05 * (std::abs(dx) + std::abs(dy));
            if (v < best) { best = v; bx = dx; by = dy; }
        }
    {
        const int cx0 = bx, cy0 = by;
        best = sad(c, r, cx0, cy0, 1) + 0.05 * (std::abs(cx0) + std::abs(cy0));
        for (int dy = -2; dy <= 2; ++dy)
            for (int dx = -2; dx <= 2; ++dx) {
                const double v = sad(c, r, cx0 + dx, cy0 + dy, 1) + 0.05 * (std::abs(cx0 + dx) + std::abs(cy0 + dy));
                if (v < best) { best = v; bx = cx0 + dx; by = cy0 + dy; }
            }
    }
    int fx = bx * 4, fy = by * 4;
    best = sad(cur, ref, fx, fy, 2);
    const int cx = fx, cy = fy;
    for (int dy = -3; dy <= 3; ++dy)
        for (int dx = -3; dx <= 3; ++dx) {
            const double v = sad(cur, ref, cx + dx, cy + dy, 2);
            if (v < best) { best = v; fx = cx + dx; fy = cy + dy; }
        }
    gx = fx * 4; gy = fy * 4;
}

// 小さな最小二乗 (正規方程式 + ガウス消去)。A: rows x n
PqmfCfg pqmf_cfg(const FrameParams& fp, bool ll_split) {
    PqmfCfg c;
    c.ll_split = ll_split;
    c.lx = fp.pqmf_log2;
    c.ly = fp.pqmf_log2y ? fp.pqmf_log2y : fp.pqmf_log2;
    c.filter = fp.pqmf_filter;
    return c;
}

// §3.2 符号器の簡易選択: 候補 (フィルタ x 帯域形状) ごとに、帯域係数の重み付き L1 (合成側の寄与) と
//  合成残差 (完全再構成からのずれ) の L1 の和が最小のものを選ぶ (疎さの推定)
PqmfCfg choose_pqmf(const Plane& Y, int L) {
    PqmfCfg best;
    best.lx = best.ly = L;
    if (L <= 0) return best;
    std::vector<std::pair<int, int>> shapes = {{L, L}};
    if (L >= 2) { shapes.push_back({L, L - 1}); shapes.push_back({L - 1, L}); }
    double bc = 1e300;
    for (auto [lx, ly] : shapes) {
        const int Mx = 1 << lx, My = 1 << ly;
        const int W = Y.w / Mx * Mx, H = Y.h / My * My;
        if (W <= 0 || H <= 0) continue;
        std::vector<double> img(static_cast<size_t>(W) * H);
        for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) img[static_cast<size_t>(y) * W + x] = Y.at(x, y);
        for (int f = 0; f < 4; ++f) {
            Pqmf2D fb(Mx, My, kPqmfFilters[f].m, kPqmfFilters[f].beta);
            std::vector<std::vector<double>> bands;
            fb.analyze(img, W, H, bands);
            double cost = 0;
            for (int ky = 0; ky < My; ++ky)
                for (int kx = 0; kx < Mx; ++kx) {
                    if (!kx && !ky) continue;  // LL は予測で大半が除かれるため評価しない
                    double a = 0;
                    for (double v : bands[ky * Mx + kx]) a += std::abs(v);
                    cost += a * fb.band_norm(kx, ky);
                }
            std::vector<double> rec;
            fb.synthesize(bands, W, H, rec);
            for (size_t i = 0; i < rec.size(); ++i) cost += std::abs(rec[i] - img[i]);
            cost /= static_cast<double>(W) * H;
            if (cost < bc) { bc = cost; best.lx = lx; best.ly = ly; best.filter = f; }
        }
    }
    return best;
}

bool solve_ls(const std::vector<std::vector<double>>& A, const std::vector<double>& b, int n, std::vector<double>& x) {
    std::vector<double> M(static_cast<size_t>(n) * (n + 1), 0.0);
    for (size_t r = 0; r < A.size(); ++r)
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) M[i * (n + 1) + j] += A[r][i] * A[r][j];
            M[i * (n + 1) + n] += A[r][i] * b[r];
        }
    for (int i = 0; i < n; ++i) M[i * (n + 1) + i] += 1e-9;
    for (int c = 0; c < n; ++c) {
        int piv = c;
        for (int r = c + 1; r < n; ++r) if (std::abs(M[r * (n + 1) + c]) > std::abs(M[piv * (n + 1) + c])) piv = r;
        if (std::abs(M[piv * (n + 1) + c]) < 1e-12) return false;
        for (int k = 0; k <= n; ++k) std::swap(M[c * (n + 1) + k], M[piv * (n + 1) + k]);
        for (int r = 0; r < n; ++r) {
            if (r == c) continue;
            const double f = M[r * (n + 1) + c] / M[c * (n + 1) + c];
            for (int k = c; k <= n; ++k) M[r * (n + 1) + k] -= f * M[c * (n + 1) + k];
        }
    }
    x.resize(n);
    for (int i = 0; i < n; ++i) x[i] = M[i * (n + 1) + n] / M[i * (n + 1) + i];
    return true;
}

// グローバル動きモデルの推定 (§7.2, informative):
//  1/2 縮小画像の 8x8 ブロック (原寸 16x16) を並進候補の周りで探索 → 平坦ブロックを除いた動きベクトル群に
//  相似/アフィン/射影を外れ値除去付き最小二乗で当てはめ、実際の補償 SAD (疎標本) が最小のモデルを選ぶ。
GlobalModel estimate_global_model(const Plane& cur, const Plane& ref, int gx, int gy) {
    GlobalModel tr;
    tr.cx = cur.w / 2.0; tr.cy = cur.h / 2.0;
    tr.tx = gx; tr.ty = gy;
    auto down = [](const Plane& p) {
        Plane o(std::max(1, p.w / 2), std::max(1, p.h / 2));
        for (int y = 0; y < o.h; ++y)
            for (int x = 0; x < o.w; ++x)
                o.at(x, y) = (p.at(std::min(2 * x, p.w - 1), std::min(2 * y, p.h - 1)) + p.at(std::min(2 * x + 1, p.w - 1), std::min(2 * y, p.h - 1)) +
                              p.at(std::min(2 * x, p.w - 1), std::min(2 * y + 1, p.h - 1)) + p.at(std::min(2 * x + 1, p.w - 1), std::min(2 * y + 1, p.h - 1)) + 2) >> 2;
        return o;
    };
    const Plane c = down(cur), r = down(ref);
    struct Pt { double u, v, dx, dy; };
    std::vector<Pt> pts;
    const int B = 8, R = 6;
    for (int by = B; by + 2 * B <= c.h; by += B)
        for (int bx = B; bx + 2 * B <= c.w; bx += B) {
            double m = 0, var = 0;
            for (int y = 0; y < B; ++y) for (int x = 0; x < B; ++x) m += c.at(bx + x, by + y);
            m /= B * B;
            for (int y = 0; y < B; ++y) for (int x = 0; x < B; ++x) { const double d = c.at(bx + x, by + y) - m; var += d * d; }
            if (var / (B * B) < 25.0 * (1 << (2 * std::max(0, 0)))) continue;  // 平坦部は動きが不定
            const double gcx = 2.0 * (bx + B / 2) - tr.cx, gcy = 2.0 * (by + B / 2) - tr.cy;
            int mx, my;
            tr.mv_at(gcx + tr.cx, gcy + tr.cy, mx, my);
            const int px = static_cast<int>(std::lround(mx / 8.0)), py = static_cast<int>(std::lround(my / 8.0));
            int64_t best = INT64_MAX, second = INT64_MAX;
            int bdx = 0, bdy = 0;
            for (int dy = py - R; dy <= py + R; ++dy)
                for (int dx = px - R; dx <= px + R; ++dx) {
                    if (bx + dx < 0 || by + dy < 0 || bx + dx + B > r.w || by + dy + B > r.h) continue;
                    int64_t sad = 0;
                    for (int y = 0; y < B; ++y)
                        for (int x = 0; x < B; ++x) sad += std::abs(c.at(bx + x, by + y) - r.at(bx + dx + x, by + dy + y));
                    if (sad < best) { second = best; best = sad; bdx = dx; bdy = dy; }
                    else if (sad < second) second = sad;
                }
            if (best == INT64_MAX || second < best * 1.1) continue;  // 曖昧な一致 (繰り返し模様など) は捨てる
            pts.push_back({gcx, gcy, 2.0 * bdx, 2.0 * bdy});
        }
    std::vector<GlobalModel> cands = {tr};
    if (pts.size() >= 12) {
        for (int type = 1; type <= 3; ++type) {
            std::vector<char> use(pts.size(), 1);
            std::vector<double> p;
            bool ok = false;
            for (int it = 0; it < 3; ++it) {
                std::vector<std::vector<double>> A;
                std::vector<double> b;
                for (size_t i = 0; i < pts.size(); ++i) {
                    if (!use[i]) continue;
                    const Pt& q = pts[i];
                    const double xp = q.u + q.dx, yp = q.v + q.dy;
                    if (type == 1) {  // [A, B, Tx, Ty]
                        A.push_back({q.u, q.v, 1, 0}); b.push_back(q.dx);
                        A.push_back({q.v, -q.u, 0, 1}); b.push_back(q.dy);
                    } else if (type == 2) {  // [A, B, Tx, C, D, Ty]
                        A.push_back({q.u, q.v, 1, 0, 0, 0}); b.push_back(q.dx);
                        A.push_back({0, 0, 0, q.u, q.v, 1}); b.push_back(q.dy);
                    } else {  // [A, B, Tx, C, D, Ty, h31, h32]  (1+A)u + Bv + Tx - x'(h31 u + h32 v) = x'
                        A.push_back({q.u, q.v, 1, 0, 0, 0, -xp * q.u, -xp * q.v}); b.push_back(xp - q.u);
                        A.push_back({0, 0, 0, q.u, q.v, 1, -yp * q.u, -yp * q.v}); b.push_back(yp - q.v);
                    }
                }
                const int n = type == 1 ? 4 : type == 2 ? 6 : 8;
                if (static_cast<int>(A.size()) < 2 * n) break;
                ok = solve_ls(A, b, n, p);
                if (!ok) break;
                GlobalModel g = tr;
                g.type = type;
                if (type == 1) { g.a = g.d = static_cast<int>(std::lround(p[0] * 65536)); g.b = static_cast<int>(std::lround(p[1] * 65536)); g.c = -g.b;
                                 g.tx = static_cast<int>(std::lround(p[2] * 4)); g.ty = static_cast<int>(std::lround(p[3] * 4)); }
                else { g.a = static_cast<int>(std::lround(p[0] * 65536)); g.b = static_cast<int>(std::lround(p[1] * 65536)); g.tx = static_cast<int>(std::lround(p[2] * 4));
                       g.c = static_cast<int>(std::lround(p[3] * 65536)); g.d = static_cast<int>(std::lround(p[4] * 65536)); g.ty = static_cast<int>(std::lround(p[5] * 4));
                       if (type == 3) { g.h31 = static_cast<int>(std::lround(p[6] * 16777216.0)); g.h32 = static_cast<int>(std::lround(p[7] * 16777216.0)); } }
                for (int* v : {&g.a, &g.b, &g.c, &g.d, &g.h31, &g.h32}) *v = std::clamp(*v, -32767, 32767);
                // 残差で外れ値を除外して再推定
                std::vector<double> res;
                for (size_t i = 0; i < pts.size(); ++i) {
                    int mx, my;
                    g.mv_at(pts[i].u + g.cx, pts[i].v + g.cy, mx, my);
                    res.push_back(std::hypot(mx / 4.0 - pts[i].dx, my / 4.0 - pts[i].dy));
                }
                std::vector<double> sr;
                for (size_t i = 0; i < res.size(); ++i) if (use[i]) sr.push_back(res[i]);
                std::nth_element(sr.begin(), sr.begin() + sr.size() / 2, sr.end());
                const double th = std::max(1.5, 2.5 * sr[sr.size() / 2]);
                for (size_t i = 0; i < res.size(); ++i) use[i] = res[i] <= th;
                if (it == 2) cands.push_back(g);
            }
        }
    }
    // 疎標本の補償 SAD で選択 (パラメータ数に応じた小さな罰則)
    double best = 1e300;
    GlobalModel bestg = tr;
    for (const GlobalModel& g : cands) {
        double sad = 0;
        int64_t n = 0;
        for (int by = 0; by + 8 <= cur.h; by += 16)
            for (int bx = 0; bx + 8 <= cur.w; bx += 16) {
                int mx, my;
                g.mv_at(bx + 4, by + 4, mx, my);
                const int ix = (mx + 2) >> 2, iy = (my + 2) >> 2;
                for (int y = 0; y < 8; ++y)
                    for (int x = 0; x < 8; ++x) {
                        const int rx = std::clamp(bx + x + ix, 0, ref.w - 1), ry = std::clamp(by + y + iy, 0, ref.h - 1);
                        sad += std::abs(cur.at(bx + x, by + y) - ref.at(rx, ry));
                        ++n;
                    }
            }
        sad = sad / std::max<int64_t>(1, n) * (1.0 + 0.004 * g.type);
        if (sad < best) { best = sad; bestg = g; }
    }
    return bestg;
}

// フレームのペイロード (rANS) を符号化/復号。org==nullptr なら復号。
void code_frame(SymIO& io, const VideoInfo& info, const FrameParams& fp, const Frame* org, Frame& rec, CodecState& st,
                TileStreams& ts, BlockUsage* usage = nullptr) {
    Models md;
    rec.p.resize(3);
    if (fp.dict_reset) st.dict.reset_dynamic();
    // 時間方向の図形予測の参照は関数の入口で固定し、更新は最後に 1 回だけ (内部の試行符号化で状態がずれないように)
    const std::vector<Shape> shapes_ref = st.last_shapes;
    std::vector<Shape> shapes_new;
    bool shapes_coded = false;
    const bool inter = fp.type != FrameType::I;
    const int cs = info.chroma == ChromaFormat::C420 ? 1 : 0;
    // 参照とグローバルパラメータ (§7.2)
    std::vector<const Frame*> refs[2];
    for (int l = 0; l < 2; ++l)
        for (int poc : fp.ref_poc[l]) {
            const Frame* f = st.find(poc);
            if (!f) throw std::runtime_error("missing reference picture");
            refs[l].push_back(f);
        }
    int gmv_x = 0, gmv_y = 0, gain_q[3] = {64, 64, 64}, offs[3] = {0, 0, 0}, hgain[3] = {64, 64, 64};
    // 帯域別ゲイン (§7.2 band_gain): 参照の高域を hg/64 倍 (フォーカス変化・ぼけ/鮮鋭化の全面的な変化)
    auto lowpass = [](const Plane& p) {
        Plane o(p.w, p.h);
        auto at = [&](int x, int y) { return p.at(std::clamp(x, 0, p.w - 1), std::clamp(y, 0, p.h - 1)); };
        for (int y = 0; y < p.h; ++y)
            for (int x = 0; x < p.w; ++x)
                o.at(x, y) = (4 * at(x, y) + 2 * (at(x - 1, y) + at(x + 1, y) + at(x, y - 1) + at(x, y + 1)) + at(x - 1, y - 1) +
                              at(x + 1, y - 1) + at(x - 1, y + 1) + at(x + 1, y + 1) + 8) >> 4;
        return o;
    };
    GlobalModel gm;
    gm.cx = info.width / 2.0; gm.cy = info.height / 2.0;
    if (inter) {
        if (io.w && fp.gm_valid) {
            gmv_x = fp.gm_x; gmv_y = fp.gm_y;
            gm = fp.gm_model;
            for (int pi = 0; pi < 3; ++pi) { gain_q[pi] = fp.gm_gain[pi]; offs[pi] = fp.gm_off[pi]; hgain[pi] = fp.gm_hg[pi]; }
        } else if (io.w) {
            estimate_global_motion(org->p[0], refs[0][0]->p[0], gmv_x, gmv_y);
            gm = estimate_global_model(org->p[0], refs[0][0]->p[0], gmv_x, gmv_y);
            gmv_x = gm.tx; gmv_y = gm.ty;
            for (int pi = 0; pi < 3; ++pi) {
                // MC(ref0, gmv) と原画の線形回帰でゲイン/オフセット
                const Plane& o = org->p[pi];
                RefPlane rp; rp.p = &refs[0][0]->p[pi];
                MotionInfo mi; mi.dir = 1; mi.mvx[0] = static_cast<int16_t>(gmv_x); mi.mvy[0] = static_cast<int16_t>(gmv_y);
                std::vector<int32_t> pr(static_cast<size_t>(o.w) * o.h);
                inter_predict(mi, &rp, nullptr, 0, 0, o.w, o.h, pi ? cs : 0, -(1 << 20), 1 << 20, pr.data());
                double mo = 0, mp = 0;
                for (size_t i = 0; i < pr.size(); ++i) { mo += o.v[i]; mp += pr[i]; }
                mo /= pr.size(); mp /= pr.size();
                double cov = 0, var = 0;
                for (size_t i = 0; i < pr.size(); ++i) { cov += (o.v[i] - mo) * (pr[i] - mp); var += (pr[i] - mp) * (pr[i] - mp); }
                const double g = var > 0 ? cov / var : 1.0;
                int gq = static_cast<int>(std::lround(g * 64.0));
                if (std::abs(gq - 64) <= 1 || gq < 32 || gq > 128) gq = 64;
                const int of = static_cast<int>(std::lround(mo - gq / 64.0 * mp));
                gain_q[pi] = gq;
                offs[pi] = std::abs(of) >= 1 ? of : 0;
                // 高域ゲイン: 原画の高域を予測の高域へ回帰
                {
                    Plane pp(o.w, o.h);
                    pp.v.assign(pr.begin(), pr.end());
                    const Plane pl = lowpass(pp), ol = lowpass(o);
                    double num = 0, den = 0, e0 = 0;
                    for (size_t i = 0; i < pr.size(); ++i) {
                        const double hp = pr[i] - pl.v[i], ho = o.v[i] - ol.v[i];
                        num += hp * ho; den += hp * hp;
                        const double e = o.v[i] - pr[i];
                        e0 += e * e;
                    }
                    const int hq = den > 0 ? std::clamp(static_cast<int>(std::lround(num / den * 64.0)), 0, 128) : 64;
                    // 高域を (g-1) 倍足したときの SSE 減少 ≈ 2(g-1)Σhp·(o-p)_hf - (g-1)²Σhp²。全体の 2% 以上で採用
                    const double g1 = hq / 64.0 - 1.0;
                    const double gain = 2.0 * g1 * (num - den) - g1 * g1 * den;
                    hgain[pi] = (std::abs(hq - 64) > 4 && gain > 0.02 * e0) ? hq : 64;
                }
            }
            fp.gm_valid = true;
            fp.gm_x = gmv_x; fp.gm_y = gmv_y;
            fp.gm_model = gm;
            for (int pi = 0; pi < 3; ++pi) { fp.gm_gain[pi] = gain_q[pi]; fp.gm_off[pi] = offs[pi]; fp.gm_hg[pi] = hgain[pi]; }
        }
        // ゲイン推定は並進で行う (上の回帰)。モデルの種類とパラメータを伝送
        gm.type = static_cast<int>(io.uint(md.global, 4, 0, static_cast<uint32_t>(gm.type)));
        if (gm.type > 3) throw std::runtime_error("corrupt stream: global model");
        if (gm.type >= 1) {
            gm.a = io.sint(md.global, 5, 0, gm.a);
            gm.b = io.sint(md.global, 6, 0, gm.b);
            if (gm.type == 1) { gm.d = gm.a; gm.c = -gm.b; }
        }
        if (gm.type >= 2) {
            gm.c = io.sint(md.global, 7, 0, gm.c);
            gm.d = io.sint(md.global, 8, 0, gm.d);
        }
        if (gm.type == 3) {
            gm.h31 = io.sint(md.global, 9, 0, gm.h31);
            gm.h32 = io.sint(md.global, 10, 0, gm.h32);
        } else {
            gm.h31 = gm.h32 = 0;
        }
        if (gm.type < 2 && gm.type == 0) gm.a = gm.b = gm.c = gm.d = 0;
        for (int v : {gm.a, gm.b, gm.c, gm.d, gm.h31, gm.h32})
            if (std::abs(v) > 32767) throw std::runtime_error("corrupt stream: global model params");
        gmv_x = io.sint(md.global, 0, 0, gmv_x);
        gmv_y = io.sint(md.global, 1, 0, gmv_y);
        gm.tx = gmv_x; gm.ty = gmv_y;
        for (int pi = 0; pi < 3; ++pi) {
            gain_q[pi] = 64 + io.sint(md.global, 2, static_cast<uint32_t>(pi), gain_q[pi] - 64);
            offs[pi] = io.sint(md.global, 3, static_cast<uint32_t>(pi), offs[pi]);
            hgain[pi] = 64 + io.sint(md.global, 11, static_cast<uint32_t>(pi), hgain[pi] - 64);
            if (hgain[pi] < 0 || hgain[pi] > 128) throw std::runtime_error("corrupt stream: band gain");
            if (gain_q[pi] < 0 || gain_q[pi] > 256 || std::abs(gmv_x) > 32000 || std::abs(gmv_y) > 32000)
                throw std::runtime_error("corrupt stream: global params");
        }
    }
    MotionField mf;
    Plane luma_rec;
    Plane fref;  // 帯域別ゲインを適用した参照 (プレーンごとに作り直す)
    for (int pi = 0; pi < 3; ++pi) {
        const int w = pi ? info.chroma_w() : info.width, h = pi ? info.chroma_h() : info.height;
        int32_t lo, hi;
        plane_range(info, pi, lo, hi);
        const int32_t mid = (lo + hi + 1) / 2;
        const int W = (w + kCtu - 1) / kCtu * kCtu, H = (h + kCtu - 1) / kCtu * kCtu;
        InterCtx ic;
        if (inter) {
            ic.enabled = true;
            ic.bframe = fp.type == FrameType::B;
            for (int l = 0; l < 2; ++l) {
                ic.nref[l] = static_cast<int>(refs[l].size());
                RefPlane* arr = l ? ic.l1 : ic.l0;
                for (size_t r = 0; r < refs[l].size() && r < 4; ++r) arr[r].p = &refs[l][r]->p[pi];
            }
            if (hgain[pi] != 64) {
                // L0 先頭参照を帯域別ゲインで変形したもの (低域そのまま + 高域 hg/64 倍)
                const Plane& rp = refs[0][0]->p[pi];
                const Plane lp = lowpass(rp);
                fref = rp;
                for (size_t i = 0; i < fref.v.size(); ++i)
                    fref.v[i] = std::clamp(lp.v[i] + static_cast<int32_t>((static_cast<int64_t>(hgain[pi]) * (rp.v[i] - lp.v[i]) + 32) >> 6), lo, hi);
                ic.l0[0].p = &fref;
            }
            ic.l0[0].gain_q = gain_q[pi];
            ic.l0[0].off = offs[pi];
            if (pi == 0) mf.init(W, H);
            ic.mf = &mf;
            ic.chroma_shift = pi ? cs : 0;
            ic.gmv_x = gmv_x; ic.gmv_y = gmv_y;
            ic.gm = gm;
            ic.cur_poc = fp.poc;
            for (int l = 0; l < 2; ++l)
                for (size_t r = 0; r < fp.ref_poc[l].size() && r < 4; ++r) ic.ref_poc[l][r] = fp.ref_poc[l][r];
            const int colp = !fp.ref_poc[1].empty() ? fp.ref_poc[1][0] : fp.ref_poc[0][0];
            const Picture* cp = st.find_pic(colp);
            if (cp && cp->mf && cp->mf->w4 * 4 == W && cp->mf->h4 * 4 == H) {
                ic.col = cp->mf.get();
                ic.col_poc = colp;
                for (int l = 0; l < 2; ++l)
                    for (size_t r = 0; r < cp->ref_poc[l].size() && r < 4; ++r) ic.col_ref_poc[l][r] = cp->ref_poc[l][r];
            }
        }
        Plane out(w, h);
        if (fp.type == FrameType::Copy) {
            // COPY: グローバル予測そのもの (§7.7)。並進以外は 8x8 輝度ブロックごとにモデルの動きで補償
            if (gm.type == 0) {
                MotionInfo mi; mi.dir = 1; mi.mvx[0] = static_cast<int16_t>(gmv_x); mi.mvy[0] = static_cast<int16_t>(gmv_y);
                inter_predict(mi, ic.l0, nullptr, 0, 0, w, h, pi ? cs : 0, lo, hi, out.v.data());
            } else {
                const int sh = pi ? cs : 0, bsz = 8 >> sh;
                std::vector<int32_t> blk(static_cast<size_t>(bsz) * bsz);
                for (int by = 0; by < h; by += bsz)
                    for (int bx = 0; bx < w; bx += bsz) {
                        const int bw = std::min(bsz, w - bx), bh = std::min(bsz, h - by);
                        int mx, my;
                        gm.mv_at((bx << sh) + 4, (by << sh) + 4, mx, my);
                        MotionInfo mi; mi.dir = 1; mi.mvx[0] = static_cast<int16_t>(mx); mi.mvy[0] = static_cast<int16_t>(my);
                        inter_predict(mi, ic.l0, nullptr, bx, by, bw, bh, sh, lo, hi, blk.data());
                        for (int y = 0; y < bh; ++y)
                            for (int x = 0; x < bw; ++x) out.at(bx + x, by + y) = blk[y * bw + x];
                    }
            }
            rec.p[pi] = std::move(out);
            continue;
        }
        Plane opad;
        if (org) opad = pad_plane(org->p[pi], W, H);
        Plane R(W, H, 0);
        BlockUsage* cur_usage = usage;
        bool allow_ns = true;  // 方式判定の試行中はノイズ置換を止める (知覚目的の加算を SSE 判定に含めない)
        // プレーンの非可逆符号化 (bands: PQMF 帯域方式 / false: 画素領域のブロック方式)
        bool ll_split = false;  // 帯域方式で LL を部分木分解するか (プレーンごと、§3.2)
        auto code_lossy = [&](SymIO& io, Models& md, TileStreams& ts, bool bands, Plane& R) {
            const int pqp = std::clamp(fp.qp + (pi ? fp.cqp_off : 0), 0, 63);
            const double step = qp_step(pqp, info.bit_depth);
            if (bands && !inter && fp.band_blocks) {
                const double lam = 0.57 * std::pow(2.0, (pqp - 12) / 3.0) * std::pow(4.0, info.bit_depth - 8);
                code_bands_blocks(io, md, pqmf_cfg(fp, ll_split), org ? &opad : nullptr, nullptr, R, step, lam, lo, hi, pi, fp.tools,
                                  fp.search, fp.min_log2, fp.max_log2, false, 0);
                if (fp.alf) code_alf(io, md.lf, R, org ? &opad : nullptr, pi == 0, info.bit_depth, lam, lo, hi);
            } else if (bands && !inter) {
                code_bands(io, md, pqmf_cfg(fp), org ? &opad : nullptr, R, step, 1.0 / 3.0, lo, hi, fp.band_tools, fp.psy,
                           static_cast<uint64_t>(fp.poc) * 3 + pi);
            } else {
                Plane lds;
                if (pi > 0 && fp.tools.cfl) lds = luma_at_chroma(luma_rec, info, W, H);
                const double lambda = 0.57 * std::pow(2.0, (pqp - 12) / 3.0) * std::pow(4.0, info.bit_depth - 8);
                // 図形レイヤ (§4): 輝度のみ、I フレームのみ。符号器は RD で採否を決める
                std::vector<Shape> shapes;
                Plane S(W, H, 0);
                bool use_shapes = false;
                if (fp.shapes && pi == 0) {
                    if (io.w) {
                        // インターでは大域予測からの差分に当てはめる (動き補償で表せない局所的な明暗・形の変化)
                        Plane fitsrc = opad;
                        if (inter) {
                            std::vector<int32_t> gp(static_cast<size_t>(W) * H);
                            MotionInfo gmi; gmi.dir = 1; gmi.mvx[0] = static_cast<int16_t>(gmv_x); gmi.mvy[0] = static_cast<int16_t>(gmv_y);
                            inter_predict(gmi, ic.l0, nullptr, 0, 0, W, H, 0, lo, hi, gp.data());
                            for (size_t i = 0; i < fitsrc.v.size(); ++i) fitsrc.v[i] -= gp[i];
                        }
                        shapes = fit_shapes(fitsrc, inter ? 16 : 32);
                        // 図形の追跡: 前フレームの図形と近いものは同じ番号に並べ替える (時間差分が小さくなる)
                        if (inter && !shapes_ref.empty()) {
                            std::vector<Shape> ord;
                            std::vector<char> used(shapes.size(), 0);
                            for (const Shape& t : shapes_ref) {
                                int bi = -1; long long bd = 1LL << 40;
                                for (size_t k = 0; k < shapes.size(); ++k) {
                                    if (used[k]) continue;
                                    const long long d = 1LL * (shapes[k].cx - t.cx) * (shapes[k].cx - t.cx) + 1LL * (shapes[k].cy - t.cy) * (shapes[k].cy - t.cy);
                                    if (d < bd) { bd = d; bi = static_cast<int>(k); }
                                }
                                if (bi < 0) break;
                                used[bi] = 1;
                                ord.push_back(shapes[bi]);
                            }
                            for (size_t k = 0; k < shapes.size(); ++k) if (!used[k]) ord.push_back(shapes[k]);
                            shapes.swap(ord);
                        }
                        if (!shapes.empty()) {
                            render_shapes(shapes, S);
                            Plane d = opad;
                            for (size_t i = 0; i < d.v.size(); ++i) d.v[i] -= S.v[i];
                            Plane r1(W, H, 0), r2(W, H, 0);
                            BlockCoder b1(&r1, &opad, nullptr, pi, lo, hi, step, lambda, fp.min_log2, fp.max_log2, fp.tools, fp.search,
                                          inter ? &ic : nullptr);
                            BlockCoder b2(&r2, &d, nullptr, pi, lo - hi, hi * 2, step, lambda, fp.min_log2, fp.max_log2, fp.tools, fp.search,
                                          inter ? &ic : nullptr);
                            double j1 = 0, j2 = lambda * shapes_bits_estimate(shapes);
                            for (int cy = 0; cy < H; cy += kCtu)
                                for (int cx = 0; cx < W; cx += kCtu) { j1 += b1.rd_ctu(cx, cy, kCtu); j2 += b2.rd_ctu(cx, cy, kCtu); }
                            if (j2 >= j1) shapes.clear();
                        }
                    }
                    code_shapes(io, md.shape, shapes, inter ? &shapes_ref : nullptr);
                    shapes_new = shapes;
                    shapes_coded = true;
                    use_shapes = !shapes.empty();
                    S = Plane(W, H, 0);
                    if (use_shapes) render_shapes(shapes, S);
                }
                Plane target;
                if (use_shapes && org) { target = opad; for (size_t i = 0; i < target.v.size(); ++i) target.v[i] -= S.v[i]; }
                const int32_t blo = use_shapes ? lo - hi : lo, bhi = use_shapes ? hi * 2 : hi;
                // 適応 QP: 色差 CTU は対応する輝度 CTU の値を使う
                struct AqpCtx { const FrameParams* fp; int shift; } actx{&fp, pi ? cs : 0};
                auto aqp_lookup = [](void* c, int cx, int cy) -> int {
                    const AqpCtx& a = *static_cast<AqpCtx*>(c);
                    if (a.fp->aqp_map.empty()) return 0;
                    const int lx = std::min(((cx << a.shift) / kCtu), a.fp->aqp_w - 1);
                    const int ly = std::min(((cy << a.shift) / kCtu), static_cast<int>(a.fp->aqp_map.size()) / a.fp->aqp_w - 1);
                    return a.fp->aqp_map[static_cast<size_t>(ly) * a.fp->aqp_w + lx];
                };
                // 動画での PQMF (§3, §7.5): P/B では予測のみをブロックで作り、残差を帯域符号化する
                Tools btools = fp.tools;
                btools.pred_only = inter && bands;
                // タイル分割と並列符号化/復号
                const int ncx = W / kCtu, ncy = H / kCtu;
                const int tcn = std::clamp(fp.tile_cols, 1, ncx), trn = std::clamp(fp.tile_rows, 1, ncy);
                const int nt = tcn * trn;
                std::vector<std::unique_ptr<BlockCoder>> coders(nt);
                std::vector<std::unique_ptr<Models>> tmd(nt);
                std::vector<std::unique_ptr<EntropyWriter>> tw(nt);
                std::vector<std::unique_ptr<EntropyReader>> tr(nt);
                std::vector<SymIO> tio(nt);
                std::vector<std::array<int, 4>> rect(nt);
                for (int ty = 0; ty < trn; ++ty)
                    for (int tx = 0; tx < tcn; ++tx) {
                        const int t = ty * tcn + tx;
                        const int x0 = tx * ncx / tcn * kCtu, x1 = (tx + 1) * ncx / tcn * kCtu;
                        const int y0 = ty * ncy / trn * kCtu, y1 = (ty + 1) * ncy / trn * kCtu;
                        rect[t] = {x0, y0, x1, y1};
                        coders[t] = std::make_unique<BlockCoder>(
                            &R, org ? (use_shapes ? &target : &opad) : nullptr, (pi > 0 && fp.tools.cfl) ? &lds : nullptr, pi, blo,
                            bhi, step, lambda, fp.min_log2, fp.max_log2, btools, fp.search, inter ? &ic : nullptr, &st.dict, x0,
                            y0, x1 - x0, y1 - y0, t << 24);
                        tmd[t] = std::make_unique<Models>();
                        if (fp.aqp) coders[t]->enable_aqp(pqp, info.bit_depth, aqp_lookup, &actx);
                        coders[t]->set_noise_seed(static_cast<uint64_t>(fp.poc) * 7919u + static_cast<uint64_t>(pi) * 104729u + static_cast<uint64_t>(t));
                        if (io.enc) {
                            tw[t] = std::make_unique<EntropyWriter>();
                            tio[t].w = tw[t].get();
                            tio[t].enc = true;
                        } else {
                            if (ts.next >= ts.in.size()) throw std::runtime_error("corrupt stream: missing tile");
                            const auto& b = ts.in[ts.next++];
                            tr[t] = std::make_unique<EntropyReader>(b.first, b.second);
                            tio[t].r = tr[t].get();
                        }
                    }
                std::vector<std::string> errs(nt);
                auto run_tile = [&](int t) {
                    try {
                        for (int cy = rect[t][1]; cy < rect[t][3]; cy += kCtu)
                            for (int cx = rect[t][0]; cx < rect[t][2]; cx += kCtu) coders[t]->code_ctu(tio[t], *tmd[t], cx, cy, kCtu);
                    } catch (const std::exception& ex) {
                        errs[t] = ex.what();
                    }
                };
                const int nth = std::clamp(fp.threads, 1, nt);
                if (nth <= 1) {
                    for (int t = 0; t < nt; ++t) run_tile(t);
                } else {
                    std::vector<std::thread> pool;
                    std::atomic<int> nextt{0};
                    for (int k = 0; k < nth; ++k)
                        pool.emplace_back([&] { for (int t; (t = nextt.fetch_add(1)) < nt;) run_tile(t); });
                    for (auto& th : pool) th.join();
                }
                for (const auto& e : errs) if (!e.empty()) throw std::runtime_error(e);
                if (io.enc) for (int t = 0; t < nt; ++t) ts.out.push_back(tw[t]->finish());
                if (io.w && std::getenv("FVC_BITS")) {
                    Models sum;
                    const char* names[] = {"split", "mode", "cbf", "last", "coef_y", "coef_c", "ibc", "cfl", "tns", "e8", "nf", "inter", "mvd", "dict", "aqp", "gs"};
                    for (int t = 0; t < nt; ++t) {
                        const CMModel* src[] = {&tmd[t]->split, &tmd[t]->mode, &tmd[t]->cbf, &tmd[t]->last, &tmd[t]->coef_y, &tmd[t]->coef_c,
                                                &tmd[t]->ibc, &tmd[t]->cfl, &tmd[t]->tns, &tmd[t]->e8, &tmd[t]->nf, &tmd[t]->inter,
                                                &tmd[t]->mvd, &tmd[t]->dict, &tmd[t]->aqp, &tmd[t]->gs};
                        for (int i = 0; i < 16; ++i)
                            std::fprintf(stderr, "[bits] plane=%d %-7s n0=%9.0f n1=%9.0f rest=%9.0f\n", pi, names[i], src[i]->stat_bits[0],
                                         src[i]->stat_bits[1], src[i]->stat_bits[2]);
                    }
                }
                if (cur_usage) for (int t = 0; t < nt; ++t) cur_usage->add(coders[t]->usage());
                // ループフィルタ用のブロック情報をタイルから合成
                EdgeInfo einfo;
                einfo.w4 = W / 4; einfo.h4 = H / 4;
                einfo.leaf.assign(static_cast<size_t>(einfo.w4) * einfo.h4, -1);
                einfo.flags.assign(einfo.leaf.size(), 0);
                for (int t = 0; t < nt; ++t)
                    for (int y = rect[t][1] / 4; y < rect[t][3] / 4; ++y)
                        for (int x = rect[t][0] / 4; x < rect[t][2] / 4; ++x) {
                            const size_t i = static_cast<size_t>(y) * einfo.w4 + x;
                            einfo.leaf[i] = coders[t]->leaf_ids()[i];
                            einfo.flags[i] = coders[t]->leaf_flags()[i];
                        }
                if (use_shapes)
                    for (size_t i = 0; i < R.v.size(); ++i) R.v[i] = std::clamp(R.v[i] + S.v[i], lo, hi);
                if (btools.pred_only) {
                    // 帯域領域の予測符号化: P (動き補償予測) の帯域を時間方向予測とし、
                    // 帯域ブロックごとにゲイン・帯域間予測・スキップを探索して符号化
                    const Plane P = R;
                    if (fp.band_blocks)
                        code_bands_blocks(io, md, pqmf_cfg(fp, ll_split), org ? &opad : nullptr, &P, R, step, lambda, lo, hi, pi, fp.tools,
                                          fp.search, fp.min_log2, fp.max_log2, fp.band_ns && allow_ns,
                                          static_cast<uint64_t>(fp.poc) * 3 + static_cast<uint64_t>(pi));
                    else
                        code_bands_video(io, md, pqmf_cfg(fp), org ? &opad : nullptr, P, R, step, lambda, lo, hi);
                }
                if (fp.lf) {
                    code_loop_filter(io, md.lf, R, org ? &opad : nullptr, einfo, inter ? &mf : nullptr, pi > 0, pi ? cs : 0, pqp,
                                     info.bit_depth, step, lambda, fp.lf_freq, fp.lf_map, lo, hi);
                }
                if (fp.cdef) code_cdef(io, md.lf, R, org ? &opad : nullptr, pi == 0, pqp, info.bit_depth, lambda, lo, hi);
                if (fp.alf) code_alf(io, md.lf, R, org ? &opad : nullptr, pi == 0, info.bit_depth, lambda, lo, hi);
            }
                };
        if (fp.lossy) {
            bool bands = fp.pqmf_log2 > 0;
            if (fp.pqmf_log2 > 0) {
                // 符号器: 帯域方式と画素ブロック方式を実際に符号化して J = D + λR で選ぶ (選択は 1 ビット)
                if (io.enc) {
                    const int pqp = std::clamp(fp.qp + (pi ? fp.cqp_off : 0), 0, 63);
                    const double lam = 0.57 * std::pow(2.0, (pqp - 12) / 3.0) * std::pow(4.0, info.bit_depth - 8);
                    double bestj = 1e300;
                    bool best_ll = false;
                    for (int b = 0; b < 3; ++b) {
                        if (b == 2 && !fp.band_blocks) break;
                        ll_split = b == 2;
                        Models mt = md;
                        EntropyWriter tw;
                        SymIO tio;
                        tio.w = &tw;
                        tio.enc = true;
                        TileStreams tts;
                        Plane Rt(W, H, 0);
                        cur_usage = nullptr;
                        allow_ns = false;
                        code_lossy(tio, mt, tts, b != 0, Rt);
                        allow_ns = true;
                        size_t bytes = tw.finish().size();
                        for (const auto& t : tts.out) bytes += t.size();
                        double d = 0;
                        for (int y = 0; y < h; ++y)
                            for (int x = 0; x < w; ++x) { const double e = org->p[pi].at(x, y) - Rt.at(x, y); d += e * e; }
                        const double j = d + lam * 8.0 * bytes;
                        if (j < bestj) { bestj = j; bands = b != 0; best_ll = b == 2; }
                    }
                    ll_split = best_ll;
                    cur_usage = usage;
                }
                bands = io.bit(md.band_mode, 9, static_cast<uint32_t>(pi), 0, bands);
                ll_split = bands && fp.band_blocks ? io.bit(md.band_mode, 10, static_cast<uint32_t>(pi), 0, ll_split) : false;
            }
            code_lossy(io, md, ts, bands, R);
        }
        if (pi == 0) luma_rec = R;
        for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) out.at(x, y) = R.at(x, y);
        if (fp.l2 || !fp.lossy) code_lossless(io, md, pi, w, h, org ? &org->p[pi] : nullptr, out, fp.lossy, mid);
        rec.p[pi] = std::move(out);
    }
    // 時間方向候補のために輝度の動きを保存 (I: 全イントラ, COPY: グローバル動き)
    {
        const int W = (info.width + kCtu - 1) / kCtu * kCtu, H = (info.height + kCtu - 1) / kCtu * kCtu;
        auto smf = std::make_shared<MotionField>();
        if (inter && fp.type != FrameType::Copy && mf.w4 * 4 == W) {
            *smf = mf;
        } else {
            smf->init(W, H);
            if (fp.type == FrameType::Copy)
                for (int y = 0; y < H; y += 4)
                    for (int x = 0; x < W; x += 4) {
                        int mx, my;
                        gm.mv_at(x + 2, y + 2, mx, my);
                        MotionInfo g; g.dir = 1; g.mvx[0] = static_cast<int16_t>(mx); g.mvy[0] = static_cast<int16_t>(my);
                        smf->fill(x, y, 4, 4, g);
                    }
        }
        st.last_mf = smf;
        st.last_ref_poc[0] = fp.ref_poc[0];
        st.last_ref_poc[1] = fp.ref_poc[1];
    }
    if (shapes_coded) st.last_shapes = shapes_new;
    // 辞書更新 (ADD_FROM_RECON, §10.2): 符号器は輝度再構成の高テクスチャブロックを選ぶ
    if (fp.tools.dict && fp.type != FrameType::Copy) {
        std::vector<std::array<int, 3>> adds;  // x/8, y/8, size
        if (io.w) {
            const Plane& L = rec.p[0];
            std::vector<std::pair<double, std::array<int, 3>>> cand;
            for (int s : {8, 16})
                for (int y = 0; y + s <= L.h; y += s)
                    for (int x = 0; x + s <= L.w; x += s) {
                        double m = 0, v = 0;
                        for (int j = 0; j < s; ++j) for (int i = 0; i < s; ++i) m += L.at(x + i, y + j);
                        m /= s * s;
                        for (int j = 0; j < s; ++j) for (int i = 0; i < s; ++i) { const double d = L.at(x + i, y + j) - m; v += d * d; }
                        cand.push_back({v / (s * s), {x / 8, y / 8, s}});
                    }
            std::sort(cand.begin(), cand.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
            if (fp.dict_add) for (size_t i = 0; i < cand.size() && i < 16; ++i) adds.push_back(cand[i].second);
        }
        const uint32_t n = io.uint(md.dict, 10, 0, static_cast<uint32_t>(adds.size()));
        if (n > 1024) throw std::runtime_error("corrupt stream: dict adds");
        adds.resize(n);
        for (auto& a : adds) {
            a[0] = static_cast<int>(io.uint(md.dict, 11, 0, static_cast<uint32_t>(a[0])));
            a[1] = static_cast<int>(io.uint(md.dict, 12, 0, static_cast<uint32_t>(a[1])));
            a[2] = io.bit(md.dict, 13, 0, 0, a[2] == 16) ? 16 : 8;
            if (a[0] * 8 >= info.width || a[1] * 8 >= info.height) throw std::runtime_error("corrupt stream: dict pos");
            st.dict.add_from(rec.p[0], a[0] * 8, a[1] * 8, a[2]);
        }
    }
}

FrameParams params_from(const EncoderConfig& c) {
    FrameParams fp;
    fp.qp = std::clamp(c.qp, 0, 63);
    fp.cqp_off = std::clamp(c.chroma_qp_offset, -12, 12);
    fp.lossy = c.lossy_layer;
    fp.l2 = c.l2_lossless;
    fp.pqmf_log2 = std::clamp(c.pqmf_log2, 0, 4);
    fp.psy = c.psy;
    Tools& t = fp.tools;
    Search& s = fp.search;
    t.cfl = true;
    t.nf = c.psy;
    t.fir = true;
    switch (c.preset) {
    case Preset::Faster:
        fp.min_log2 = 3; fp.max_log2 = 5; t.all_angular = false; s.rd_modes = 1;
        t.fir = false; s.me_range = 2; s.me_bi = false; s.qpel = false; s.inter_skip_intra = true; s.max_rd_cands = 1; s.skip_split_on_skip = true; s.approx_subpel = true;
        break;
    case Preset::Fast:
        fp.min_log2 = 3; fp.max_log2 = 6; s.rd_modes = 2; s.me_range = 4;
        fp.lf_map = true; s.rdoq = true; s.inter_skip_intra = true; s.max_rd_cands = 2; s.skip_split_on_skip = true; s.approx_subpel = true;
        break;
    case Preset::Medium:
        fp.min_log2 = 2; fp.max_log2 = 6; s.rd_modes = 3; s.me_range = 8;
        fp.band_tools = true; fp.lf_freq = fp.lf_map = true;
        s.rdoq = true;
        break;
    case Preset::Slow:
        fp.min_log2 = 2; fp.max_log2 = 6; s.rd_modes = 6; s.me_range = 16;
        t.tns = t.ibc = t.dict = true; s.try_tns = true; s.ibc_range = 32; s.rdoq = true;
        fp.band_tools = true; fp.lf_freq = fp.lf_map = true;
        break;
    case Preset::Placebo:
        fp.min_log2 = 2; fp.max_log2 = 6; s.rd_modes = 35; s.me_range = 32;
        t.tns = t.e8 = t.ibc = t.dict = true; s.try_tns = s.try_e8 = true; s.ibc_range = 96; s.rdoq = true;
        fp.shapes = true; fp.band_tools = true; fp.lf_freq = fp.lf_map = true;
        break;
    }
    if (c.ibc >= 0) t.ibc = c.ibc != 0;
    if (c.e8 >= 0) { t.e8 = s.try_e8 = c.e8 != 0; }
    if (c.tns >= 0) { t.tns = s.try_tns = c.tns != 0; }
    if (c.cfl >= 0) t.cfl = c.cfl != 0;
    if (c.dict >= 0) t.dict = c.dict != 0;
    if (c.shapes >= 0) fp.shapes = c.shapes != 0;
    if (c.fir >= 0) { t.fir = s.try_fir = c.fir != 0; }
    if (c.loop_filter >= 0) fp.lf = fp.lf_freq = fp.lf_map = c.loop_filter != 0;
    t.tmvp = c.preset != Preset::Faster;
    t.rect = c.preset == Preset::Placebo;  // vtest では効果 ±0・時間 +30% のため placebo のみ
    if (c.rect >= 0) t.rect = c.rect != 0;
    fp.alf = c.preset >= Preset::Fast;
    t.mts = s.try_mts = c.preset >= Preset::Slow;  // 実測 +1% / 時間 2 倍
    s.psy = c.tune_psnr ? 0.0 : c.psy_strength;
    s.chroma_weight = c.tune_psnr ? 1.0 : 2.0;
    if (c.mts >= 0) t.mts = s.try_mts = c.mts != 0;
    if (c.alf >= 0) fp.alf = c.alf != 0;
    fp.band_blocks = !c.band_samples;
    fp.band_ns = !c.tune_psnr;
    t.inter_ns = !c.tune_psnr;
    fp.cdef = c.cdef >= 0 ? c.cdef != 0 : c.preset >= Preset::Medium;
    if (c.tmvp >= 0) t.tmvp = c.tmvp != 0;
    t.firb = c.firb >= 0 ? c.firb != 0 : c.preset == Preset::Placebo;  // 9 フレーム試験では効果が雑音 (±0.1 dB) 以下
    t.gs = c.gs >= 0 ? c.gs != 0 : c.preset == Preset::Placebo;  // 実測: RD 混在で -0.03 dB (slow では時間だけ増える)
    // タイル: 既定は placebo 以外 2x2 (並列化のため)。threads は符号化結果に影響しない
    // 実測: 2x2 は 1x1 より 4-7% 効率が落ちるため、medium 以上は 1x1 (速度より効率)
    const int dt = c.preset >= Preset::Medium ? 1 : 2;
    fp.tile_cols = std::clamp(c.tile_cols > 0 ? c.tile_cols : dt, 1, 16);
    fp.tile_rows = std::clamp(c.tile_rows > 0 ? c.tile_rows : dt, 1, 16);
    fp.threads = c.threads > 0 ? c.threads : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    return fp;
}

void write_frame_header(std::vector<uint8_t>& p, const FrameParams& fp) {
    put_u8(p, static_cast<uint32_t>(fp.type));
    put_uv(p, static_cast<uint32_t>(fp.poc));
    put_u8(p, static_cast<uint32_t>(fp.qp));
    put_u8(p, (fp.l2 ? 1u : 0u) | (fp.lossy ? 2u : 0u) | (static_cast<uint32_t>(fp.pqmf_log2) << 2) |
                  (static_cast<uint32_t>(fp.pqmf_filter) << 5));
    put_u8(p, static_cast<uint32_t>(fp.pqmf_log2y));
    put_u8(p, static_cast<uint32_t>(fp.min_log2) | (static_cast<uint32_t>(fp.max_log2) << 4));
    put_u8(p, tools_byte(fp.tools));
    put_u8(p, (fp.shapes ? 1u : 0u) | (fp.band_tools ? 2u : 0u) | (fp.dict_reset ? 4u : 0u) | (fp.lf ? 8u : 0u) |
                  (fp.lf_freq ? 16u : 0u) | (fp.lf_map ? 32u : 0u) | (fp.tools.rect ? 64u : 0u) | (fp.tools.tmvp ? 128u : 0u));
    put_u8(p, static_cast<uint32_t>((fp.tile_cols - 1) | ((fp.tile_rows - 1) << 4)));
    put_u8(p, static_cast<uint32_t>(fp.cqp_off + 32) | (fp.aqp ? 128u : 0u));
    put_u8(p, (fp.alf ? 1u : 0u) | (fp.tools.mts ? 2u : 0u) | (fp.band_blocks ? 4u : 0u) | (fp.band_ns ? 8u : 0u) |
                  (fp.tools.inter_ns ? 16u : 0u) | (fp.cdef ? 32u : 0u) | (fp.tools.gs ? 64u : 0u) | (fp.tools.firb ? 128u : 0u));
    for (int l = 0; l < 2; ++l) {
        put_u8(p, static_cast<uint32_t>(fp.ref_poc[l].size()));
        for (int poc : fp.ref_poc[l]) put_uv(p, static_cast<uint32_t>(poc));
    }
}

bool read_frame_header(ByteReader& br, FrameParams& fp) {
    const uint32_t t = br.u8();
    if (t > 3) return false;
    fp.type = static_cast<FrameType>(t);
    fp.poc = static_cast<int>(br.uv());
    fp.qp = static_cast<int>(br.u8());
    const uint32_t f = br.u8();
    fp.l2 = f & 1; fp.lossy = (f >> 1) & 1; fp.pqmf_log2 = (f >> 2) & 7; fp.pqmf_filter = (f >> 5) & 3;
    fp.pqmf_log2y = static_cast<int>(br.u8());
    if (fp.pqmf_log2y > 4 || (fp.pqmf_log2y && fp.pqmf_log2 == 0)) return false;
    const uint32_t lg = br.u8();
    fp.min_log2 = lg & 15; fp.max_log2 = lg >> 4;
    fp.tools = tools_from_byte(static_cast<uint8_t>(br.u8()));
    const uint32_t f2 = br.u8();
    fp.shapes = f2 & 1; fp.band_tools = (f2 >> 1) & 1; fp.dict_reset = (f2 >> 2) & 1;
    fp.lf = (f2 >> 3) & 1; fp.lf_freq = (f2 >> 4) & 1; fp.lf_map = (f2 >> 5) & 1;
    fp.tools.rect = (f2 >> 6) & 1; fp.tools.tmvp = (f2 >> 7) & 1;
    const uint32_t tl = br.u8();
    fp.tile_cols = static_cast<int>(tl & 15) + 1;
    fp.tile_rows = static_cast<int>(tl >> 4) + 1;
    const uint32_t cq = br.u8();
    fp.aqp = (cq >> 7) & 1;
    fp.cqp_off = static_cast<int>(cq & 127) - 32;
    const uint32_t f3 = br.u8();
    fp.alf = f3 & 1;
    fp.tools.mts = (f3 >> 1) & 1;
    fp.band_blocks = (f3 >> 2) & 1;
    fp.band_ns = (f3 >> 3) & 1;
    fp.tools.inter_ns = (f3 >> 4) & 1;
    fp.cdef = (f3 >> 5) & 1;
    fp.tools.gs = (f3 >> 6) & 1;
    fp.tools.firb = (f3 >> 7) & 1;
    if (fp.cqp_off < -12 || fp.cqp_off > 12) return false;
    for (int l = 0; l < 2; ++l) {
        const uint32_t n = br.u8();
        if (n > 4) return false;
        fp.ref_poc[l].clear();
        for (uint32_t i = 0; i < n; ++i) fp.ref_poc[l].push_back(static_cast<int>(br.uv()));
    }
    if (br.fail || fp.qp > 63 || fp.pqmf_log2 > 4 || fp.min_log2 < 2 || fp.max_log2 > kCtuLog2 || fp.min_log2 > fp.max_log2)
        return false;
    if (fp.type != FrameType::I && fp.ref_poc[0].empty()) return false;
    if (fp.type == FrameType::B && fp.ref_poc[1].empty()) return false;
    return true;
}

}  // namespace

// ---------------- Encoder ----------------
Encoder::Encoder(const VideoInfo& info, const EncoderConfig& cfg) : info_(info), cfg_(cfg), st_(new CodecState) {}
Encoder::~Encoder() = default;

std::vector<uint8_t> Encoder::sequence_header() const {
    std::vector<uint8_t> p, out;
    put_u32(p, kMagic);
    put_u8(p, 2);  // profile
    put_u16(p, static_cast<uint32_t>(info_.width));
    put_u16(p, static_cast<uint32_t>(info_.height));
    put_u8(p, static_cast<uint32_t>(info_.bit_depth));
    put_u8(p, static_cast<uint32_t>(info_.chroma));
    put_u8(p, static_cast<uint32_t>(info_.ct));
    put_uv(p, static_cast<uint32_t>(info_.fps_num));
    put_uv(p, static_cast<uint32_t>(info_.fps_den));
    put_unit(out, UnitType::Seq, p);
    return out;
}

std::vector<uint8_t> Encoder::encode_picture(const Frame& f, int poc, FrameType type, int depth,
                                             const std::vector<const Frame*>& look) {
    FrameParams fp = params_from(cfg_);
    fp.dict_add = cfg_.total_frames != 1;
    if (fp.pqmf_log2 > 0) {
        const PqmfCfg pc = choose_pqmf(f.p[0], fp.pqmf_log2);
        fp.pqmf_log2 = pc.lx; fp.pqmf_log2y = pc.ly; fp.pqmf_filter = pc.filter;
    }
    fp.poc = poc;
    fp.type = type;
    if (type == FrameType::I) {
        fp.dict_reset = true;
    } else {
        // 参照選択: L0 = 過去 (近い順), L1 = 未来 (近い順)
        std::vector<int> past, fut;
        // P はアンカー (I/P) のみ参照、B は全参照可
        for (const auto& p : st_->dpb)
            if (type == FrameType::B || p.anchor) (p.poc < poc ? past : fut).push_back(p.poc);
        std::sort(past.begin(), past.end(), std::greater<int>());
        std::sort(fut.begin(), fut.end());
        int nref = cfg_.refs > 0 ? cfg_.refs
                                 : (cfg_.preset == Preset::Placebo ? 3 : cfg_.preset == Preset::Slow ? 2 : 1);
        nref = std::clamp(nref, 1, 4);
        if (type == FrameType::B) { nref = 1; if (fut.empty()) type = fp.type = FrameType::P; }
        for (int i = 0; i < nref && i < static_cast<int>(past.size()); ++i) fp.ref_poc[0].push_back(past[i]);
        if (fp.ref_poc[0].empty()) { type = fp.type = FrameType::I; fp.dict_reset = true; }
        if (type == FrameType::B) fp.ref_poc[1].push_back(fut[0]);
        // QP カスケード: P +1, B は階層深さ d に応じて +1+d
        // QP カスケード (実測で決定: I -5, P -2, B +1+深さ)
        constexpr int kPOff = -2, kBOff = 1;
        if (type == FrameType::P) fp.qp = std::clamp(fp.qp + kPOff, 0, 63);
        if (type == FrameType::B) fp.qp = std::clamp(fp.qp + kBOff + std::max(1, depth), 0, 63);
    }
    if (type == FrameType::I) {
        fp.ref_poc[0].clear();
        constexpr int kIOff = -5;
        fp.qp = std::clamp(fp.qp + kIOff, 0, 63);
    }
    // 適応 QP: 後続フレーム (このフレームを直接/間接に参照する) との同位置差分が小さい CTU ほど QP を下げる
    const bool aqp_on = cfg_.aqp >= 0 ? cfg_.aqp != 0 : true;
    auto apply_aqp = [&](FrameParams& fp) {
    fp.aqp = false;
    fp.aqp_map.clear();
    // B フレーム: 参照との差が大きい (動きがある) CTU は QP を下げ、動物体の崩れを防ぐ
    if (aqp_on && cfg_.lossy_layer && fp.type == FrameType::B && look.empty()) {
        const Frame* r0 = st_->find(fp.ref_poc[0][0]);
        const Frame* r1 = fp.ref_poc[1].empty() ? nullptr : st_->find(fp.ref_poc[1][0]);
        const Plane& A = f.p[0];
        const int aw = (A.w + kCtu - 1) / kCtu, ah = (A.h + kCtu - 1) / kCtu;
        const double step = qp_step(fp.qp, info_.bit_depth);
        fp.aqp_w = aw;
        fp.aqp_map.assign(static_cast<size_t>(aw) * ah, 0);
        bool any = false;
        for (int cy = 0; cy < ah; ++cy)
            for (int cx = 0; cx < aw; ++cx) {
                // 双方向の同位置差分の小さい方 (動きのない側) を動きの指標とする
                double d0 = 0, d1 = 0;
                int64_t cnt = 0;
                for (int y = cy * kCtu; y < std::min(A.h, (cy + 1) * kCtu); y += 2)
                    for (int x = cx * kCtu; x < std::min(A.w, (cx + 1) * kCtu); x += 2) {
                        d0 += std::abs(A.at(x, y) - r0->p[0].at(x, y));
                        if (r1) d1 += std::abs(A.at(x, y) - r1->p[0].at(x, y));
                        ++cnt;
                    }
                const double d = (r1 ? std::min(d0, d1) : d0) / std::max<int64_t>(1, cnt);
                const int dq = d > 2.0 * step ? -3 : d > 1.0 * step ? -2 : d > 0.5 * step ? -1 : 0;
                fp.aqp_map[static_cast<size_t>(cy) * aw + cx] = static_cast<int8_t>(dq);
                any |= dq != 0;
            }
        fp.aqp = any;
        if (!any) fp.aqp_map.clear();
        return;
    }
    if (aqp_on && cfg_.lossy_layer && !look.empty()) {
        const Plane& A = f.p[0];
        const int aw = (A.w + kCtu - 1) / kCtu, ah = (A.h + kCtu - 1) / kCtu;
        fp.aqp_w = aw;
        fp.aqp_map.assign(static_cast<size_t>(aw) * ah, 0);
        const double step = qp_step(fp.qp, info_.bit_depth);
        constexpr double kAqs = 1.0;  // 実測で決定 (0.5/1.0/1.5 を比較)
        const double strength = std::min(6.0, kAqs * std::log2(1.0 + look.size()));
        bool any = false;
        for (int cy = 0; cy < ah; ++cy)
            for (int cx = 0; cx < aw; ++cx) {
                double d = 0;
                int64_t cnt = 0;
                for (const Frame* L : look)
                    for (int y = cy * kCtu; y < std::min(A.h, (cy + 1) * kCtu); y += 2)
                        for (int x = cx * kCtu; x < std::min(A.w, (cx + 1) * kCtu); x += 2) {
                            d += std::abs(A.at(x, y) - L->p[0].at(x, y));
                            ++cnt;
                        }
                d /= std::max<int64_t>(1, cnt);
                const double stat = std::clamp(1.0 - d / step, 0.0, 1.0);
                const int dq = -static_cast<int>(std::lround(strength * stat));
                fp.aqp_map[static_cast<size_t>(cy) * aw + cx] = static_cast<int8_t>(dq);
                any |= dq != 0;
            }
        fp.aqp = any;
        if (!any) fp.aqp_map.clear();
    }
    };
    apply_aqp(fp);
    BlockUsage usage;
    auto run = [&](FrameParams& par, Frame& rec) {
        EntropyWriter ew;
        SymIO io;
        io.w = &ew;
        io.enc = true;
        TileStreams ts;
        usage = BlockUsage{};
        code_frame(io, info_, par, &f, rec, *st_, ts, &usage);
        std::vector<uint8_t> p;
        write_frame_header(p, par);
        auto bytes = ew.finish();
        put_uv(p, static_cast<uint32_t>(bytes.size()));
        p.insert(p.end(), bytes.begin(), bytes.end());
        put_uv(p, static_cast<uint32_t>(ts.out.size()));
        for (size_t i = 0; i < ts.out.size(); ++i) {  // 最後のタイルの長さは残り全部 (省略)
            if (i + 1 < ts.out.size()) put_uv(p, static_cast<uint32_t>(ts.out[i].size()));
            p.insert(p.end(), ts.out[i].begin(), ts.out[i].end());
        }
        return p;
    };
    Frame rec;
    std::vector<uint8_t> payload;
    bool done = false;
    int trials = 0;
    const bool ssim_mode = cfg_.target_ssim > 0 && cfg_.lossy_layer && !cfg_.l2_lossless;
    if (type != FrameType::I && cfg_.copy_frames && cfg_.lossy_layer && !cfg_.l2_lossless) {
        // COPY 判定: グローバル予測の MSE が量子化雑音相当以下 (目標 SSIM 時は SSIM を満たす) なら COPY
        FrameParams cp = fp;
        cp.type = FrameType::Copy;
        cp.ref_poc[1].clear();
        Frame crec;
        auto pl = run(cp, crec);
        ++trials;
        fp.gm_valid = cp.gm_valid;
        fp.gm_x = cp.gm_x; fp.gm_y = cp.gm_y;
        fp.gm_model = cp.gm_model;
        for (int pi = 0; pi < 3; ++pi) fp.gm_hg[pi] = cp.gm_hg[pi];
        for (int pi = 0; pi < 3; ++pi) { fp.gm_gain[pi] = cp.gm_gain[pi]; fp.gm_off[pi] = cp.gm_off[pi]; }
        const double step = qp_step(fp.qp, info_.bit_depth);
        double se = 0;
        for (size_t i = 0; i < f.p[0].v.size(); ++i) { const double d = f.p[0].v[i] - crec.p[0].v[i]; se += d * d; }
        // 実測の量子化歪み (RDO 後) は Δ²/12 の 1 割未満なので、それと同等以下の時のみ COPY
        constexpr double kCopyK = 0.06;
        const bool ok = ssim_mode ? plane_ssim(f.p[0], crec.p[0], info_.bit_depth) >= cfg_.target_ssim
                                  : se / f.p[0].v.size() <= kCopyK * step * step / 12.0;
        if (ok) { payload = std::move(pl); rec = std::move(crec); done = true; type = FrameType::Copy; }
    }
    if (!done && ssim_mode) {
        // 目標 SSIM: 種別ごとに前回 QP から探索し、条件を満たす最大 QP を選ぶ。
        // 試行ごとに辞書・動き情報の状態を戻し、採用した試行の状態を復元する。
        const Dictionary dict0 = st_->dict;
        const std::vector<Shape> sh0 = st_->last_shapes;
        struct Trial { int qp; double ssim; std::vector<uint8_t> payload; Frame rec; Dictionary dict; std::vector<Shape> sh; std::shared_ptr<MotionField> mf;
                       std::vector<int> rp[2]; BlockUsage usage; };
        std::vector<Trial> tr;
        auto attempt = [&](int q) -> const Trial& {
            for (const Trial& t : tr) if (t.qp == q) return t;
            st_->dict = dict0;
            st_->last_shapes = sh0;
            FrameParams par = fp;
            par.qp = q;
            apply_aqp(par);
            Trial t;
            t.qp = q;
            t.payload = run(par, t.rec);
            ++trials;
            t.ssim = plane_ssim(f.p[0], t.rec.p[0], info_.bit_depth);
            t.dict = st_->dict;
            t.sh = st_->last_shapes;
            t.mf = st_->last_mf;
            t.rp[0] = st_->last_ref_poc[0]; t.rp[1] = st_->last_ref_poc[1];
            t.usage = usage;
            tr.push_back(std::move(t));
            return tr.back();
        };
        const int ti = static_cast<int>(type);
        int q = last_q_[ti] >= 0 ? last_q_[ti] : fp.qp;
        const double T = cfg_.target_ssim;
        if (attempt(q).ssim >= T) {
            // 満たす → QP を 2 ずつ上げ、失敗したら間の +1 を試す
            while (q + 2 <= 63 && tr.size() < 8 && attempt(q + 2).ssim >= T) q += 2;
            if (q + 1 <= 63 && attempt(q + 1).ssim >= T) q += 1;
        } else {
            // 満たさない → 満たすまで 2 ずつ下げ (QP 0 まで)、成功したら間の +1 を試す
            while (q > 0) {
                q = std::max(0, q - 2);
                if (attempt(q).ssim >= T) break;
            }
            if (attempt(q).ssim >= T && attempt(q + 1).ssim >= T) q += 1;
        }
        // 条件を満たす最大 QP (なければ最小 QP の試行)
        const Trial* best = nullptr;
        for (const Trial& t : tr)
            if (t.ssim >= T && (!best || t.qp > best->qp)) best = &t;
        if (!best)
            for (const Trial& t : tr)
                if (!best || t.qp < best->qp) best = &t;
        last_q_[ti] = best->qp;
        fp.qp = best->qp;
        payload = best->payload;
        rec = best->rec;
        st_->dict = best->dict;
        st_->last_shapes = best->sh;
        st_->last_mf = best->mf;
        st_->last_ref_poc[0] = best->rp[0]; st_->last_ref_poc[1] = best->rp[1];
        usage = best->usage;
        done = true;
    }
    if (!done && type == FrameType::I && cfg_.preset >= Preset::Slow && cfg_.lossy_layer && !cfg_.l2_lossless &&
        (fp.tools.tns || fp.tools.ibc || fp.tools.dict || fp.tools.mts)) {
        // フレーム単位の道具 RD 試行 (I のみ): 稀にしか選ばれない道具はフラグの符号量で損をするため、
        // TNS/IBC/辞書/MTS を外した符号化と D + λR を比べ、小さい方を採る
        const Dictionary dict0 = st_->dict;
        const std::vector<Shape> sh0 = st_->last_shapes;
        auto mf0 = st_->last_mf;
        const std::vector<int> rp0[2] = {st_->last_ref_poc[0], st_->last_ref_poc[1]};
        const double lam = 0.57 * std::pow(2.0, (fp.qp - 12) / 3.0) * std::pow(4.0, info_.bit_depth - 8);
        auto jcost = [&](const std::vector<uint8_t>& pl, const Frame& r) {
            double d = 0;
            for (int pi = 0; pi < 3; ++pi)
                for (size_t i = 0; i < f.p[pi].v.size(); ++i) { const double e = f.p[pi].v[i] - r.p[pi].v[i]; d += e * e; }
            return d + lam * 8.0 * static_cast<double>(pl.size());
        };
        payload = run(fp, rec);
        ++trials;
        const double j0 = jcost(payload, rec);
        const Dictionary dict1 = st_->dict;
        const std::vector<Shape> sh1 = st_->last_shapes;
        auto mf1 = st_->last_mf;
        const std::vector<int> rp1[2] = {st_->last_ref_poc[0], st_->last_ref_poc[1]};
        const BlockUsage u1 = usage;
        st_->dict = dict0; st_->last_shapes = sh0; st_->last_mf = mf0; st_->last_ref_poc[0] = rp0[0]; st_->last_ref_poc[1] = rp0[1];
        FrameParams lite = fp;
        lite.tools.tns = lite.tools.ibc = lite.tools.dict = lite.tools.mts = false;
        lite.search.try_tns = lite.search.try_mts = false;
        Frame rec2;
        auto pl2 = run(lite, rec2);
        ++trials;
        if (jcost(pl2, rec2) < j0) {
            payload = std::move(pl2); rec = std::move(rec2); fp = lite;
        } else {
            st_->dict = dict1; st_->last_shapes = sh1; st_->last_mf = mf1; st_->last_ref_poc[0] = rp1[0]; st_->last_ref_poc[1] = rp1[1];
            usage = u1;
        }
        done = true;
    }
    if (!done) { payload = run(fp, rec); ++trials; }
    std::vector<uint8_t> out;
    put_unit(out, UnitType::Frame, payload);
    st_->push(poc, rec, type != FrameType::B);
    FrameStats fs;
    fs.poc = poc;
    fs.type = type;
    fs.bytes = out.size();
    for (int p = 0; p < 3; ++p) fs.psnr[p] = plane_psnr(f.p[p], rec.p[p], info_.bit_depth);
    fs.ssim = plane_ssim(f.p[0], rec.p[0], info_.bit_depth);
    fs.qp = fp.qp;
    fs.trials = trials;
    fs.usage = usage;
    stats_.push_back(fs);
    if (cfg_.keep_recon) recon_[poc] = rec;
    return out;
}

std::vector<uint8_t> Encoder::encode(const Frame& f) {
    const int poc = next_poc_++;
    const bool key = poc == 0 || (cfg_.keyint > 0 && poc % cfg_.keyint == 0);
    int bf = cfg_.bframes >= 0 ? cfg_.bframes : (cfg_.preset == Preset::Faster ? 3 : 7);
    bf = std::clamp(bf, 0, 15);
    if (!cfg_.lossy_layer) bf = 0;
    std::vector<uint8_t> out;
    if (key) {
        // I フレームは後続ミニ GOP を先読みしてから符号化する (適応 QP 用)
        out = flush_pending();
        key_pending_ = true;
    }
    pend_.push_back({poc, f});
    if (static_cast<int>(pend_.size()) >= bf + 1 + (key_pending_ ? 1 : 0)) {
        const auto u = flush_pending();
        out.insert(out.end(), u.begin(), u.end());
    }
    return out;
}

// 保留フレームを符号化: [I] → 最後をアンカー P → 間を二分の階層 B
std::vector<uint8_t> Encoder::flush_pending() {
    std::vector<uint8_t> out;
    if (pend_.empty()) return out;
    std::vector<std::pair<int, Frame>> v;
    v.swap(pend_);
    const bool key = key_pending_;
    key_pending_ = false;
    const int n = static_cast<int>(v.size());
    auto append = [&](const std::vector<uint8_t>& u) { out.insert(out.end(), u.begin(), u.end()); };
    int start = 0;
    if (key) {
        std::vector<const Frame*> look;
        for (int i = 1; i < n; ++i) look.push_back(&v[i].second);
        append(encode_picture(v[0].second, v[0].first, FrameType::I, 0, look));
        start = 1;
        if (n == 1) return out;
    }
    std::vector<const Frame*> look;
    for (int i = start; i < n - 1; ++i) look.push_back(&v[i].second);
    append(encode_picture(v[n - 1].second, v[n - 1].first, FrameType::P, 0, look));
    encode_b_range(v, start - 1, n - 1, 1, out);
    return out;
}

void Encoder::encode_b_range(const std::vector<std::pair<int, Frame>>& v, int a, int b, int depth, std::vector<uint8_t>& out) {
    if (b - a < 2) return;
    const int mid = (a + b) / 2;
    const auto u = encode_picture(v[mid].second, v[mid].first, FrameType::B, depth);
    out.insert(out.end(), u.begin(), u.end());
    encode_b_range(v, a, mid, depth + 1, out);
    encode_b_range(v, mid, b, depth + 1, out);
}

std::vector<uint8_t> Encoder::flush() {
    std::vector<uint8_t> out = flush_pending();
    put_unit(out, UnitType::Eos, {});
    return out;
}

// ---------------- Decoder ----------------
// 1 ユニットを取り出す (メモリ上のストリーム or 入力ストリームから逐次)
bool Decoder::read_unit(uint8_t& type, std::vector<uint8_t>& payload) {
    if (in_) {
        uint8_t h[5];
        if (!in_->read(reinterpret_cast<char*>(h), 5)) return false;
        type = h[0];
        const uint32_t len = (static_cast<uint32_t>(h[1]) << 24) | (static_cast<uint32_t>(h[2]) << 16) |
                             (static_cast<uint32_t>(h[3]) << 8) | h[4];
        if (len > (1u << 30)) { ok_ = false; return false; }
        payload.resize(len);
        if (len && !in_->read(reinterpret_cast<char*>(payload.data()), len)) { ok_ = false; return false; }
        return true;
    }
    if (pos_ + 5 > s_.size()) return false;
    ByteReader br{s_.data() + pos_, s_.size() - pos_};
    type = static_cast<uint8_t>(br.u8());
    const uint32_t len = br.u32();
    if (pos_ + 5 + len > s_.size()) { ok_ = false; return false; }
    payload.assign(s_.data() + pos_ + 5, s_.data() + pos_ + 5 + len);
    pos_ += 5 + len;
    return true;
}

bool Decoder::parse_seq() {
    uint8_t t;
    std::vector<uint8_t> pl;
    if (!read_unit(t, pl) || t != static_cast<uint8_t>(UnitType::Seq)) return false;
    ByteReader br{pl.data(), pl.size()};
    if (br.u32() != kMagic) return false;
    br.u8();
    info_.width = static_cast<int>(br.u16());
    info_.height = static_cast<int>(br.u16());
    info_.bit_depth = static_cast<int>(br.u8());
    info_.chroma = static_cast<ChromaFormat>(br.u8());
    info_.ct = static_cast<ColorTransform>(br.u8());
    info_.fps_num = static_cast<int>(br.uv());
    info_.fps_den = static_cast<int>(br.uv());
    return !br.fail && info_.width > 0 && info_.height > 0 && info_.bit_depth >= 8 && info_.bit_depth <= 16;
}

Decoder::Decoder(const std::vector<uint8_t>& stream, int threads) : s_(stream), st_(new CodecState) {
    threads_ = threads > 0 ? threads : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    ok_ = parse_seq();
}

Decoder::Decoder(std::istream& in, int threads) : st_(new CodecState), in_(&in) {
    threads_ = threads > 0 ? threads : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    ok_ = parse_seq();
}
Decoder::~Decoder() = default;

bool Decoder::decode_unit() {
    uint8_t utype;
    std::vector<uint8_t> buf;
    while (ok_ && !eos_ && read_unit(utype, buf)) {
        const auto type = static_cast<UnitType>(utype);
        const uint32_t len = static_cast<uint32_t>(buf.size());
        const uint8_t* pl = buf.data();
        if (type == UnitType::Eos) { eos_ = true; return false; }
        if (type != UnitType::Frame) continue;
        ByteReader hr{pl, len};
        FrameParams fp;
        if (!read_frame_header(hr, fp)) { ok_ = false; return false; }
        const uint32_t mlen = hr.uv();
        if (hr.fail || hr.pos + mlen > len) { ok_ = false; return false; }
        EntropyReader er(pl + hr.pos, mlen);
        hr.pos += mlen;
        TileStreams ts;
        const uint32_t ntl = hr.uv();
        if (ntl > 4096) { ok_ = false; return false; }
        for (uint32_t i = 0; i < ntl && !hr.fail; ++i) {
            const uint32_t tl = i + 1 < ntl ? hr.uv() : static_cast<uint32_t>(len - std::min<size_t>(len, hr.pos));
            if (hr.fail || hr.pos + tl > len) { ok_ = false; return false; }
            ts.in.push_back({pl + hr.pos, tl});
            hr.pos += tl;
        }
        if (hr.fail) { ok_ = false; return false; }
        fp.threads = threads_;
        SymIO io;
        io.r = &er;
        Frame f;
        code_frame(io, info_, fp, nullptr, f, *st_, ts);
        st_->push(fp.poc, f, fp.type != FrameType::B);
        out_[fp.poc] = std::move(f);
        return true;
    }
    eos_ = true;
    return false;
}

bool Decoder::next(Frame& f) {
    if (!ok_) return false;
    while (out_.find(next_out_) == out_.end()) {
        if (eos_ || !decode_unit()) {
            if (out_.empty()) return false;
            next_out_ = out_.begin()->first;  // 欠番は飛ばす
            break;
        }
    }
    auto it = out_.find(next_out_);
    f = std::move(it->second);
    out_.erase(it);
    ++next_out_;
    return true;
}

}  // namespace fvc
