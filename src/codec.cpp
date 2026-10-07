#include "fvc/codec.hpp"

#include <algorithm>
#include <array>
#include <cmath>
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

struct ByteReader {
    const uint8_t* d; size_t n, pos = 0;
    bool fail = false;
    uint32_t u8() { if (pos >= n) { fail = true; return 0; } return d[pos++]; }
    uint32_t u16() { uint32_t a = u8(); return (a << 8) | u8(); }
    uint32_t u32() { uint32_t a = u16(); return (a << 16) | u16(); }
};

void put_unit(std::vector<uint8_t>& out, UnitType t, const std::vector<uint8_t>& payload) {
    put_u8(out, static_cast<uint8_t>(t));
    put_u32(out, static_cast<uint32_t>(payload.size()));
    out.insert(out.end(), payload.begin(), payload.end());
}

// ---------------- PQMF 帯域符号化 (§3, 帯域間差分 §3.3, ノイズ置換 §5.6) ----------------
// band_mode: 0 = 通常 (量子化値を符号化), 1 = 帯域間差分 (左/上の隣接帯域を鏡像参照, ゲイン a/64), 2 = ノイズ置換 (RMS のみ)
void code_bands(SymIO& io, Models& md, int pqmf_log2, const Plane* org, Plane& rec, double step, double rnd,
                int32_t lo, int32_t hi, bool band_tools, bool psy, uint64_t seed) {
    const int M = 1 << pqmf_log2, W = rec.w, H = rec.h;
    Pqmf2D fb(M, M);
    const int32_t off = (lo + hi + 1) / 2;
    std::vector<std::vector<double>> bands, src;
    if (io.w) {
        std::vector<double> x(static_cast<size_t>(W) * H);
        for (size_t i = 0; i < x.size(); ++i) x[i] = org->v[i] - off;
        fb.analyze(x, W, H, bands);
        src = bands;
    } else {
        bands.assign(M * M, std::vector<double>(static_cast<size_t>(W / M) * (H / M), 0.0));
    }
    const int bw = W / M, bh = H / M;
    for (int ky = 0; ky < M; ++ky)
        for (int kx = 0; kx < M; ++kx) {
            const int k = ky * M + kx;
            auto& b = bands[k];
            const double st = step / fb.band_norm(kx, ky) * (1.0 + 0.08 * (kx + ky));
            const bool ll = kx == 0 && ky == 0;
            // 参照帯域 (再構成済み): 左隣 or 上隣。奇数帯域のスペクトル鏡像は (-1)^n 変調で補正
            const int rk = kx > 0 ? k - 1 : (ky > 0 ? k - M : -1);
            const bool horiz = kx > 0;
            auto refv = [&](int x, int y) {
                const double v = bands[rk][static_cast<size_t>(y) * bw + x];
                return ((horiz ? x : y) & 1) ? -v : v;
            };
            int mode = 0, gain_q = 0, rms_q = 0;
            if (io.w && band_tools && !ll) {
                double e = 0;
                for (double v : src[k]) e += v * v;
                const double rms = std::sqrt(e / src[k].size());
                if (rk >= 0) {
                    double num = 0, den = 0;
                    for (int y = 0; y < bh; ++y)
                        for (int x = 0; x < bw; ++x) { const double r = refv(x, y); num += r * src[k][y * bw + x]; den += r * r; }
                    if (den > 0) {
                        gain_q = std::clamp(static_cast<int>(std::lround(num / den * 64.0)), -128, 128);
                        const double red = num * num / den;
                        if (gain_q != 0 && red > 0.2 * e && e > 4.0 * st * st * src[k].size() * 0.05) mode = 1;
                    }
                }
                if (mode == 0 && psy && rms < 0.5 * st && rms > 0.15 * st) {
                    mode = 2;
                    rms_q = std::clamp(static_cast<int>(std::lround(rms / st * 32.0)), 1, 63);
                }
            }
            if (band_tools && !ll) {
                mode = static_cast<int>(io.uint(md.band_mode, 0, static_cast<uint32_t>(std::min(kx + ky, 15)), static_cast<uint32_t>(mode)));
                if (mode > 2 || (mode == 1 && rk < 0)) throw std::runtime_error("corrupt stream: band mode");
                if (mode == 1) gain_q = io.sint(md.band_mode, 1, 0, gain_q);
                if (mode == 2) rms_q = static_cast<int>(io.uint(md.band_mode, 2, 0, static_cast<uint32_t>(rms_q)));
                if (std::abs(gain_q) > 128 || rms_q > 63) throw std::runtime_error("corrupt stream: band params");
            }
            if (mode == 2) {
                SplitMix64 rng(seed * 1315423911ull + static_cast<uint64_t>(k));
                const double amp = rms_q / 32.0 * st;
                for (auto& v : b) v = (rng.next() >> 63) ? amp : -amp;
                continue;
            }
            std::vector<double> pr(b.size(), 0.0);
            if (mode == 1)
                for (int y = 0; y < bh; ++y)
                    for (int x = 0; x < bw; ++x) pr[y * bw + x] = gain_q / 64.0 * refv(x, y);
            std::vector<int32_t> q(b.size(), 0);
            for (int y = 0; y < bh; ++y)
                for (int x = 0; x < bw; ++x) {
                    const size_t i = static_cast<size_t>(y) * bw + x;
                    const int32_t qw = x ? q[i - 1] : 0, qn = y ? q[i - bw] : 0, qnw = (x && y) ? q[i - bw - 1] : 0;
                    if (ll) {
                        int32_t p;
                        if (!x && !y) p = 0; else if (!y) p = qw; else if (!x) p = qn;
                        else { const int32_t mx = std::max(qw, qn), mn = std::min(qw, qn); p = qnw >= mx ? mn : qnw <= mn ? mx : qw + qn - qnw; }
                        const uint32_t act = static_cast<uint32_t>(std::min(std::abs(qw - qnw) + std::abs(qn - qnw), 31));
                        const int32_t d = io.sint(md.band_ll, act, 0, io.w ? quant_dz(src[k][i], st, 0.5) - p : 0);
                        q[i] = p + d;
                    } else {
                        const uint32_t a = static_cast<uint32_t>(std::min(std::abs(qw) + std::abs(qn) + std::abs(qnw), 31));
                        q[i] = io.sint(md.band_hi, a, static_cast<uint32_t>(std::min(kx + ky, 31)) + (mode == 1 ? 32u : 0u),
                                       io.w ? quant_dz(src[k][i] - pr[i], st, rnd) : 0);
                    }
                    b[i] = pr[i] + q[i] * st;
                }
        }
    std::vector<double> y;
    fb.synthesize(bands, W, H, y);
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
    int min_log2 = 2, max_log2 = kCtuLog2;
    Tools tools;               // ビットストリームで伝送
    bool shapes = false;       // 図形レイヤ (I のみ)
    bool band_tools = false;   // 帯域間差分/ノイズ置換
    bool dict_reset = false;
    bool lf = true, lf_freq = false, lf_map = false;  // ループフィルタ (§9)
    int tile_cols = 1, tile_rows = 1;                  // タイル分割 (CTU 単位で均等)
    int threads = 1;                                   // 符号器/復号器のスレッド数 (ビットストリームに影響しない)
    // 符号器: グローバルパラメータのキャッシュ (COPY 判定と本符号化で共有)
    mutable bool gm_valid = false;
    mutable int gm_x = 0, gm_y = 0, gm_gain[3] = {64, 64, 64}, gm_off[3] = {0, 0, 0};
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

struct Picture { int poc = 0; Frame f; };

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
    static constexpr size_t kMaxDpb = 6;
    const Frame* find(int poc) const {
        for (const auto& p : dpb) if (p.poc == poc) return &p.f;
        return nullptr;
    }
    void push(int poc, const Frame& f) {
        dpb.push_back({poc, f});
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

// フレームのペイロード (rANS) を符号化/復号。org==nullptr なら復号。
void code_frame(SymIO& io, const VideoInfo& info, const FrameParams& fp, const Frame* org, Frame& rec, CodecState& st,
                TileStreams& ts) {
    Models md;
    rec.p.resize(3);
    if (fp.dict_reset) st.dict.reset_dynamic();
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
    int gmv_x = 0, gmv_y = 0, gain_q[3] = {64, 64, 64}, offs[3] = {0, 0, 0};
    if (inter) {
        if (io.w && fp.gm_valid) {
            gmv_x = fp.gm_x; gmv_y = fp.gm_y;
            for (int pi = 0; pi < 3; ++pi) { gain_q[pi] = fp.gm_gain[pi]; offs[pi] = fp.gm_off[pi]; }
        } else if (io.w) {
            estimate_global_motion(org->p[0], refs[0][0]->p[0], gmv_x, gmv_y);
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
            }
            fp.gm_valid = true;
            fp.gm_x = gmv_x; fp.gm_y = gmv_y;
            for (int pi = 0; pi < 3; ++pi) { fp.gm_gain[pi] = gain_q[pi]; fp.gm_off[pi] = offs[pi]; }
        }
        gmv_x = io.sint(md.global, 0, 0, gmv_x);
        gmv_y = io.sint(md.global, 1, 0, gmv_y);
        for (int pi = 0; pi < 3; ++pi) {
            gain_q[pi] = 64 + io.sint(md.global, 2, static_cast<uint32_t>(pi), gain_q[pi] - 64);
            offs[pi] = io.sint(md.global, 3, static_cast<uint32_t>(pi), offs[pi]);
            if (gain_q[pi] < 0 || gain_q[pi] > 256 || std::abs(gmv_x) > 32000 || std::abs(gmv_y) > 32000)
                throw std::runtime_error("corrupt stream: global params");
        }
    }
    MotionField mf;
    Plane luma_rec;
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
            ic.l0[0].gain_q = gain_q[pi];
            ic.l0[0].off = offs[pi];
            if (pi == 0) mf.init(W, H);
            ic.mf = &mf;
            ic.chroma_shift = pi ? cs : 0;
            ic.gmv_x = gmv_x; ic.gmv_y = gmv_y;
        }
        Plane out(w, h);
        if (fp.type == FrameType::Copy) {
            // COPY: グローバル予測そのもの (§7.7)
            MotionInfo mi; mi.dir = 1; mi.mvx[0] = static_cast<int16_t>(gmv_x); mi.mvy[0] = static_cast<int16_t>(gmv_y);
            inter_predict(mi, ic.l0, nullptr, 0, 0, w, h, pi ? cs : 0, lo, hi, out.v.data());
            rec.p[pi] = std::move(out);
            continue;
        }
        Plane opad;
        if (org) opad = pad_plane(org->p[pi], W, H);
        Plane R(W, H, 0);
        if (fp.lossy) {
            const double step = qp_step(fp.qp, info.bit_depth);
            if (fp.pqmf_log2 > 0 && !inter) {
                code_bands(io, md, fp.pqmf_log2, org ? &opad : nullptr, R, step, 1.0 / 3.0, lo, hi, fp.band_tools, fp.psy,
                           static_cast<uint64_t>(fp.poc) * 3 + pi);
            } else {
                Plane lds;
                if (pi > 0 && fp.tools.cfl) lds = luma_at_chroma(luma_rec, info, W, H);
                const double lambda = 0.57 * std::pow(2.0, (fp.qp - 12) / 3.0) * std::pow(4.0, info.bit_depth - 8);
                // 図形レイヤ (§4): 輝度のみ、I フレームのみ。符号器は RD で採否を決める
                std::vector<Shape> shapes;
                Plane S(W, H, 0);
                bool use_shapes = false;
                if (fp.shapes && pi == 0 && !inter) {
                    if (io.w) {
                        shapes = fit_shapes(opad, 32);
                        if (!shapes.empty()) {
                            render_shapes(shapes, S);
                            Plane d = opad;
                            for (size_t i = 0; i < d.v.size(); ++i) d.v[i] -= S.v[i];
                            Plane r1(W, H, 0), r2(W, H, 0);
                            BlockCoder b1(&r1, &opad, nullptr, pi, lo, hi, step, lambda, fp.min_log2, fp.max_log2, fp.tools, fp.search);
                            BlockCoder b2(&r2, &d, nullptr, pi, lo - hi, hi * 2, step, lambda, fp.min_log2, fp.max_log2, fp.tools, fp.search);
                            double j1 = 0, j2 = lambda * shapes_bits_estimate(shapes);
                            for (int cy = 0; cy < H; cy += kCtu)
                                for (int cx = 0; cx < W; cx += kCtu) { j1 += b1.rd_ctu(cx, cy, kCtu); j2 += b2.rd_ctu(cx, cy, kCtu); }
                            if (j2 >= j1) shapes.clear();
                        }
                    }
                    code_shapes(io, md.shape, shapes);
                    use_shapes = !shapes.empty();
                    S = Plane(W, H, 0);
                    if (use_shapes) render_shapes(shapes, S);
                }
                Plane target;
                if (use_shapes && org) { target = opad; for (size_t i = 0; i < target.v.size(); ++i) target.v[i] -= S.v[i]; }
                const int32_t blo = use_shapes ? lo - hi : lo, bhi = use_shapes ? hi * 2 : hi;
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
                            bhi, step, lambda, fp.min_log2, fp.max_log2, fp.tools, fp.search, inter ? &ic : nullptr, &st.dict, x0,
                            y0, x1 - x0, y1 - y0, t << 24);
                        tmd[t] = std::make_unique<Models>();
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
                if (fp.lf) {
                    code_loop_filter(io, md.lf, R, org ? &opad : nullptr, einfo, inter ? &mf : nullptr, pi > 0, pi ? cs : 0, fp.qp,
                                     info.bit_depth, step, lambda, fp.lf_freq, fp.lf_map, lo, hi);
                }
            }
        }
        if (pi == 0) luma_rec = R;
        for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) out.at(x, y) = R.at(x, y);
        if (fp.l2 || !fp.lossy) code_lossless(io, md, pi, w, h, org ? &org->p[pi] : nullptr, out, fp.lossy, mid);
        rec.p[pi] = std::move(out);
    }
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
            for (size_t i = 0; i < cand.size() && i < 16; ++i) adds.push_back(cand[i].second);
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
        fp.lf_map = true; s.inter_skip_intra = true; s.max_rd_cands = 2; s.skip_split_on_skip = true; s.approx_subpel = true;
        break;
    case Preset::Medium:
        fp.min_log2 = 2; fp.max_log2 = 6; s.rd_modes = 3; s.me_range = 8;
        fp.band_tools = true; fp.lf_freq = fp.lf_map = true;
        break;
    case Preset::Slow:
        fp.min_log2 = 2; fp.max_log2 = 6; s.rd_modes = 6; s.me_range = 16;
        t.tns = t.ibc = t.dict = true; s.try_tns = true; s.ibc_range = 32;
        fp.band_tools = true; fp.lf_freq = fp.lf_map = true;
        break;
    case Preset::Placebo:
        fp.min_log2 = 2; fp.max_log2 = 6; s.rd_modes = 35; s.me_range = 32;
        t.tns = t.e8 = t.ibc = t.dict = true; s.try_tns = s.try_e8 = true; s.ibc_range = 96;
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
    // タイル: 既定は placebo 以外 2x2 (並列化のため)。threads は符号化結果に影響しない
    const int dt = c.preset == Preset::Placebo ? 1 : 2;
    fp.tile_cols = std::clamp(c.tile_cols > 0 ? c.tile_cols : dt, 1, 16);
    fp.tile_rows = std::clamp(c.tile_rows > 0 ? c.tile_rows : dt, 1, 16);
    fp.threads = c.threads > 0 ? c.threads : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    return fp;
}

void write_frame_header(std::vector<uint8_t>& p, const FrameParams& fp) {
    put_u8(p, static_cast<uint32_t>(fp.type));
    put_u32(p, static_cast<uint32_t>(fp.poc));
    put_u8(p, static_cast<uint32_t>(fp.qp));
    put_u8(p, (fp.l2 ? 1u : 0u) | (fp.lossy ? 2u : 0u) | (static_cast<uint32_t>(fp.pqmf_log2) << 2));
    put_u8(p, static_cast<uint32_t>(fp.min_log2) | (static_cast<uint32_t>(fp.max_log2) << 4));
    put_u8(p, tools_byte(fp.tools));
    put_u8(p, (fp.shapes ? 1u : 0u) | (fp.band_tools ? 2u : 0u) | (fp.dict_reset ? 4u : 0u) | (fp.lf ? 8u : 0u) |
                  (fp.lf_freq ? 16u : 0u) | (fp.lf_map ? 32u : 0u));
    put_u8(p, static_cast<uint32_t>((fp.tile_cols - 1) | ((fp.tile_rows - 1) << 4)));
    for (int l = 0; l < 2; ++l) {
        put_u8(p, static_cast<uint32_t>(fp.ref_poc[l].size()));
        for (int poc : fp.ref_poc[l]) put_u32(p, static_cast<uint32_t>(poc));
    }
}

bool read_frame_header(ByteReader& br, FrameParams& fp) {
    const uint32_t t = br.u8();
    if (t > 3) return false;
    fp.type = static_cast<FrameType>(t);
    fp.poc = static_cast<int>(br.u32());
    fp.qp = static_cast<int>(br.u8());
    const uint32_t f = br.u8();
    fp.l2 = f & 1; fp.lossy = (f >> 1) & 1; fp.pqmf_log2 = (f >> 2) & 7;
    const uint32_t lg = br.u8();
    fp.min_log2 = lg & 15; fp.max_log2 = lg >> 4;
    fp.tools = tools_from_byte(static_cast<uint8_t>(br.u8()));
    const uint32_t f2 = br.u8();
    fp.shapes = f2 & 1; fp.band_tools = (f2 >> 1) & 1; fp.dict_reset = (f2 >> 2) & 1;
    fp.lf = (f2 >> 3) & 1; fp.lf_freq = (f2 >> 4) & 1; fp.lf_map = (f2 >> 5) & 1;
    const uint32_t tl = br.u8();
    fp.tile_cols = static_cast<int>(tl & 15) + 1;
    fp.tile_rows = static_cast<int>(tl >> 4) + 1;
    for (int l = 0; l < 2; ++l) {
        const uint32_t n = br.u8();
        if (n > 4) return false;
        fp.ref_poc[l].clear();
        for (uint32_t i = 0; i < n; ++i) fp.ref_poc[l].push_back(static_cast<int>(br.u32()));
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
    put_u32(p, static_cast<uint32_t>(info_.fps_num));
    put_u32(p, static_cast<uint32_t>(info_.fps_den));
    put_unit(out, UnitType::Seq, p);
    return out;
}

std::vector<uint8_t> Encoder::encode_picture(const Frame& f, int poc, FrameType type) {
    FrameParams fp = params_from(cfg_);
    fp.poc = poc;
    fp.type = type;
    if (type == FrameType::I) {
        fp.dict_reset = true;
    } else {
        // 参照選択: L0 = 過去 (近い順), L1 = 未来 (近い順)
        std::vector<int> past, fut;
        for (const auto& p : st_->dpb) (p.poc < poc ? past : fut).push_back(p.poc);
        std::sort(past.begin(), past.end(), std::greater<int>());
        std::sort(fut.begin(), fut.end());
        int nref = cfg_.refs > 0 ? cfg_.refs
                                 : (cfg_.preset == Preset::Placebo ? 3 : cfg_.preset == Preset::Slow ? 2 : 1);
        nref = std::clamp(nref, 1, 4);
        if (type == FrameType::B) { nref = 1; if (fut.empty()) type = fp.type = FrameType::P; }
        for (int i = 0; i < nref && i < static_cast<int>(past.size()); ++i) fp.ref_poc[0].push_back(past[i]);
        if (fp.ref_poc[0].empty()) { type = fp.type = FrameType::I; fp.dict_reset = true; }
        if (type == FrameType::B) fp.ref_poc[1].push_back(fut[0]);
        constexpr int kPOff = 1, kBOff = 3;
        if (type == FrameType::P) fp.qp = std::min(63, fp.qp + kPOff);
        if (type == FrameType::B) fp.qp = std::min(63, fp.qp + kBOff);
    }
    if (type == FrameType::I) fp.ref_poc[0].clear();
    auto run = [&](FrameParams& par, Frame& rec) {
        EntropyWriter ew;
        SymIO io;
        io.w = &ew;
        io.enc = true;
        TileStreams ts;
        code_frame(io, info_, par, &f, rec, *st_, ts);
        std::vector<uint8_t> p;
        write_frame_header(p, par);
        auto bytes = ew.finish();
        put_u32(p, static_cast<uint32_t>(bytes.size()));
        p.insert(p.end(), bytes.begin(), bytes.end());
        put_u16(p, static_cast<uint32_t>(ts.out.size()));
        for (const auto& t : ts.out) {
            put_u32(p, static_cast<uint32_t>(t.size()));
            p.insert(p.end(), t.begin(), t.end());
        }
        return p;
    };
    Frame rec;
    std::vector<uint8_t> payload;
    bool done = false;
    if (type != FrameType::I && cfg_.copy_frames && cfg_.lossy_layer && !cfg_.l2_lossless) {
        // COPY 判定: グローバル予測の MSE が量子化雑音相当以下なら COPY (辞書状態は変化しない)
        FrameParams cp = fp;
        cp.type = FrameType::Copy;
        cp.ref_poc[1].clear();
        Frame crec;
        auto pl = run(cp, crec);
        fp.gm_valid = cp.gm_valid;
        fp.gm_x = cp.gm_x; fp.gm_y = cp.gm_y;
        for (int pi = 0; pi < 3; ++pi) { fp.gm_gain[pi] = cp.gm_gain[pi]; fp.gm_off[pi] = cp.gm_off[pi]; }
        const double step = qp_step(fp.qp, info_.bit_depth);
        double se = 0;
        for (size_t i = 0; i < f.p[0].v.size(); ++i) { const double d = f.p[0].v[i] - crec.p[0].v[i]; se += d * d; }
        if (se / f.p[0].v.size() <= 0.25 * step * step / 12.0) { payload = std::move(pl); rec = std::move(crec); done = true; type = FrameType::Copy; }
    }
    if (!done) payload = run(fp, rec);
    std::vector<uint8_t> out;
    put_unit(out, UnitType::Frame, payload);
    st_->push(poc, rec);
    FrameStats fs;
    fs.poc = poc;
    fs.type = type;
    fs.bytes = out.size();
    for (int p = 0; p < 3; ++p) fs.psnr[p] = plane_psnr(f.p[p], rec.p[p], info_.bit_depth);
    stats_.push_back(fs);
    if (cfg_.keep_recon) recon_[poc] = rec;
    return out;
}

std::vector<uint8_t> Encoder::encode(const Frame& f) {
    const int poc = next_poc_++;
    const bool key = poc == 0 || (cfg_.keyint > 0 && poc % cfg_.keyint == 0);
    int bf = cfg_.bframes >= 0 ? cfg_.bframes : (cfg_.preset >= Preset::Medium ? 1 : 0);
    if (!cfg_.lossy_layer) bf = 0;
    std::vector<uint8_t> out;
    auto append = [&](const std::vector<uint8_t>& u) { out.insert(out.end(), u.begin(), u.end()); };
    if (key) {
        if (have_pending_) { append(encode_picture(pending_, pending_poc_, FrameType::P)); have_pending_ = false; }
        append(encode_picture(f, poc, FrameType::I));
        return out;
    }
    if (bf == 0) { append(encode_picture(f, poc, FrameType::P)); return out; }
    if (!have_pending_) { pending_ = f; pending_poc_ = poc; have_pending_ = true; return out; }
    // P B 並べ替え: 新フレームを P として先に符号化し、保留フレームを B に
    append(encode_picture(f, poc, FrameType::P));
    append(encode_picture(pending_, pending_poc_, FrameType::B));
    have_pending_ = false;
    return out;
}

std::vector<uint8_t> Encoder::flush() {
    std::vector<uint8_t> out;
    if (have_pending_) {
        out = encode_picture(pending_, pending_poc_, FrameType::P);
        have_pending_ = false;
    }
    put_unit(out, UnitType::Eos, {});
    return out;
}

// ---------------- Decoder ----------------
Decoder::Decoder(const std::vector<uint8_t>& stream, int threads) : s_(stream), st_(new CodecState) {
    threads_ = threads > 0 ? threads : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    ByteReader br{s_.data(), s_.size()};
    if (br.u8() != static_cast<uint8_t>(UnitType::Seq)) return;
    const uint32_t len = br.u32();
    const size_t start = br.pos;
    if (br.u32() != kMagic) return;
    br.u8();
    info_.width = static_cast<int>(br.u16());
    info_.height = static_cast<int>(br.u16());
    info_.bit_depth = static_cast<int>(br.u8());
    info_.chroma = static_cast<ChromaFormat>(br.u8());
    info_.ct = static_cast<ColorTransform>(br.u8());
    info_.fps_num = static_cast<int>(br.u32());
    info_.fps_den = static_cast<int>(br.u32());
    if (br.fail || info_.width <= 0 || info_.height <= 0 || info_.bit_depth < 8 || info_.bit_depth > 16) return;
    pos_ = start + len;
    ok_ = pos_ <= s_.size();
}
Decoder::~Decoder() = default;

bool Decoder::decode_unit() {
    while (ok_ && !eos_ && pos_ + 5 <= s_.size()) {
        ByteReader br{s_.data() + pos_, s_.size() - pos_};
        const auto type = static_cast<UnitType>(br.u8());
        const uint32_t len = br.u32();
        if (pos_ + 5 + len > s_.size()) { ok_ = false; return false; }
        const uint8_t* pl = s_.data() + pos_ + 5;
        pos_ += 5 + len;
        if (type == UnitType::Eos) { eos_ = true; return false; }
        if (type != UnitType::Frame) continue;
        ByteReader hr{pl, len};
        FrameParams fp;
        if (!read_frame_header(hr, fp)) { ok_ = false; return false; }
        const uint32_t mlen = hr.u32();
        if (hr.fail || hr.pos + mlen > len) { ok_ = false; return false; }
        EntropyReader er(pl + hr.pos, mlen);
        hr.pos += mlen;
        TileStreams ts;
        const uint32_t ntl = hr.u16();
        for (uint32_t i = 0; i < ntl && !hr.fail; ++i) {
            const uint32_t tl = hr.u32();
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
        st_->push(fp.poc, f);
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
