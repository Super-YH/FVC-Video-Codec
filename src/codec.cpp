#include "fvc/codec.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "fvc/entropy.hpp"
#include "fvc/pqmf.hpp"
#include "fvc/transform.hpp"

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

// ---------------- 符号器/復号器共有の構文 I/O ----------------
// w != nullptr なら符号化 (値をそのまま返す)、そうでなければ復号した値を返す。
struct SymIO {
    EntropyWriter* w = nullptr;
    EntropyReader* r = nullptr;
    int bit(CMModel& m, uint32_t node, uint32_t a, uint32_t b, int v) {
        if (w) { w->bit(m, node, a, b, v); return v; }
        return r->bit(m, node, a, b);
    }
    int32_t sint(CMModel& m, uint32_t a, uint32_t b, int32_t v) {
        if (w) { w->sint(m, a, b, v); return v; }
        return r->sint(m, a, b);
    }
    uint32_t uint(CMModel& m, uint32_t a, uint32_t b, uint32_t v) {
        if (w) { w->uint(m, a, b, v); return v; }
        return r->uint(m, a, b);
    }
};

struct Models {
    CMModel split, mode, cbf, last, coef_y, coef_c, band_ll, band_hi, lossless, l2;
};

// ---------------- 走査順 (対角) ----------------
const std::vector<int>& diag_scan(int log2s) {
    static std::array<std::vector<int>, 11> cache;
    auto& sc = cache[log2s];
    if (sc.empty()) {
        const int s = 1 << log2s;
        for (int d = 0; d <= 2 * (s - 1); ++d)
            for (int v = std::min(d, s - 1); v >= 0; --v) {
                const int u = d - v;
                if (u < s) sc.push_back(v * s + u);  // v: 行 (垂直周波数), u: 列
            }
    }
    return sc;
}

// ---------------- イントラ予測 ----------------
enum IntraMode { kDC = 0, kPlanar = 1, kVer = 2, kHor = 3, kNumModes = 4 };

void intra_pred(const Plane& rec, int x0, int y0, int s, int mode, int32_t mid, int32_t* pred) {
    std::vector<int32_t> top(2 * s), left(2 * s);
    const bool ht = y0 > 0, hl = x0 > 0;
    for (int i = 0; i < 2 * s; ++i) {
        const int xx = std::min(x0 + i, rec.w - 1), yy = std::min(y0 + i, rec.h - 1);
        top[i] = ht ? rec.at(i < s ? x0 + i : xx, y0 - 1) : (hl ? rec.at(x0 - 1, y0) : mid);
        left[i] = hl ? rec.at(x0 - 1, i < s ? y0 + i : yy) : (ht ? rec.at(x0, y0 - 1) : mid);
    }
    // 右上/左下は未復号の可能性があるため s 番目以降は端値で代用
    for (int i = s; i < 2 * s; ++i) { top[i] = top[s - 1]; left[i] = left[s - 1]; }
    switch (mode) {
    case kDC: {
        int64_t sum = 0;
        for (int i = 0; i < s; ++i) sum += top[i] + left[i];
        const int32_t dc = static_cast<int32_t>((sum + s) / (2 * s));
        for (int i = 0; i < s * s; ++i) pred[i] = dc;
        break;
    }
    case kPlanar: {
        const int sh = [s] { int l = 0; while ((1 << l) < s) ++l; return l; }();
        for (int y = 0; y < s; ++y)
            for (int x = 0; x < s; ++x) {
                const int64_t h = static_cast<int64_t>(s - 1 - x) * left[y] + static_cast<int64_t>(x + 1) * top[s];
                const int64_t v = static_cast<int64_t>(s - 1 - y) * top[x] + static_cast<int64_t>(y + 1) * left[s];
                pred[y * s + x] = static_cast<int32_t>((h + v + s) >> (sh + 1));
            }
        break;
    }
    case kVer:
        for (int y = 0; y < s; ++y) for (int x = 0; x < s; ++x) pred[y * s + x] = top[x];
        break;
    default:
        for (int y = 0; y < s; ++y) for (int x = 0; x < s; ++x) pred[y * s + x] = left[y];
        break;
    }
}

// ---------------- 量子化 ----------------
double qp_step(int qp, int bit_depth) { return std::pow(2.0, (qp - 4) / 6.0) * (1 << (bit_depth - 8)); }

int32_t quant_dz(double c, double step, double rnd) {
    const double a = std::abs(c) / step;
    const int32_t q = static_cast<int32_t>(a + rnd);
    return c < 0 ? -q : q;
}

// 係数レートの近似 (RD 探索用)
double approx_bits(const std::vector<int32_t>& q, const std::vector<int>& scan) {
    int last = -1;
    for (int i = static_cast<int>(scan.size()) - 1; i >= 0; --i) if (q[scan[i]]) { last = i; break; }
    if (last < 0) return 1.0;
    double b = 2.0 + 2.0 * std::log2(2.0 + last);
    for (int i = 0; i <= last; ++i) {
        const int32_t a = std::abs(q[scan[i]]);
        b += a ? 2.0 + 2.0 * std::log2(1.0 + a) : 0.6;
    }
    return b;
}

// ---------------- ブロック符号化 ----------------
struct PlaneCoder {
    Plane* rec;              // パディング済み再構成
    const Plane* org;        // 符号器のみ (パディング済み原画)
    int plane;
    int32_t lo, hi, mid;
    double step, lambda, rnd;
    int min_log2, max_log2;
    int nmodes;
    std::vector<std::vector<int8_t>> split_map, mode_map;  // [log2] -> grid

    int grid_w(int l) const { return rec->w >> l; }
    int8_t& split_at(int x, int y, int l) { return split_map[l][(y >> l) * grid_w(l) + (x >> l)]; }
    int8_t& mode_at(int x, int y, int l) { return mode_map[l][(y >> l) * grid_w(l) + (x >> l)]; }

    TxType tx_for(int s) const { return s == 4 ? TxType::DST7 : TxType::DCT2; }

    // 予測+変換+量子化 → q (符号器)
    void analyze_block(int x0, int y0, int log2s, int mode, std::vector<int32_t>& pred, std::vector<int32_t>& q) {
        const int s = 1 << log2s;
        pred.resize(s * s);
        q.assign(s * s, 0);
        intra_pred(*rec, x0, y0, s, mode, mid, pred.data());
        std::vector<double> r(s * s), c(s * s);
        for (int y = 0; y < s; ++y)
            for (int x = 0; x < s; ++x) r[y * s + x] = org->at(x0 + x, y0 + y) - pred[y * s + x];
        forward_2d(tx_for(s), tx_for(s), r.data(), s, s, c.data());
        for (int i = 0; i < s * s; ++i) q[i] = quant_dz(c[i], step, rnd);
    }

    // q を逆量子化・逆変換して再構成に書き込む (符号器/復号器共通)
    void reconstruct(int x0, int y0, int log2s, const std::vector<int32_t>& pred, const std::vector<int32_t>& q) {
        const int s = 1 << log2s;
        bool any = false;
        for (int32_t v : q) if (v) { any = true; break; }
        std::vector<double> c(s * s), r(s * s, 0.0);
        if (any) {
            for (int i = 0; i < s * s; ++i) c[i] = q[i] * step;
            inverse_2d(tx_for(s), tx_for(s), c.data(), s, s, r.data());
        }
        for (int y = 0; y < s; ++y)
            for (int x = 0; x < s; ++x) {
                const int32_t v = pred[y * s + x] + static_cast<int32_t>(std::lround(r[y * s + x]));
                rec->at(x0 + x, y0 + y) = std::clamp(v, lo, hi);
            }
    }

    double block_sse(int x0, int y0, int s) const {
        double e = 0;
        for (int y = 0; y < s; ++y)
            for (int x = 0; x < s; ++x) { const double d = org->at(x0 + x, y0 + y) - rec->at(x0 + x, y0 + y); e += d * d; }
        return e;
    }

    void save(int x0, int y0, int s, std::vector<int32_t>& buf) const {
        buf.resize(s * s);
        for (int y = 0; y < s; ++y) std::memcpy(&buf[y * s], &rec->v[(y0 + y) * rec->w + x0], s * sizeof(int32_t));
    }
    void restore(int x0, int y0, int s, const std::vector<int32_t>& buf) {
        for (int y = 0; y < s; ++y) std::memcpy(&rec->v[(y0 + y) * rec->w + x0], &buf[y * s], s * sizeof(int32_t));
    }

    // RD 探索 (符号器): 分割とモードを決め、選んだ再構成を rec に残す
    double rd_node(int x0, int y0, int l) {
        const int s = 1 << l;
        const auto& scan = diag_scan(l);
        std::vector<int32_t> before, pred, q, best_rec;
        save(x0, y0, s, before);
        double best = 1e300;
        int best_mode = 0;
        for (int m = 0; m < nmodes; ++m) {
            analyze_block(x0, y0, l, m, pred, q);
            reconstruct(x0, y0, l, pred, q);
            const double j = block_sse(x0, y0, s) + lambda * (approx_bits(q, scan) + 2.0 + (l > min_log2));
            if (j < best) { best = j; best_mode = m; save(x0, y0, s, best_rec); }
            restore(x0, y0, s, before);
        }
        mode_at(x0, y0, l) = static_cast<int8_t>(best_mode);
        restore(x0, y0, s, best_rec);
        if (l <= min_log2) return best;
        // 分割候補: 早期打ち切り (葉コストが極小なら分割しない)
        if (best < lambda * 4.0) { split_at(x0, y0, l) = 0; return best; }
        restore(x0, y0, s, before);
        const int h = s / 2;
        double js = lambda * 1.0;
        js += rd_node(x0, y0, l - 1);
        js += rd_node(x0 + h, y0, l - 1);
        js += rd_node(x0, y0 + h, l - 1);
        js += rd_node(x0 + h, y0 + h, l - 1);
        if (best <= js) { split_at(x0, y0, l) = 0; restore(x0, y0, s, best_rec); return best; }
        split_at(x0, y0, l) = 1;
        return js;
    }

    // 構文の符号化/復号 + 再構成 (符号器/復号器共通)
    void code_node(SymIO& io, Models& md, int x0, int y0, int l) {
        int split = 0;
        if (l > min_log2) split = io.bit(md.split, 0, static_cast<uint32_t>(l), static_cast<uint32_t>(plane), io.w ? split_at(x0, y0, l) : 0);
        if (split) {
            const int h = 1 << (l - 1);
            code_node(io, md, x0, y0, l - 1);
            code_node(io, md, x0 + h, y0, l - 1);
            code_node(io, md, x0, y0 + h, l - 1);
            code_node(io, md, x0 + h, y0 + h, l - 1);
            return;
        }
        const int s = 1 << l;
        const uint32_t pc = plane ? 1u : 0u;
        int mode = io.w ? mode_at(x0, y0, l) : 0;
        mode = io.bit(md.mode, 0, l, pc, mode >> 1) << 1;
        mode |= io.bit(md.mode, 1 + (mode >> 1), l, pc, io.w ? (mode_at(x0, y0, l) & 1) : 0);
        std::vector<int32_t> pred, q;
        if (io.w) analyze_block(x0, y0, l, mode, pred, q);
        else { pred.resize(s * s); q.assign(s * s, 0); intra_pred(*rec, x0, y0, s, mode, mid, pred.data()); }
        const auto& scan = diag_scan(l);
        int last = -1;
        if (io.w) for (int i = s * s - 1; i >= 0; --i) if (q[scan[i]]) { last = i; break; }
        const int cbf = io.bit(md.cbf, 0, l, pc * 4 + static_cast<uint32_t>(mode), last >= 0);
        if (cbf) {
            last = static_cast<int>(io.uint(md.last, l, pc, static_cast<uint32_t>(last)));
            if (last >= s * s) throw std::runtime_error("corrupt stream: last");
            CMModel& cm = plane ? md.coef_c : md.coef_y;
            for (int i = 0; i <= last; ++i) {
                const int pos = scan[i];
                const int u = pos & (s - 1), v = pos >> l;
                const int32_t n1 = i > 0 ? std::abs(q[scan[i - 1]]) : 0;
                const int32_t n2 = i > 1 ? std::abs(q[scan[i - 2]]) : 0;
                const uint32_t a = static_cast<uint32_t>(std::min(n1 + n2, 15)) | (static_cast<uint32_t>(std::min(u + v, 15)) << 4);
                const uint32_t b = static_cast<uint32_t>(l) * 4 + (i == last ? 1u : 0u) + (i == 0 ? 2u : 0u);
                if (i == last) {
                    // 最終係数は非ゼロ確定: |c|-1 と符号
                    const int32_t cv = io.w ? q[pos] : 0;
                    const uint32_t mag = io.uint(cm, a, b, io.w ? static_cast<uint32_t>(std::abs(cv)) - 1u : 0u) + 1u;
                    const int neg = io.bit(cm, 1, a, b, cv < 0);
                    if (!io.w) q[pos] = neg ? -static_cast<int32_t>(mag) : static_cast<int32_t>(mag);
                } else {
                    q[pos] = io.sint(cm, a, b, q[pos]);
                }
            }
        } else if (io.w) {
            std::fill(q.begin(), q.end(), 0);
        }
        reconstruct(x0, y0, l, pred, q);
    }
};

// ---------------- PQMF 帯域符号化 ----------------
void code_bands(SymIO& io, Models& md, int pqmf_log2, const Plane* org, Plane& rec, double step, double rnd,
                int32_t lo, int32_t hi) {
    const int M = 1 << pqmf_log2, W = rec.w, H = rec.h;
    Pqmf2D fb(M, M);
    const int32_t off = (lo + hi + 1) / 2;
    std::vector<std::vector<double>> bands;
    if (io.w) {
        std::vector<double> x(static_cast<size_t>(W) * H);
        for (size_t i = 0; i < x.size(); ++i) x[i] = org->v[i] - off;
        fb.analyze(x, W, H, bands);
    } else {
        bands.assign(M * M, std::vector<double>(static_cast<size_t>(W / M) * (H / M), 0.0));
    }
    const int bw = W / M, bh = H / M;
    for (int ky = 0; ky < M; ++ky)
        for (int kx = 0; kx < M; ++kx) {
            auto& b = bands[ky * M + kx];
            // 合成後の画素領域でステップが step 相当になるよう正規化 + 高域ほど粗く
            const double st = step / fb.band_norm(kx, ky) * (1.0 + 0.08 * (kx + ky));
            std::vector<int32_t> q(b.size(), 0);
            const bool ll = kx == 0 && ky == 0;
            for (int y = 0; y < bh; ++y)
                for (int x = 0; x < bw; ++x) {
                    const size_t i = static_cast<size_t>(y) * bw + x;
                    const int32_t qw = x ? q[i - 1] : 0, qn = y ? q[i - bw] : 0, qnw = (x && y) ? q[i - bw - 1] : 0;
                    if (ll) {
                        int32_t p;
                        if (!x && !y) p = 0; else if (!y) p = qw; else if (!x) p = qn;
                        else { const int32_t mx = std::max(qw, qn), mn = std::min(qw, qn); p = qnw >= mx ? mn : qnw <= mn ? mx : qw + qn - qnw; }
                        const uint32_t act = static_cast<uint32_t>(std::min(std::abs(qw - qnw) + std::abs(qn - qnw), 31));
                        const int32_t d = io.sint(md.band_ll, act, 0, io.w ? quant_dz(b[i], st, 0.5) - p : 0);
                        q[i] = p + d;
                    } else {
                        const uint32_t a = static_cast<uint32_t>(std::min(std::abs(qw) + std::abs(qn) + std::abs(qnw), 31));
                        q[i] = io.sint(md.band_hi, a, static_cast<uint32_t>(std::min(kx + ky, 31)), io.w ? quant_dz(b[i], st, rnd) : 0);
                    }
                    b[i] = q[i] * st;
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

struct FrameParams {
    int qp = 32;
    bool lossy = true, l2 = false;
    int pqmf_log2 = 0;
    int min_log2 = 2, max_log2 = kCtuLog2, nmodes = kNumModes;
};

// フレームのペイロード (rANS) を符号化/復号。org==nullptr なら復号。
void code_frame(SymIO& io, const VideoInfo& info, const FrameParams& fp, const Frame* org, Frame& rec) {
    Models md;
    rec.p.resize(3);
    for (int pi = 0; pi < 3; ++pi) {
        const int w = pi ? info.chroma_w() : info.width, h = pi ? info.chroma_h() : info.height;
        int32_t lo, hi;
        plane_range(info, pi, lo, hi);
        const int32_t mid = (lo + hi + 1) / 2;
        const int W = (w + kCtu - 1) / kCtu * kCtu, H = (h + kCtu - 1) / kCtu * kCtu;
        Plane opad;
        if (org) opad = pad_plane(org->p[pi], W, H);
        Plane R(W, H, 0);
        if (fp.lossy) {
            const double step = qp_step(fp.qp, info.bit_depth);
            if (fp.pqmf_log2 > 0) {
                code_bands(io, md, fp.pqmf_log2, org ? &opad : nullptr, R, step, 1.0 / 3.0, lo, hi);
            } else {
                PlaneCoder pc;
                pc.rec = &R; pc.org = org ? &opad : nullptr; pc.plane = pi;
                pc.lo = lo; pc.hi = hi; pc.mid = mid;
                pc.step = step;
                pc.lambda = 0.57 * std::pow(2.0, (fp.qp - 12) / 3.0) * std::pow(4.0, info.bit_depth - 8);
                pc.rnd = 1.0 / 3.0;
                pc.min_log2 = fp.min_log2; pc.max_log2 = fp.max_log2; pc.nmodes = fp.nmodes;
                pc.split_map.resize(kCtuLog2 + 1); pc.mode_map.resize(kCtuLog2 + 1);
                for (int l = 0; l <= kCtuLog2; ++l) {
                    pc.split_map[l].assign(static_cast<size_t>(W >> l) * (H >> l), 0);
                    pc.mode_map[l].assign(static_cast<size_t>(W >> l) * (H >> l), 0);
                }
                for (int cy = 0; cy < H; cy += kCtu)
                    for (int cx = 0; cx < W; cx += kCtu) {
                        // CTU を max_log2 のブロックに固定分割
                        for (int by = cy; by < cy + kCtu; by += 1 << fp.max_log2)
                            for (int bx = cx; bx < cx + kCtu; bx += 1 << fp.max_log2) {
                                if (io.w) pc.rd_node(bx, by, fp.max_log2);
                                pc.code_node(io, md, bx, by, fp.max_log2);
                            }
                    }
            }
        }
        Plane out(w, h);
        for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) out.at(x, y) = R.at(x, y);
        if (fp.l2 || !fp.lossy) code_lossless(io, md, pi, w, h, org ? &org->p[pi] : nullptr, out, fp.lossy, mid);
        rec.p[pi] = std::move(out);
    }
}

FrameParams params_from(const EncoderConfig& c) {
    FrameParams fp;
    fp.qp = std::clamp(c.qp, 0, 63);
    fp.lossy = c.lossy_layer;
    fp.l2 = c.l2_lossless;
    fp.pqmf_log2 = std::clamp(c.pqmf_log2, 0, 4);
    switch (c.preset) {
    case Preset::Faster:  fp.min_log2 = 3; fp.max_log2 = 5; fp.nmodes = 2; break;
    case Preset::Fast:    fp.min_log2 = 3; fp.max_log2 = 6; fp.nmodes = 4; break;
    case Preset::Medium:  fp.min_log2 = 2; fp.max_log2 = 6; fp.nmodes = 4; break;
    case Preset::Slow:    fp.min_log2 = 2; fp.max_log2 = 6; fp.nmodes = 4; break;
    case Preset::Placebo: fp.min_log2 = 2; fp.max_log2 = 6; fp.nmodes = 4; break;
    }
    return fp;
}

}  // namespace

// ---------------- Encoder ----------------
Encoder::Encoder(const VideoInfo& info, const EncoderConfig& cfg) : info_(info), cfg_(cfg) {}

std::vector<uint8_t> Encoder::sequence_header() const {
    std::vector<uint8_t> p, out;
    put_u32(p, kMagic);
    put_u8(p, 1);  // profile
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

std::vector<uint8_t> Encoder::encode(const Frame& f, Frame* recon) {
    const FrameParams fp = params_from(cfg_);
    EntropyWriter ew;
    SymIO io;
    io.w = &ew;
    Frame rec;
    code_frame(io, info_, fp, &f, rec);
    std::vector<uint8_t> p, out;
    put_u8(p, 0);  // frame_type = I
    put_u8(p, static_cast<uint32_t>(fp.qp));
    put_u8(p, (fp.l2 ? 1u : 0u) | (fp.lossy ? 2u : 0u) | (static_cast<uint32_t>(fp.pqmf_log2) << 2));
    put_u8(p, static_cast<uint32_t>(fp.min_log2) | (static_cast<uint32_t>(fp.max_log2) << 4));
    put_u8(p, static_cast<uint32_t>(fp.nmodes));
    auto bytes = ew.finish();
    p.insert(p.end(), bytes.begin(), bytes.end());
    put_unit(out, UnitType::Frame, p);
    if (recon) *recon = std::move(rec);
    return out;
}

std::vector<uint8_t> Encoder::end_of_stream() const {
    std::vector<uint8_t> out;
    put_unit(out, UnitType::Eos, {});
    return out;
}

// ---------------- Decoder ----------------
Decoder::Decoder(const std::vector<uint8_t>& stream) : s_(stream) {
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

bool Decoder::next(Frame& f) {
    while (ok_ && pos_ + 5 <= s_.size()) {
        ByteReader br{s_.data() + pos_, s_.size() - pos_};
        const auto type = static_cast<UnitType>(br.u8());
        const uint32_t len = br.u32();
        if (pos_ + 5 + len > s_.size()) { ok_ = false; return false; }
        const uint8_t* pl = s_.data() + pos_ + 5;
        pos_ += 5 + len;
        if (type == UnitType::Eos) return false;
        if (type != UnitType::Frame) continue;
        if (len < 5) { ok_ = false; return false; }
        FrameParams fp;
        if (pl[0] != 0) { ok_ = false; return false; }
        fp.qp = pl[1];
        fp.l2 = pl[2] & 1;
        fp.lossy = (pl[2] >> 1) & 1;
        fp.pqmf_log2 = (pl[2] >> 2) & 7;
        fp.min_log2 = pl[3] & 15;
        fp.max_log2 = pl[3] >> 4;
        fp.nmodes = pl[4];
        if (fp.pqmf_log2 > 4 || fp.min_log2 < 2 || fp.max_log2 > kCtuLog2 || fp.min_log2 > fp.max_log2 ||
            fp.nmodes < 1 || fp.nmodes > kNumModes) { ok_ = false; return false; }
        EntropyReader er(pl + 5, len - 5);
        SymIO io;
        io.r = &er;
        code_frame(io, info_, fp, nullptr, f);
        return true;
    }
    return false;
}

}  // namespace fvc
