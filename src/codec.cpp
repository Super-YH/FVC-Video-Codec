#include "fvc/codec.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "fvc/entropy.hpp"
#include "fvc/pqmf.hpp"
#include "fvc/transform.hpp"
#include "block_coder.hpp"

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
    int min_log2 = 2, max_log2 = kCtuLog2;
    Tools tools;    // ビットストリームで伝送
    Search search;  // 符号器のみ
};

uint8_t tools_byte(const Tools& t) {
    return static_cast<uint8_t>((t.ibc ? 1 : 0) | (t.tns ? 2 : 0) | (t.e8 ? 4 : 0) | (t.cfl ? 8 : 0) | (t.nf ? 16 : 0) |
                                (t.all_angular ? 32 : 0));
}
Tools tools_from_byte(uint8_t b) {
    Tools t;
    t.ibc = b & 1; t.tns = b & 2; t.e8 = b & 4; t.cfl = b & 8; t.nf = b & 16; t.all_angular = b & 32;
    return t;
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

// フレームのペイロード (rANS) を符号化/復号。org==nullptr なら復号。
void code_frame(SymIO& io, const VideoInfo& info, const FrameParams& fp, const Frame* org, Frame& rec) {
    Models md;
    rec.p.resize(3);
    Plane luma_rec;  // 輝度の非可逆再構成 (パディング済み)
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
                Plane lds;
                if (pi > 0 && fp.tools.cfl) lds = luma_at_chroma(luma_rec, info, W, H);
                const double lambda = 0.57 * std::pow(2.0, (fp.qp - 12) / 3.0) * std::pow(4.0, info.bit_depth - 8);
                BlockCoder bc(&R, org ? &opad : nullptr, (pi > 0 && fp.tools.cfl) ? &lds : nullptr, pi, lo, hi, step,
                              lambda, fp.min_log2, fp.max_log2, fp.tools, fp.search);
                for (int cy = 0; cy < H; cy += kCtu)
                    for (int cx = 0; cx < W; cx += kCtu) bc.code_ctu(io, md, cx, cy, kCtu);
            }
        }
        if (pi == 0) luma_rec = R;
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
    Tools& t = fp.tools;
    Search& s = fp.search;
    t.cfl = true;
    t.nf = c.psy;
    switch (c.preset) {
    case Preset::Faster:
        fp.min_log2 = 3; fp.max_log2 = 5; t.all_angular = false; s.rd_modes = 1;
        break;
    case Preset::Fast:
        fp.min_log2 = 3; fp.max_log2 = 6; s.rd_modes = 2;
        break;
    case Preset::Medium:
        fp.min_log2 = 2; fp.max_log2 = 6; s.rd_modes = 3;
        break;
    case Preset::Slow:
        fp.min_log2 = 2; fp.max_log2 = 6; s.rd_modes = 6;
        t.tns = t.ibc = true; s.try_tns = true; s.ibc_range = 32;
        break;
    case Preset::Placebo:
        fp.min_log2 = 2; fp.max_log2 = 6; s.rd_modes = 35;
        t.tns = t.e8 = t.ibc = true; s.try_tns = s.try_e8 = true; s.ibc_range = 96;
        break;
    }
    if (c.ibc >= 0) t.ibc = c.ibc != 0;
    if (c.e8 >= 0) { t.e8 = s.try_e8 = c.e8 != 0; }
    if (c.tns >= 0) { t.tns = s.try_tns = c.tns != 0; }
    if (c.cfl >= 0) t.cfl = c.cfl != 0;
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
    put_u8(p, tools_byte(fp.tools));
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
        fp.tools = tools_from_byte(pl[4]);
        if (fp.pqmf_log2 > 4 || fp.min_log2 < 2 || fp.max_log2 > kCtuLog2 || fp.min_log2 > fp.max_log2) { ok_ = false; return false; }
        EntropyReader er(pl + 5, len - 5);
        SymIO io;
        io.r = &er;
        code_frame(io, info_, fp, nullptr, f);
        return true;
    }
    return false;
}

}  // namespace fvc
