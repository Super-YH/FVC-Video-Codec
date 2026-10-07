#include "loop_filter.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "fvc/transform.hpp"

namespace fvc {

namespace {
constexpr int kBeta[52] = {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  6,  7,
                           8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 22, 24, 26, 28, 30, 32,
                           34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64};
constexpr int kTc[54] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,  1,  1,  1,  1,  1,  1,  1, 1,
                         2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 8, 9, 10, 11, 13, 14, 16, 18, 20, 22, 24};

int bs_of(const EdgeInfo& e, const MotionField* mf, bool chroma, int chroma_shift, int ax, int ay, int bx, int by) {
    const size_t ia = static_cast<size_t>(ay / 4) * e.w4 + ax / 4, ib = static_cast<size_t>(by / 4) * e.w4 + bx / 4;
    if (e.leaf[ia] == e.leaf[ib]) return 0;
    const uint8_t fa = e.flags[ia], fb = e.flags[ib];
    if ((fa | fb) & 1) return 2;
    if (chroma) return 0;  // 色差: イントラ境界のみ
    if ((fa | fb) & 2) return 1;
    if (!mf) return 0;
    const int lax = std::min(ax << chroma_shift, mf->w4 * 4 - 1), lay = std::min(ay << chroma_shift, mf->h4 * 4 - 1);
    const int lbx = std::min(bx << chroma_shift, mf->w4 * 4 - 1), lby = std::min(by << chroma_shift, mf->h4 * 4 - 1);
    const MotionInfo& a = mf->at(lax, lay);
    const MotionInfo& b = mf->at(lbx, lby);
    if (a.dir != b.dir) return 1;
    for (int l = 0; l < 2; ++l)
        if ((a.dir >> l) & 1)
            if (a.ref[l] != b.ref[l] || std::abs(a.mvx[l] - b.mvx[l]) >= 4 || std::abs(a.mvy[l] - b.mvy[l]) >= 4) return 1;
    return 0;
}

// 1 本の 4 画素エッジ区間 (P 側 p[-1..-4], Q 側 q[0..3])。get/set は区間内の行 k (0..3) と境界からのオフセット i で指す
template <class Px>
void filter_luma_segment(Px px, int beta, int tc, int32_t lo, int32_t hi) {
    auto P = [&](int k, int i) -> int32_t& { return px(k, -1 - i); };
    auto Q = [&](int k, int i) -> int32_t& { return px(k, i); };
    const int dp0 = std::abs(P(0, 2) - 2 * P(0, 1) + P(0, 0)), dq0 = std::abs(Q(0, 2) - 2 * Q(0, 1) + Q(0, 0));
    const int dp3 = std::abs(P(3, 2) - 2 * P(3, 1) + P(3, 0)), dq3 = std::abs(Q(3, 2) - 2 * Q(3, 1) + Q(3, 0));
    const int d = dp0 + dq0 + dp3 + dq3;
    if (d >= beta) return;
    auto strong_ok = [&](int k, int dpq) {
        return 2 * dpq < (beta >> 2) && std::abs(P(k, 3) - P(k, 0)) + std::abs(Q(k, 0) - Q(k, 3)) < (beta >> 3) &&
               std::abs(P(k, 0) - Q(k, 0)) < ((5 * tc + 1) >> 1);
    };
    const bool strong = strong_ok(0, dp0 + dq0) && strong_ok(3, dp3 + dq3);
    const bool dEp = (dp0 + dp3) < ((beta + (beta >> 1)) >> 3), dEq = (dq0 + dq3) < ((beta + (beta >> 1)) >> 3);
    for (int k = 0; k < 4; ++k) {
        const int32_t p0 = P(k, 0), p1 = P(k, 1), p2 = P(k, 2), p3 = P(k, 3);
        const int32_t q0 = Q(k, 0), q1 = Q(k, 1), q2 = Q(k, 2), q3 = Q(k, 3);
        if (strong) {
            auto c = [&](int32_t v, int32_t o) { return std::clamp(v, o - 2 * tc, o + 2 * tc); };
            P(k, 0) = c((p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3, p0);
            P(k, 1) = c((p2 + p1 + p0 + q0 + 2) >> 2, p1);
            P(k, 2) = c((2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3, p2);
            Q(k, 0) = c((p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3, q0);
            Q(k, 1) = c((p0 + q0 + q1 + q2 + 2) >> 2, q1);
            Q(k, 2) = c((p0 + q0 + q1 + 3 * q2 + 2 * q3 + 4) >> 3, q2);
        } else {
            int delta = (9 * (q0 - p0) - 3 * (q1 - p1) + 8) >> 4;
            if (std::abs(delta) >= tc * 10) continue;
            delta = std::clamp(delta, -tc, tc);
            P(k, 0) = std::clamp(p0 + delta, lo, hi);
            Q(k, 0) = std::clamp(q0 - delta, lo, hi);
            if (dEp) P(k, 1) = std::clamp(p1 + std::clamp((((p2 + p0 + 1) >> 1) - p1 + delta) >> 1, -(tc >> 1), tc >> 1), lo, hi);
            if (dEq) Q(k, 1) = std::clamp(q1 + std::clamp((((q2 + q0 + 1) >> 1) - q1 - delta) >> 1, -(tc >> 1), tc >> 1), lo, hi);
        }
    }
}
}  // namespace

void deblock_plane(Plane& p, const EdgeInfo& e, const MotionField* mf, bool chroma, int chroma_shift, int qp,
                   int bit_depth, int beta_off, int tc_off, int32_t lo, int32_t hi) {
    const int bdsh = bit_depth - 8;
    for (int dir = 0; dir < 2; ++dir) {  // 0: 垂直エッジ (水平方向にフィルタ), 1: 水平エッジ
        for (int y = 0; y < p.h; y += 4)
            for (int x = 0; x < p.w; x += 4) {
                const int ex = x, ey = y;
                if (dir == 0 && (ex == 0 || (ex & 7))) continue;
                if (dir == 1 && (ey == 0 || (ey & 7))) continue;
                const int bs = dir == 0 ? bs_of(e, mf, chroma, chroma_shift, ex - 4, ey, ex, ey) : bs_of(e, mf, chroma, chroma_shift, ex, ey - 4, ex, ey);
                if (bs == 0) continue;
                const int beta = kBeta[std::clamp(qp + beta_off, 0, 51)] << bdsh;
                const int tc = kTc[std::clamp(qp + 2 * (bs - 1) + tc_off, 0, 53)] << bdsh;
                if (tc == 0) continue;
                if (chroma) {
                    // 色差 (BS=2 のみ): p0, q0 のみ修正
                    for (int k = 0; k < 4; ++k) {
                        int32_t& p0 = dir == 0 ? p.at(ex - 1, ey + k) : p.at(ex + k, ey - 1);
                        int32_t& q0 = dir == 0 ? p.at(ex, ey + k) : p.at(ex + k, ey);
                        const int32_t p1 = dir == 0 ? p.at(ex - 2, ey + k) : p.at(ex + k, ey - 2);
                        const int32_t q1 = dir == 0 ? p.at(ex + 1, ey + k) : p.at(ex + k, ey + 1);
                        const int d = std::clamp((((q0 - p0) * 4) + p1 - q1 + 4) >> 3, -tc, tc);
                        p0 = std::clamp(p0 + d, lo, hi);
                        q0 = std::clamp(q0 - d, lo, hi);
                    }
                    continue;
                }
                if (dir == 0)
                    filter_luma_segment([&](int k, int i) -> int32_t& { return p.at(ex + i, ey + k); }, beta, tc, lo, hi);
                else
                    filter_luma_segment([&](int k, int i) -> int32_t& { return p.at(ex + k, ey + i); }, beta, tc, lo, hi);
            }
    }
}

void freq_gain_filter(const Plane& in, Plane& out, double step, int mu_q, int32_t lo, int32_t hi) {
    out = in;
    if (mu_q <= 0) return;
    constexpr int N = 8, hop = 4;
    constexpr double kPi = 3.14159265358979323846;
    double w[N];
    for (int n = 0; n < N; ++n) w[n] = std::sin(kPi * (n + 0.5) / N);
    const double mu = mu_q / 4.0, sigma2 = step * step / 12.0;
    std::vector<double> acc(in.v.size(), 0.0), wsum(in.v.size(), 0.0);
    double blk[N * N], c[N * N], r[N * N];
    for (int by = -hop; by < in.h; by += hop)
        for (int bx = -hop; bx < in.w; bx += hop) {
            for (int y = 0; y < N; ++y)
                for (int x = 0; x < N; ++x)
                    blk[y * N + x] = w[x] * w[y] * in.at(std::clamp(bx + x, 0, in.w - 1), std::clamp(by + y, 0, in.h - 1));
            forward_2d(TxType::DCT2, TxType::DCT2, blk, N, N, c);
            for (int i = 1; i < N * N; ++i) {
                const double c2 = c[i] * c[i];
                c[i] *= c2 > 0 ? std::max(0.0, 1.0 - mu * sigma2 / c2) : 0.0;
            }
            inverse_2d(TxType::DCT2, TxType::DCT2, c, N, N, r);
            for (int y = 0; y < N; ++y)
                for (int x = 0; x < N; ++x) {
                    const int X = bx + x, Y = by + y;
                    if (X < 0 || Y < 0 || X >= in.w || Y >= in.h) continue;
                    const size_t i = static_cast<size_t>(Y) * in.w + X;
                    acc[i] += w[x] * w[y] * r[y * N + x];
                    wsum[i] += w[x] * w[x] * w[y] * w[y];
                }
        }
    for (size_t i = 0; i < acc.size(); ++i)
        out.v[i] = std::clamp(static_cast<int32_t>(std::lround(acc[i] / wsum[i])), lo, hi);
}

void code_loop_filter(SymIO& io, CMModel& m, Plane& R, const Plane* org, const EdgeInfo& e, const MotionField* mf,
                      bool chroma, int chroma_shift, int qp, int bit_depth, double step, double lambda, bool allow_freq,
                      bool allow_map, int32_t lo, int32_t hi) {
    LoopFilterParams lp;
    const Plane pre = R;
    Plane D = R;
    // 9.1 デブロック (常時、オフセットは 0 を伝送)
    lp.beta_off = io.sint(m, 0, 0, 0);
    lp.tc_off = io.sint(m, 1, 0, 0);
    if (std::abs(lp.beta_off) > 12 || std::abs(lp.tc_off) > 12) throw std::runtime_error("corrupt stream: deblock offsets");
    deblock_plane(D, e, mf, chroma, chroma_shift, qp, bit_depth, lp.beta_off, lp.tc_off, lo, hi);
    auto sse = [&](const Plane& a) {
        double s = 0;
        for (size_t i = 0; i < a.v.size(); ++i) { const double d = a.v[i] - org->v[i]; s += d * d; }
        return s;
    };
    // 9.2 周波数ゲイン: 符号器は μ 候補から最小 SSE を選ぶ
    Plane F;
    if (allow_freq) {
        if (io.enc) {
            double best = sse(D);
            int bm = 0;
            for (int mq : {1, 2, 4, 8}) {
                Plane t;
                freq_gain_filter(D, t, step, mq, lo, hi);
                const double s = sse(t);
                if (s < best) { best = s; bm = mq; F = std::move(t); }
            }
            lp.mu_q = bm;
        }
        lp.mu_q = static_cast<int>(io.uint(m, 2, 0, static_cast<uint32_t>(lp.mu_q)));
        if (lp.mu_q > 15) throw std::runtime_error("corrupt stream: mu");
        if (!io.enc || lp.mu_q == 0) freq_gain_filter(D, F, step, lp.mu_q, lo, hi);
    } else {
        F = D;
    }
    // 9.3 画素ゲインマップ
    const int cw = (R.w + kGainCell - 1) / kGainCell, ch = (R.h + kGainCell - 1) / kGainCell;
    lp.gamma.assign(static_cast<size_t>(cw) * ch, 64);
    int use_map = 0;
    if (allow_map) {
        if (io.enc) {
            double gain = 0;
            for (int cy = 0; cy < ch; ++cy)
                for (int cx = 0; cx < cw; ++cx) {
                    double num = 0, den = 0, e1 = 0;
                    for (int y = cy * kGainCell; y < std::min(R.h, (cy + 1) * kGainCell); ++y)
                        for (int x = cx * kGainCell; x < std::min(R.w, (cx + 1) * kGainCell); ++x) {
                            const double dd = F.at(x, y) - pre.at(x, y), ee = org->at(x, y) - pre.at(x, y);
                            num += dd * ee; den += dd * dd;
                            const double r1 = ee - dd;
                            e1 += r1 * r1;
                        }
                    int g = den > 0 ? std::clamp(static_cast<int>(std::lround(num / den * 64.0)), 0, 128) : 64;
                    const double gg = g / 64.0;
                    // γ=1 との差 (SSE 減少量)
                    const double e2 = e1 - 2 * (gg - 1) * (num - den) + (gg - 1) * (gg - 1) * den;
                    gain += e1 - e2;
                    lp.gamma[cy * cw + cx] = g;
                }
            use_map = gain > lambda * 3.0 * cw * ch;
        }
        use_map = io.bit(m, 3, 0, 0, use_map);
        if (use_map) {
            int prev = 64;
            for (auto& g : lp.gamma) {
                g = prev + io.sint(m, 4, static_cast<uint32_t>(std::min(std::abs(prev - 64) >> 3, 7)), g - prev);
                if (g < 0 || g > 128) throw std::runtime_error("corrupt stream: gamma");
                prev = g;
            }
        } else {
            std::fill(lp.gamma.begin(), lp.gamma.end(), 64);
        }
    }
    for (int y = 0; y < R.h; ++y)
        for (int x = 0; x < R.w; ++x) {
            const int g = lp.gamma[(y / kGainCell) * cw + x / kGainCell];
            const int32_t d = F.at(x, y) - pre.at(x, y);
            R.at(x, y) = std::clamp(pre.at(x, y) + static_cast<int32_t>((static_cast<int64_t>(g) * d + (d >= 0 ? 32 : -32)) / 64), lo, hi);
        }
}

}  // namespace fvc
