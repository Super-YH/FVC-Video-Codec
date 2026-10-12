#include "loop_filter.hpp"
#include "lattice_adam.hpp"
#include "parallel.hpp"
#include "prof.hpp"

#include <algorithm>
#include <map>
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

// QP > 51 (B 階層の高 QP) でもステップに比例して閾値を伸ばす (HEVC 表の外挿)
int beta_of(int q) { q = std::clamp(q, 0, 75); return q <= 51 ? kBeta[q] : 64 + 2 * (q - 51); }
int tc_of(int q) {
    q = std::clamp(q, 0, 77);
    return q <= 53 ? kTc[q] : static_cast<int>(std::lround(24.0 * std::exp2((q - 53) / 6.0)));
}

int bs_of(const EdgeInfo& e, const MotionField* mf, bool chroma, int chroma_shift, int ax, int ay, int bx, int by) {
    const size_t ia = static_cast<size_t>(ay / 4) * e.w4 + ax / 4, ib = static_cast<size_t>(by / 4) * e.w4 + bx / 4;
    if (e.leaf[ia] == e.leaf[ib]) return 0;
    const uint8_t fa = e.flags[ia], fb = e.flags[ib];
    if ((fa | fb) & 1) return 2;
    if ((fa | fb) & 2) return 1;  // 残差のある境界 (色差も)
    if (chroma) return 0;
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
void filter_luma_segment(Px px, int beta, int tc, int32_t lo, int32_t hi, int nlong) {
    auto P = [&](int k, int i) -> int32_t& { return px(k, -1 - i); };
    auto Q = [&](int k, int i) -> int32_t& { return px(k, i); };
    // 長タップ (VVC 型): 両側が 16 画素以上の平坦なブロックなら境界から n 画素を線形ランプで滑らかにする
    if (nlong >= 5) {
        const int n = nlong;
        auto flat = [&](int k) {
            const int dp = std::abs(P(k, 2) - 2 * P(k, 1) + P(k, 0)), dq = std::abs(Q(k, 2) - 2 * Q(k, 1) + Q(k, 0));
            const int sp = std::abs(P(k, 0) - P(k, 3)) + std::abs(P(k, 3) - P(k, n)), sq = std::abs(Q(k, 0) - Q(k, 3)) + std::abs(Q(k, 3) - Q(k, n));
            return 2 * (dp + dq) < (beta >> 4) && sp + sq < ((3 * beta) >> 5) && std::abs(P(k, 0) - Q(k, 0)) < ((5 * tc + 1) >> 1);
        };
        if (flat(0) && flat(3)) {
            // 重み f_i = 64 (2(n-i)-1) / 2n (n=7 で VVC の {59,...,5} と一致)、修正量上限 tt_i·tc/2
            int f[15], tt[15];
            for (int i = 0; i < n; ++i) {
                f[i] = (64 * (2 * (n - i) - 1) + n) / (2 * n);
                tt[i] = std::max(1, (6 * (n - i) + n / 2) / n);
            }
            for (int k = 0; k < 4; ++k) {
                int32_t pv[16], qv[16];
                for (int i = 0; i <= n; ++i) { pv[i] = P(k, i); qv[i] = Q(k, i); }
                const int refP = (pv[n] + pv[n - 1] + 1) >> 1, refQ = (qv[n] + qv[n - 1] + 1) >> 1;
                int sm = 0;
                for (int i = 0; i < n; ++i) sm += pv[i] + qv[i];
                const int refM = (sm + n) / (2 * n);
                for (int i = 0; i < n; ++i) {
                    const int lim = (tt[i] * tc) >> 1;
                    P(k, i) = std::clamp(std::clamp((f[i] * refM + (64 - f[i]) * refP + 32) >> 6, pv[i] - lim, pv[i] + lim), lo, hi);
                    Q(k, i) = std::clamp(std::clamp((f[i] * refM + (64 - f[i]) * refQ + 32) >> 6, qv[i] - lim, qv[i] + lim), lo, hi);
                }
            }
            return;
        }
    }
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

void decay_deblock(Plane& p, const EdgeInfo& e, int rq, int thr, int32_t lo, int32_t hi) {
    if (rq <= 0 || thr <= 0) return;
    // 重み表 (Q8): W[0] = 128 (境界の両側で段差の半分ずつ), W[k] = W[k-1] * rq / 16
    int W[16];
    W[0] = 128;
    for (int k = 1; k < 16; ++k) W[k] = (W[k - 1] * rq + 8) >> 4;
    auto leaf = [&](int cx, int cy) { return e.leaf[static_cast<size_t>(cy) * e.w4 + cx]; };
    auto run = [&](int cx, int cy, int dx, int dy) {
        const int32_t id = leaf(cx, cy);
        int len = 0;
        while (len < 8 && cx >= 0 && cy >= 0 && cx < e.w4 && cy < e.h4 && leaf(cx, cy) == id) { ++len; cx += dx; cy += dy; }
        return len * 4;
    };
    const int W4 = std::min(e.w4, p.w / 4), H4 = std::min(e.h4, p.h / 4);
    for (int dir = 0; dir < 2; ++dir)
        for (int cy = dir; cy < H4; ++cy)
            for (int cx = 1 - dir; cx < W4; ++cx) {
                const int ax = dir == 0 ? cx - 1 : cx, ay = dir == 0 ? cy : cy - 1;
                if (leaf(ax, ay) == leaf(cx, cy)) continue;
                const int lp = dir == 0 ? run(ax, ay, -1, 0) : run(ax, ay, 0, -1);
                const int lq = dir == 0 ? run(cx, cy, 1, 0) : run(cx, cy, 0, 1);
                const int n = std::max(1, std::min(lp, lq) / 2);
                for (int k = 0; k < 4; ++k) {
                    // i < 0: P 側 (p_{-i-1}), i >= 0: Q 側
                    auto px = [&](int i) -> int32_t& { return dir == 0 ? p.at(cx * 4 + i, cy * 4 + k) : p.at(cx * 4 + k, cy * 4 + i); };
                    // 段差から両側の局所勾配を差し引く (なめらかな傾斜は段差とみなさない)
                    const int32_t d = n >= 2 ? (4 * (px(0) - px(-1)) - (px(-1) - px(-2)) - (px(1) - px(0))) / 4 : px(0) - px(-1);
                    if (d == 0 || std::abs(d) >= thr) continue;
                    for (int j = 0; j < n && W[j]; ++j) {
                        const int32_t c = (d * W[j] + (d >= 0 ? 128 : -128)) / 256;
                        if (c == 0) break;
                        px(-1 - j) = std::clamp(px(-1 - j) + c, lo, hi);
                        px(j) = std::clamp(px(j) - c, lo, hi);
                    }
                }
            }
}

void deblock_plane(Plane& p, const EdgeInfo& e, const MotionField* mf, bool chroma, int chroma_shift, int qp,
                   int bit_depth, int beta_off, int tc_off, int32_t lo, int32_t hi) {
    const int bdsh = bit_depth - 8;
    // 垂直エッジは 4 行の帯ごと、水平エッジは 4 列の帯ごとに独立 (触る画素が重ならない) なので並列に処理する。
    // 帯の中の順序は従来どおりなので結果は同一
    for (int dir = 0; dir < 2; ++dir) {  // 0: 垂直エッジ (水平方向にフィルタ), 1: 水平エッジ
        const int nstrip = dir == 0 ? (p.h + 3) / 4 : (p.w + 3) / 4;
        parallel_for(nstrip, [&](int sidx) {
        const int y_beg = dir == 0 ? sidx * 4 : 0, y_end = dir == 0 ? sidx * 4 + 1 : p.h;
        const int x_beg = dir == 0 ? 0 : sidx * 4, x_end = dir == 0 ? p.w : sidx * 4 + 1;
        for (int y = y_beg; y < y_end; y += 4)
            for (int x = x_beg; x < x_end; x += 4) {
                const int ex = x, ey = y;
                if (dir == 0 && (ex == 0 || (ex & 7))) continue;
                if (dir == 1 && (ey == 0 || (ey & 7))) continue;
                const int bs = dir == 0 ? bs_of(e, mf, chroma, chroma_shift, ex - 4, ey, ex, ey) : bs_of(e, mf, chroma, chroma_shift, ex, ey - 4, ex, ey);
                if (bs == 0) continue;
                const int beta = beta_of(qp + beta_off) << bdsh;
                const int tc = tc_of(qp + 2 * (bs - 1) + tc_off) << bdsh;
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
                // 境界に垂直な方向のブロック長 (4x4 セル単位で同一葉が続く長さ)
                auto run = [&](int cx, int cy, int dx, int dy) {
                    const int32_t id = e.leaf[static_cast<size_t>(cy) * e.w4 + cx];
                    int len = 0;
                    while (len < 16 && cx >= 0 && cy >= 0 && cx < e.w4 && cy < e.h4 && e.leaf[static_cast<size_t>(cy) * e.w4 + cx] == id) {
                        ++len; cx += dx; cy += dy;
                    }
                    return len * 4;
                };
                const int lp = dir == 0 ? run(ex / 4 - 1, ey / 4, -1, 0) : run(ex / 4, ey / 4 - 1, 0, -1);
                const int lq = dir == 0 ? run(ex / 4, ey / 4, 1, 0) : run(ex / 4, ey / 4, 0, 1);
                const int ml = std::min(lp, lq);
                const int nlong = ml >= 64 ? 15 : ml >= 32 ? 7 : ml >= 16 ? 5 : 0;
                if (dir == 0)
                    filter_luma_segment([&](int k, int i) -> int32_t& { return p.at(ex + i, ey + k); }, beta, tc, lo, hi, nlong);
                else
                    filter_luma_segment([&](int k, int i) -> int32_t& { return p.at(ex + k, ey + i); }, beta, tc, lo, hi, nlong);
            }
        });
    }
}

// 複数の μ をまとめて処理する (順変換は共通なので 1 回だけ)。各 μ の結果は単独で呼んだときと同一
void freq_gain_filter_multi(const Plane& in, Plane* outs, const int* mqs, int nm, double step, int32_t lo, int32_t hi) {
    constexpr int N = 8, hop = 4;
    constexpr double kPi = 3.14159265358979323846;
    double w[N];
    for (int n = 0; n < N; ++n) w[n] = std::sin(kPi * (n + 0.5) / N);
    const double sigma2 = step * step / 12.0;
    const int nby = (in.h + hop - 1) / hop + 1, nbx = (in.w + hop - 1) / hop + 1;
    const size_t nb = static_cast<size_t>(nby) * nbx;
    // 窓付きブロックの順変換係数 (行ごとに並列)
    std::vector<double> coef(nb * N * N);
    parallel_for(nby, [&](int j) {
        const int by = -hop + j * hop;
        double blk[N * N];
        for (int i = 0; i < nbx; ++i) {
            const int bx = -hop + i * hop;
            for (int y = 0; y < N; ++y)
                for (int x = 0; x < N; ++x)
                    blk[y * N + x] = w[x] * w[y] * in.at(std::clamp(bx + x, 0, in.w - 1), std::clamp(by + y, 0, in.h - 1));
            forward_2d(TxType::DCT2, TxType::DCT2, blk, N, N, &coef[(static_cast<size_t>(j) * nbx + i) * N * N]);
        }
    });
    std::vector<double> rows(nb * N * N), acc(in.v.size()), wsum(in.v.size());
    for (int q = 0; q < nm; ++q) {
        Plane& out = outs[q];
        out = in;
        if (mqs[q] <= 0) continue;
        const double mu = mqs[q] / 4.0;
        // 縮小と逆変換は行ごとに並列、画素への足し込みは従来と同じ順序で直列 (結果は同一)
        parallel_for(nby, [&](int j) {
            double c[N * N];
            for (int i = 0; i < nbx; ++i) {
                const size_t o = (static_cast<size_t>(j) * nbx + i) * N * N;
                std::copy(&coef[o], &coef[o] + N * N, c);
                for (int k = 1; k < N * N; ++k) {
                    const double c2 = c[k] * c[k];
                    c[k] *= c2 > 0 ? std::max(0.0, 1.0 - mu * sigma2 / c2) : 0.0;
                }
                inverse_2d(TxType::DCT2, TxType::DCT2, c, N, N, &rows[o]);
            }
        });
        // 画素への足し込み: 画素行ごとに並列。各画素への加算順 (ブロック行 j の昇順 → ブロック列 i の昇順) は従来と同じ
        parallel_for(in.h, [&](int Y) {
            double* a = &acc[static_cast<size_t>(Y) * in.w];
            double* ws = &wsum[static_cast<size_t>(Y) * in.w];
            std::fill(a, a + in.w, 0.0);
            std::fill(ws, ws + in.w, 0.0);
            for (int j = 0; j < nby; ++j) {
                const int y = Y - (-hop + j * hop);
                if (y < 0 || y >= N) continue;
                for (int i = 0; i < nbx; ++i) {
                    const int bx = -hop + i * hop;
                    const double* r = &rows[(static_cast<size_t>(j) * nbx + i) * N * N + y * N];
                    for (int x = 0; x < N; ++x) {
                        const int X = bx + x;
                        if (X < 0 || X >= in.w) continue;
                        a[X] += w[x] * w[y] * r[x];
                        ws[X] += w[x] * w[x] * w[y] * w[y];
                    }
                }
            }
            int32_t* o = &out.v[static_cast<size_t>(Y) * in.w];
            for (int X = 0; X < in.w; ++X) o[X] = std::clamp(static_cast<int32_t>(std::lround(a[X] / ws[X])), lo, hi);
        });
    }
}

void freq_gain_filter(const Plane& in, Plane& out, double step, int mu_q, int32_t lo, int32_t hi) {
    if (mu_q <= 0) { out = in; return; }
    freq_gain_filter_multi(in, &out, &mu_q, 1, step, lo, hi);
}

void code_loop_filter(SymIO& io, CMModel& m, Plane& R, const Plane* org, const EdgeInfo& e, const MotionField* mf,
                      bool chroma, int chroma_shift, int qp, int bit_depth, double step, double lambda, bool allow_freq,
                      bool allow_map, int32_t lo, int32_t hi, bool deblock) {
    LoopFilterParams lp;
    const Plane pre = R;
    Plane D = R;
    auto sse = [&](const Plane& a) {
        double s = 0;
        for (size_t i = 0; i < a.v.size(); ++i) { const double d = a.v[i] - org->v[i]; s += d * d; }
        return s;
    };
    if (deblock) {
        FVC_PROF("lf_deblock");
        // 9.1 デブロック: 符号器は強さオフセット (β, tc) を候補から最小 SSE で選ぶ
        if (io.enc && org) {
            // 座標降下: (β, tc) オフセットを 1 変数ずつ、良くなる方向へ 2 刻みで動かす (全 20 通りの代わり)
            std::map<std::pair<int, int>, double> memo;
            auto eval = [&](int bo, int to) {
                const auto k = std::make_pair(bo, to);
                auto it = memo.find(k);
                if (it != memo.end()) return it->second;
                Plane t = R;
                deblock_plane(t, e, mf, chroma, chroma_shift, qp, bit_depth, bo, to, lo, hi);
                const double s = sse(t);
                memo[k] = s;
                return s;
            };
            // SPSA-Adam (2 刻みの格子: v = (bo/2, (to+2)/2))
            int bo = 2, to = 2;
            double best = 1e300;
            auto f = [&](const std::array<int, 2>& v) {
                const double s = eval(v[0] * 2, v[1] * 2 - 2);
                if (s < best) { best = s; bo = v[0] * 2; to = v[1] * 2 - 2; }
                return s;
            };
            const std::array<int, 2> st[1] = {{1, 2}};
            lattice_adam<2>(f, st, 1, {0, 0}, {3, 4}, 1.5, nullptr, nullptr, 7, 8);
            lp.beta_off = bo; lp.tc_off = to;
            deblock_plane(D, e, mf, chroma, chroma_shift, qp, bit_depth, bo, to, lo, hi);
        }
        lp.beta_off = io.sint(m, 0, 0, lp.beta_off);
        lp.tc_off = io.sint(m, 1, 0, lp.tc_off);
        if (std::abs(lp.beta_off) > 12 || std::abs(lp.tc_off) > 12) throw std::runtime_error("corrupt stream: deblock offsets");
        if (!(io.enc && org)) deblock_plane(D, e, mf, chroma, chroma_shift, qp, bit_depth, lp.beta_off, lp.tc_off, lo, hi);
    } else {
        // 減衰デブロック: (rq, 閾値) を候補から最小 SSE で選ぶ。閾値は量子化ステップの t/2 倍
        int rq = 0, tq = 0;
        auto thr_of = [&](int t) { return std::max(1, static_cast<int>(std::lround(step * t * 0.5))); };
        if (io.enc && org) {
            // (減衰率, 閾値) の SPSA-Adam (42 通りの評価の代わり)
            static constexpr int kR[6] = {6, 9, 11, 13, 14, 15}, kT[7] = {1, 2, 3, 4, 6, 8, 12};
            double best = sse(D);
            auto f = [&](const std::array<int, 2>& v) {
                Plane t2 = R;
                decay_deblock(t2, e, kR[v[0]], thr_of(kT[v[1]]), lo, hi);
                const double s = sse(t2);
                if (s < best) { best = s; rq = kR[v[0]]; tq = kT[v[1]]; D = std::move(t2); }
                return s;
            };
            const std::array<int, 2> st[1] = {{3, 2}};
            lattice_adam<2>(f, st, 1, {0, 0}, {5, 6}, 1.5, nullptr, nullptr, 11, 8);
        }
        rq = static_cast<int>(io.uint(m, 6, 0, static_cast<uint32_t>(rq)));
        if (rq > 15) throw std::runtime_error("corrupt stream: decay deblock");
        if (rq) {
            tq = static_cast<int>(io.uint(m, 7, 0, static_cast<uint32_t>(tq)));
            if (tq < 1 || tq > 64) throw std::runtime_error("corrupt stream: decay threshold");
        }
        if (!(io.enc && org)) decay_deblock(D, e, rq, thr_of(tq), lo, hi);
    }
    // 9.2 周波数ゲイン: 符号器は μ 候補から最小 SSE を選ぶ
    Plane F;
    if (allow_freq) {
        FVC_PROF("lf_freq");
        if (io.enc) {
            double best = sse(D);
            int bm = 0;
            static constexpr int kMq[4] = {1, 2, 4, 8};
            Plane t[4];
            freq_gain_filter_multi(D, t, kMq, 4, step, lo, hi);
            for (int k = 0; k < 4; ++k) {
                const double s = sse(t[k]);
                if (s < best) { best = s; bm = kMq[k]; F = std::move(t[k]); }
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

namespace {
// 帯域: 半径 (u+v で 6 区分) × 向き (u > v: 横方向の変化, v > u: 縦方向の変化, u = v: 斜め)
constexpr int kBrRad = 6, kBrBands = kBrRad * 3;
inline int br_band(int u, int v) {
    static constexpr int kB[15] = {0, 0, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5, 5, 5};
    return kB[u + v] * 3 + (u > v ? 0 : v > u ? 1 : 2);
}
// 8x8 正弦窓 DCT の重畳加算。cls_gain(cls, band) を係数に掛ける。classify はブロックの AC エネルギーからクラスを返す
template <class F>
void br_apply(const Plane& in, Plane& out, int ncls, double sigma2, F&& gain, int32_t lo, int32_t hi,
              std::vector<double>* num = nullptr, std::vector<double>* den = nullptr, const Plane* org = nullptr) {
    constexpr int N = 8, hop = 4;
    constexpr double kPi = 3.14159265358979323846;
    double w[N];
    for (int n = 0; n < N; ++n) w[n] = std::sin(kPi * (n + 0.5) / N);
    std::vector<double> acc(in.v.size(), 0.0), wsum(in.v.size(), 0.0);
    double blk[N * N], c[N * N], r[N * N], ob[N * N], oc[N * N];
    for (int by = -hop; by < in.h; by += hop)
        for (int bx = -hop; bx < in.w; bx += hop) {
            for (int y = 0; y < N; ++y)
                for (int x = 0; x < N; ++x) {
                    const int X = std::clamp(bx + x, 0, in.w - 1), Y = std::clamp(by + y, 0, in.h - 1);
                    blk[y * N + x] = w[x] * w[y] * in.at(X, Y);
                    if (org) ob[y * N + x] = w[x] * w[y] * org->at(X, Y);
                }
            forward_2d(TxType::DCT2, TxType::DCT2, blk, N, N, c);
            double ac = 0;
            for (int i = 1; i < N * N; ++i) ac += c[i] * c[i];
            // 活動度クラス: log2(AC エネルギー / σ²) を 2 刻みで
            const double rel = ac / std::max(sigma2, 1e-9);
            int cls = rel < 4 ? 0 : rel < 16 ? 1 : rel < 64 ? 2 : 3;
            cls = std::min(cls, ncls - 1);
            if (org) {
                forward_2d(TxType::DCT2, TxType::DCT2, ob, N, N, oc);
                for (int i = 1; i < N * N; ++i) {
                    const int k = cls * kBrBands + br_band(i % N, i / N);
                    (*num)[k] += c[i] * oc[i];
                    (*den)[k] += c[i] * c[i];
                }
                continue;
            }
            for (int i = 1; i < N * N; ++i) c[i] *= gain(cls, br_band(i % N, i / N));
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
    if (org) return;
    out = in;
    for (size_t i = 0; i < acc.size(); ++i)
        out.v[i] = std::clamp(static_cast<int32_t>(std::lround(acc[i] / wsum[i])), lo, hi);
}
}  // namespace

void code_band_restore(SymIO& io, CMModel& m, Plane& R, const Plane* org, bool luma, double step, double lambda,
                       int32_t lo, int32_t hi) {
    const int ncls = luma ? 4 : 2;
    const double sigma2 = step * step / 12.0;
    std::vector<int> g(static_cast<size_t>(ncls) * kBrBands, 32);
    int on = 0;
    Plane F;
    if (io.enc && org) {
        std::vector<double> num(g.size(), 0.0), den(g.size(), 0.0);
        Plane dummy;
        br_apply(R, dummy, ncls, sigma2, [](int, int) { return 1.0; }, lo, hi, &num, &den, org);
        // 係数領域の SSE 減少 (num - den)² / den が符号量 (約 6 ビット) に見合うゲインだけ 1 以外にする
        for (size_t k = 0; k < g.size(); ++k) {
            if (den[k] <= 0) continue;
            const int q = std::clamp(static_cast<int>(std::lround(num[k] / den[k] * 32.0)), 0, 64);
            const double gg = q / 32.0, red = 2.0 * (gg - 1.0) * (num[k] - den[k]) - (gg - 1.0) * (gg - 1.0) * den[k];
            g[k] = red > lambda * 6.0 ? q : 32;
        }
        br_apply(R, F, ncls, sigma2, [&](int c, int b) { return g[static_cast<size_t>(c) * kBrBands + b] / 32.0; }, lo, hi);
        double s0 = 0, s1 = 0;
        for (size_t i = 0; i < R.v.size(); ++i) {
            const double d0 = R.v[i] - org->v[i], d1 = F.v[i] - org->v[i];
            s0 += d0 * d0; s1 += d1 * d1;
        }
        int nz = 0;
        for (int v : g) nz += v != 32;
        on = nz > 0 && s0 - s1 > lambda * (6.0 * nz + 1.0 * static_cast<double>(g.size()));
    }
    on = io.bit(m, 70, luma ? 1 : 0, 0, on);
    if (!on) return;
    for (size_t k = 0; k < g.size(); ++k) {
        // 同じ帯域の 1 つ下のクラスのゲインとの差分
        const int pr = k >= kBrBands ? g[k - kBrBands] : 32;
        g[k] = pr + io.sint(m, 71 + static_cast<uint32_t>(k % kBrBands), luma ? 1u : 0u, g[k] - pr);
        if (g[k] < 0 || g[k] > 64) throw std::runtime_error("corrupt stream: band restore gain");
    }
    if (!(io.enc && org)) br_apply(R, F, ncls, sigma2, [&](int c, int b) { return g[static_cast<size_t>(c) * kBrBands + b] / 32.0; }, lo, hi);
    R = std::move(F);
}

}  // namespace fvc
