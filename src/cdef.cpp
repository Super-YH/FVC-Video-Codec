#include "cdef.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "parallel.hpp"

namespace fvc {

namespace {
constexpr int kDirs[8][2][2] = {{{-1, 1}, {-2, 2}}, {{0, 1}, {-1, 2}}, {{0, 1}, {0, 2}}, {{0, 1}, {1, 2}},
                                {{1, 1}, {2, 2}},   {{1, 0}, {2, 1}},  {{1, 0}, {2, 0}}, {{1, 0}, {2, -1}}};
constexpr int kDiv[9] = {0, 840, 420, 280, 210, 168, 140, 120, 105};
constexpr int kPri[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
constexpr int kSec[4] = {0, 1, 2, 4};
constexpr int kBlk = 64;

int ilog2(int v) { int l = 0; while ((2 << l) <= v) ++l; return l; }

// AV1 cdef_find_dir: 8 方向の射影エネルギーから方向と分散を求める
int find_dir(const Plane& p, int x0, int y0, int sh, int& var) {
    int32_t partial[8][15] = {};
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j) {
            const int x = (p.at(std::min(x0 + j, p.w - 1), std::min(y0 + i, p.h - 1)) >> sh) - 128;
            partial[0][i + j] += x;
            partial[1][i + j / 2] += x;
            partial[2][i] += x;
            partial[3][3 + i - j / 2] += x;
            partial[4][7 + i - j] += x;
            partial[5][3 - i / 2 + j] += x;
            partial[6][j] += x;
            partial[7][i / 2 + j] += x;
        }
    int64_t cost[8] = {};
    for (int i = 0; i < 8; ++i) {
        cost[2] += static_cast<int64_t>(partial[2][i]) * partial[2][i];
        cost[6] += static_cast<int64_t>(partial[6][i]) * partial[6][i];
    }
    cost[2] *= kDiv[8];
    cost[6] *= kDiv[8];
    for (int i = 0; i < 7; ++i) {
        cost[0] += (static_cast<int64_t>(partial[0][i]) * partial[0][i] + static_cast<int64_t>(partial[0][14 - i]) * partial[0][14 - i]) * kDiv[i + 1];
        cost[4] += (static_cast<int64_t>(partial[4][i]) * partial[4][i] + static_cast<int64_t>(partial[4][14 - i]) * partial[4][14 - i]) * kDiv[i + 1];
    }
    cost[0] += static_cast<int64_t>(partial[0][7]) * partial[0][7] * kDiv[8];
    cost[4] += static_cast<int64_t>(partial[4][7]) * partial[4][7] * kDiv[8];
    for (int i = 1; i < 8; i += 2) {
        for (int j = 0; j < 5; ++j) cost[i] += static_cast<int64_t>(partial[i][3 + j]) * partial[i][3 + j];
        cost[i] *= kDiv[8];
        for (int j = 0; j < 3; ++j)
            cost[i] += (static_cast<int64_t>(partial[i][j]) * partial[i][j] + static_cast<int64_t>(partial[i][10 - j]) * partial[i][10 - j]) * kDiv[2 * j + 2];
    }
    int best = 0;
    for (int d = 1; d < 8; ++d) if (cost[d] > cost[best]) best = d;
    var = static_cast<int>((cost[best] - cost[(best + 4) & 7]) >> 10);
    return best;
}

// 8x8 ブロックを 1 組の強さでフィルタ (入力 src, 出力 dst)
void filter_block(const Plane& src, int32_t* dst, int ds, int x0, int y0, int dir, int var, int pri, int sec, int damping, bool luma,
                  int32_t lo, int32_t hi) {
    int ps = pri;
    if (luma && pri) {
        const int i = var ? std::min(ilog2(var >> 6), 12) : 0;
        ps = var ? (pri * (4 + i) + 8) >> 4 : 0;
    }
    const int pt0 = (ps & 1) ? 3 : 4, pt1 = (ps & 1) ? 3 : 2;
    // 12x12 の窓 (境界はクランプ) に取り出し、内側のループからクランプと ilog2 を除く
    constexpr int B = 12, O = 2;
    int32_t win[B * B];
    for (int y = 0; y < B; ++y) {
        const int sy = std::clamp(y0 + y - O, 0, src.h - 1);
        const int32_t* r = &src.v[static_cast<size_t>(sy) * src.w];
        for (int x = 0; x < B; ++x) win[y * B + x] = r[std::clamp(x0 + x - O, 0, src.w - 1)];
    }
    const int psh = ps ? std::max(0, damping - ilog2(ps)) : 0;
    const int ssh = sec ? std::max(0, damping - 1 - ilog2(sec)) : 0;
    auto con = [](int diff, int thr, int shift) {
        const int a = std::abs(diff);
        const int v = std::clamp(thr - (a >> shift), 0, a);
        return diff < 0 ? -v : v;
    };
    int po[2], so[2][2];
    for (int k = 0; k < 2; ++k) {
        po[k] = kDirs[dir][k][0] * B + kDirs[dir][k][1];
        so[0][k] = kDirs[(dir + 2) & 7][k][0] * B + kDirs[(dir + 2) & 7][k][1];
        so[1][k] = kDirs[(dir + 6) & 7][k][0] * B + kDirs[(dir + 6) & 7][k][1];
    }
    const int ye = std::min(y0 + 8, src.h), xe = std::min(x0 + 8, src.w);
    for (int y = y0; y < ye; ++y)
        for (int x = x0; x < xe; ++x) {
            const int32_t* q = &win[(y - y0 + O) * B + (x - x0 + O)];
            const int c = q[0];
            int sum = 0, mx = c, mn = c;
            for (int k = 0; k < 2; ++k) {
                const int p0 = q[po[k]], p1 = q[-po[k]];
                if (ps) sum += (k ? pt1 : pt0) * (con(p0 - c, ps, psh) + con(p1 - c, ps, psh));
                mx = std::max({mx, p0, p1}); mn = std::min({mn, p0, p1});
                for (int j = 0; j < 2; ++j) {
                    const int s0 = q[so[j][k]], s1 = q[-so[j][k]];
                    if (sec) sum += (k ? 1 : 2) * (con(s0 - c, sec, ssh) + con(s1 - c, sec, ssh));
                    mx = std::max({mx, s0, s1}); mn = std::min({mn, s0, s1});
                }
            }
            const int v = c + ((8 + sum - (sum < 0)) >> 4);
            dst[(y - y0) * ds + (x - x0)] = std::clamp(std::clamp(v, mn, mx), lo, hi);
        }
}
}  // namespace

void code_cdef(SymIO& io, CMModel& m, Plane& R, const Plane* org, bool luma, int qp, int bit_depth, double lambda,
               int32_t lo, int32_t hi) {
    const int sh = bit_depth - 8;
    const int damping = 3 + std::clamp((qp - 20) / 10, 0, 3) + sh;
    const int bw8 = (R.w + 7) / 8, bh8 = (R.h + 7) / 8;
    std::vector<int> dirs(static_cast<size_t>(bw8) * bh8), vars(dirs.size());
    for (int by = 0; by < bh8; ++by)
        for (int bx = 0; bx < bw8; ++bx) dirs[by * bw8 + bx] = find_dir(R, bx * 8, by * 8, sh, vars[by * bw8 + bx]);
    const int cw = (R.w + kBlk - 1) / kBlk, ch = (R.h + kBlk - 1) / kBlk;
    // 符号器: 候補 (主, 副) ごとに 64x64 ブロックの SSE を求め、最大 4 組を貪欲に選ぶ
    std::vector<std::pair<int, int>> presets;
    std::vector<int> choice(static_cast<size_t>(cw) * ch, 0);
    if (io.enc) {
        // 2 段探索 (高速化): 主強度を副 0 で評価 → 総 SSE の良い主強度 2 つについて副強度を評価 (36 → 17 回のフィルタ)
        std::vector<std::pair<int, int>> cand;
        std::vector<std::vector<double>> sse;
        // 候補ごとの 64x64 ブロック SSE。8x8 ごとに局所バッファへフィルタして誤差を足す (整数の和なので順序によらず同値)。
        // 8x8 の行ごとに並列
        auto eval = [&](int p, int s2) {
            std::vector<std::vector<int64_t>> part(bh8, std::vector<int64_t>(cw, 0));
            parallel_for(bh8, [&](int by) {
                int32_t buf[64];
                const int y0 = by * 8, ye = std::min(y0 + 8, R.h);
                for (int bx = 0; bx < bw8; ++bx) {
                    const int x0 = bx * 8, xe = std::min(x0 + 8, R.w);
                    const bool f = p || s2;
                    if (f) filter_block(R, buf, 8, x0, y0, dirs[by * bw8 + bx], vars[by * bw8 + bx], p << sh, s2 << sh, damping, luma, lo, hi);
                    int64_t acc = 0;
                    for (int y = y0; y < ye; ++y)
                        for (int x = x0; x < xe; ++x) {
                            const int64_t d = org->at(x, y) - (f ? buf[(y - y0) * 8 + (x - x0)] : R.at(x, y));
                            acc += d * d;
                        }
                    part[by][x0 / kBlk] += acc;
                }
            });
            std::vector<double> e(choice.size(), 0.0);
            for (int by = 0; by < bh8; ++by)
                for (int c = 0; c < cw; ++c) e[(by * 8 / kBlk) * cw + c] += static_cast<double>(part[by][c]);
            cand.push_back({p, s2});
            sse.push_back(std::move(e));
        };
        // 座標降下 (全候補の評価の代わり): 主強度の格子 {0,1,2,3,4,6,8,11,15} 上で、中央 (4) から総 SSE が
        // 下がる向きへ 1 段ずつ進む。最良の主強度について副強度 {1,2,4} を評価する (17 → 約 8 回のフィルタ)
        static constexpr int kPg[9] = {0, 1, 2, 3, 4, 6, 8, 11, 15};
        auto total_of = [&](size_t c) { double t = 0; for (double v : sse[c]) t += v; return t; };
        auto find_or_eval = [&](int p, int s2) {
            for (size_t c = 0; c < cand.size(); ++c) if (cand[c].first == p && cand[c].second == s2) return c;
            eval(p, s2);
            return cand.size() - 1;
        };
        find_or_eval(0, 0);
        int gi = 4;
        double bt = total_of(find_or_eval(kPg[gi], 0));
        for (int dir : {-1, 1}) {
            for (;;) {
                const int ni = gi + dir;
                if (ni < 1 || ni > 8) break;
                const double t = total_of(find_or_eval(kPg[ni], 0));
                if (t < bt) { bt = t; gi = ni; } else break;
            }
        }
        for (int s2 : {1, 2, 4}) find_or_eval(kPg[gi], s2);
        find_or_eval(0, 1);
        std::vector<size_t> sel = {0};  // (0,0) = 無効 は常に含める
        auto total = [&](const std::vector<size_t>& s) {
            double t = 0;
            for (size_t b = 0; b < choice.size(); ++b) {
                double mbest = 1e300;
                for (size_t c : s) mbest = std::min(mbest, sse[c][b]);
                t += mbest;
            }
            return t + lambda * (6.0 * s.size() + (s.size() > 1 ? 2.0 * choice.size() : 0.0));
        };
        double cur = total(sel);
        while (sel.size() < 4) {
            double best = cur;
            size_t bc = 0;
            for (size_t c = 1; c < cand.size(); ++c) {
                if (std::find(sel.begin(), sel.end(), c) != sel.end()) continue;
                auto s2 = sel; s2.push_back(c);
                const double t = total(s2);
                if (t < best) { best = t; bc = c; }
            }
            if (!bc) break;
            sel.push_back(bc);
            cur = best;
        }
        for (size_t c : sel) presets.push_back(cand[c]);
        for (size_t b = 0; b < choice.size(); ++b) {
            double mbest = 1e300;
            for (size_t i = 0; i < sel.size(); ++i)
                if (sse[sel[i]][b] < mbest) { mbest = sse[sel[i]][b]; choice[b] = static_cast<int>(i); }
        }
    }
    const int np = static_cast<int>(io.uint(m, 60, luma, io.enc ? static_cast<uint32_t>(presets.size()) - 1u : 0u)) + 1;
    if (np < 1 || np > 4) throw std::runtime_error("corrupt stream: cdef presets");
    if (!io.enc) presets.assign(np, {0, 0});
    for (int i = 1; i < np; ++i) {  // 0 番は常に (0, 0)
        presets[i].first = static_cast<int>(io.uint(m, 61, luma, static_cast<uint32_t>(presets[i].first)));
        int si = 0;
        for (int k = 0; k < 4; ++k) if (kSec[k] == presets[i].second) si = k;
        si = io.bit(m, 62, luma, 0, si >> 1) * 2;
        si += io.bit(m, 63, luma, 0, io.enc ? (presets[i].second == kSec[si + 1]) : 0);
        presets[i].second = kSec[si];
        if (presets[i].first > 15) throw std::runtime_error("corrupt stream: cdef strength");
    }
    if (np == 1) return;
    int prev = 0;
    for (size_t b = 0; b < choice.size(); ++b) {
        int c = choice[b];
        int v = io.bit(m, 64, luma, static_cast<uint32_t>(prev), c != 0);
        if (v) {
            int k = 1;
            while (k < np - 1 && io.bit(m, 65 + static_cast<uint32_t>(k), luma, 0, c > k)) ++k;
            v = k;
        }
        choice[b] = v;
        prev = v;
    }
    // 適用 (選択された組で 8x8 単位)
    const Plane src = R;
    parallel_for(bh8, [&](int by) {
        for (int bx = 0; bx < bw8; ++bx) {
            const int c = choice[((by * 8) / kBlk) * cw + (bx * 8) / kBlk];
            if (!c) continue;
            filter_block(src, &R.at(bx * 8, by * 8), R.w, bx * 8, by * 8, dirs[by * bw8 + bx], vars[by * bw8 + bx], presets[c].first << sh,
                         presets[c].second << sh, damping, luma, lo, hi);
        }
    });
    (void)kPri;
}

}  // namespace fvc
