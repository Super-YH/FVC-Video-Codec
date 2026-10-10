#include "cdef.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

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

inline int constrain(int diff, int thr, int damping) {
    if (!thr) return 0;
    const int shift = std::max(0, damping - ilog2(thr));
    const int a = std::abs(diff);
    const int v = std::clamp(thr - (a >> shift), 0, a);
    return diff < 0 ? -v : v;
}

// 8x8 ブロックを 1 組の強さでフィルタ (入力 src, 出力 dst)
void filter_block(const Plane& src, Plane& dst, int x0, int y0, int dir, int var, int pri, int sec, int damping, bool luma,
                  int32_t lo, int32_t hi) {
    int ps = pri;
    if (luma && pri) {
        const int i = var ? std::min(ilog2(var >> 6), 12) : 0;
        ps = var ? (pri * (4 + i) + 8) >> 4 : 0;
    }
    const int pt0 = (ps & 1) ? 3 : 4, pt1 = (ps & 1) ? 3 : 2;
    auto px = [&](int x, int y) { return src.at(std::clamp(x, 0, src.w - 1), std::clamp(y, 0, src.h - 1)); };
    for (int y = y0; y < std::min(y0 + 8, src.h); ++y)
        for (int x = x0; x < std::min(x0 + 8, src.w); ++x) {
            const int c = src.at(x, y);
            int sum = 0, mx = c, mn = c;
            for (int k = 0; k < 2; ++k) {
                const int dy = kDirs[dir][k][0], dx = kDirs[dir][k][1];
                const int p0 = px(x + dx, y + dy), p1 = px(x - dx, y - dy);
                const int t = k ? pt1 : pt0;
                sum += t * (constrain(p0 - c, ps, damping) + constrain(p1 - c, ps, damping));
                mx = std::max({mx, p0, p1}); mn = std::min({mn, p0, p1});
                for (int s2 : {(dir + 2) & 7, (dir + 6) & 7}) {
                    const int sy = kDirs[s2][k][0], sx = kDirs[s2][k][1];
                    const int s0 = px(x + sx, y + sy), s1 = px(x - sx, y - sy);
                    sum += (k ? 1 : 2) * (constrain(s0 - c, sec, damping - 1) + constrain(s1 - c, sec, damping - 1));
                    mx = std::max({mx, s0, s1}); mn = std::min({mn, s0, s1});
                }
            }
            const int v = c + ((8 + sum - (sum < 0)) >> 4);
            dst.at(x, y) = std::clamp(std::clamp(v, mn, mx), lo, hi);
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
    auto filt_all = [&](int pri, int sec, Plane& out) {
        out = R;
        for (int by = 0; by < bh8; ++by)
            for (int bx = 0; bx < bw8; ++bx)
                filter_block(R, out, bx * 8, by * 8, dirs[by * bw8 + bx], vars[by * bw8 + bx], pri << sh, sec << sh, damping, luma, lo, hi);
    };
    // 符号器: 候補 (主, 副) ごとに 64x64 ブロックの SSE を求め、最大 4 組を貪欲に選ぶ
    std::vector<std::pair<int, int>> presets;
    std::vector<int> choice(static_cast<size_t>(cw) * ch, 0);
    if (io.enc) {
        std::vector<std::pair<int, int>> cand;
        for (int p : {0, 1, 2, 3, 4, 6, 8, 11, 15})
            for (int s2 : {0, 1, 2, 4}) cand.push_back({p, s2});
        std::vector<std::vector<double>> sse(cand.size(), std::vector<double>(choice.size(), 0.0));
        Plane F;
        for (size_t c = 0; c < cand.size(); ++c) {
            if (cand[c].first == 0 && cand[c].second == 0) F = R;
            else filt_all(cand[c].first, cand[c].second, F);
            for (int y = 0; y < R.h; ++y)
                for (int x = 0; x < R.w; ++x) {
                    const double d = org->at(x, y) - F.at(x, y);
                    sse[c][(y / kBlk) * cw + x / kBlk] += d * d;
                }
        }
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
    for (int by = 0; by < bh8; ++by)
        for (int bx = 0; bx < bw8; ++bx) {
            const int c = choice[((by * 8) / kBlk) * cw + (bx * 8) / kBlk];
            if (!c) continue;
            filter_block(src, R, bx * 8, by * 8, dirs[by * bw8 + bx], vars[by * bw8 + bx], presets[c].first << sh,
                         presets[c].second << sh, damping, luma, lo, hi);
        }
    (void)kPri;
}

}  // namespace fvc
