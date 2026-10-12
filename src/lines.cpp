#include "lines.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

namespace fvc {

std::vector<LineSeg> extract_lines(const Plane& ref, int vw, int vh, int bit_depth, int thr8) {
    std::vector<LineSeg> out;
    const int W = std::min(vw, ref.w), H = std::min(vh, ref.h);
    // |gx|+|gy| の下限 (平坦なセルには線を置かない)。実測: 64 に下げても改善しない (弱いエッジの動き推定が不安定)
    const int32_t thr = thr8 << std::max(0, bit_depth - 8);
    auto at = [&](int x, int y) { return ref.at(std::clamp(x, 0, W - 1), std::clamp(y, 0, H - 1)); };
    for (int cy = 0; cy < H; cy += kLineCell)
        for (int cx = 0; cx < W; cx += kLineCell) {
            // セルの内側 (端 2 画素を除く) で Sobel 勾配が最大の点、次に最初の点から 6 画素以上離れた 2 番目の点
            // (曲線や角を 2 本の線で表す)
            LineSeg first;
            for (int pass = 0; pass < 2; ++pass) {
                LineSeg best;
                int32_t bm = thr - 1;
                for (int y = cy + 2; y < std::min(cy + kLineCell - 2, H - 1); ++y)
                    for (int x = cx + 2; x < std::min(cx + kLineCell - 2, W - 1); ++x) {
                        if (pass && (x - first.x) * (x - first.x) + (y - first.y) * (y - first.y) < 36) continue;
                        const int32_t gx = (at(x + 1, y - 1) + 2 * at(x + 1, y) + at(x + 1, y + 1)) - (at(x - 1, y - 1) + 2 * at(x - 1, y) + at(x - 1, y + 1));
                        const int32_t gy = (at(x - 1, y + 1) + 2 * at(x, y + 1) + at(x + 1, y + 1)) - (at(x - 1, y - 1) + 2 * at(x, y - 1) + at(x + 1, y - 1));
                        const int32_t m = std::abs(gx) + std::abs(gy);
                        if (m > bm) { bm = m; best.x = x; best.y = y; best.gx = gx; best.gy = gy; }
                    }
                if (bm < thr) break;
                out.push_back(best);
                first = best;
            }
        }
    return out;
}

namespace {
constexpr int kWin = 8;  // 動き推定の窓 (線の現在位置を中心に 2*kWin 四方)

// 線 L が動き (mvx, mvy) で動いたときの窓の SAD。窓は現フレームでの線の位置 c = r - mv を中心にとる
// (予測は pred(p) = ref(p + mv))。half: 0 = 全体, 1 = 法線の + 側, -1 = − 側
int64_t window_sad(const Plane& cur, const Plane& ref, const LineSeg& L, int mvx, int mvy, int64_t bound, int half = 0) {
    const int ix = mvx >> 2, iy = mvy >> 2, fx = mvx & 3, fy = mvy & 3;
    const int w00 = (4 - fx) * (4 - fy), w10 = fx * (4 - fy), w01 = (4 - fx) * fy, w11 = fx * fy;
    auto r = [&](int x, int y) { return ref.at(std::clamp(x, 0, ref.w - 1), std::clamp(y, 0, ref.h - 1)); };
    const int x0 = L.x - (mvx >> 2), y0 = L.y - (mvy >> 2);
    int64_t sad = 0;
    for (int y = y0 - kWin; y < y0 + kWin && sad < bound; ++y) {
        const int cy = std::clamp(y, 0, cur.h - 1);
        for (int x = x0 - kWin; x < x0 + kWin; ++x) {
            if (half) {
                const int64_t dn = static_cast<int64_t>(x - x0) * L.gx + static_cast<int64_t>(y - y0) * L.gy;
                if ((half > 0) != (dn > 0) || dn == 0) continue;
            }
            const int cx = std::clamp(x, 0, cur.w - 1);
            const int32_t p = (w00 * r(cx + ix, cy + iy) + w10 * r(cx + ix + 1, cy + iy) + w01 * r(cx + ix, cy + iy + 1) +
                               w11 * r(cx + ix + 1, cy + iy + 1) + 8) >> 4;
            sad += std::abs(cur.at(cx, cy) - p);
        }
    }
    return sad;
}

double mv_bits(int dx, int dy) { return 2.0 + 2.0 * std::log2(1.0 + std::abs(dx)) + 2.0 * std::log2(1.0 + std::abs(dy)); }
}  // namespace

void estimate_line_motion(std::vector<LineSeg>& lines, const Plane& cur, const Plane& ref, const GlobalModel& gm,
                          double lambda, int range) {
    const double lsad = std::sqrt(lambda);
    for (LineSeg& L : lines) {
        const int px = L.bx, py = L.by;
        const double c0 = static_cast<double>(window_sad(cur, ref, L, px, py, INT64_MAX)) + lsad * 1.0;
        int bx = (px + 2) >> 2, by = (py + 2) >> 2;
        double bc = 1e300;
        auto test = [&](int ix, int iy) {
            const double c = static_cast<double>(window_sad(cur, ref, L, ix * 4, iy * 4, static_cast<int64_t>(std::min(bc, 1e15)) + 1)) +
                             lsad * mv_bits(ix * 4 - px, iy * 4 - py);
            if (c < bc) { bc = c; bx = ix; by = iy; }
        };
        const int cx = bx, cy = by;
        for (int dy = -range; dy <= range; ++dy)
            for (int dx = -range; dx <= range; ++dx) test(cx + dx, cy + dy);
        // 粗い格子 (大きな動き)。実測 (v17 特許安全): ±48 画素に広げても改善せず (-0.02 dB)、±4·range に留める
        for (int dy = -4 * range; dy <= 4 * range; dy += 4)
            for (int dx = -4 * range; dx <= 4 * range; dx += 4) test(cx + dx, cy + dy);
        if (bx != cx || by != cy) {
            const int ox = bx, oy = by;
            for (int dy = -2; dy <= 2; ++dy)
                for (int dx = -2; dx <= 2; ++dx) test(ox + dx, oy + dy);
        }
        test(0, 0);
        int mx = bx * 4, my = by * 4;
        double sc = static_cast<double>(window_sad(cur, ref, L, mx, my, INT64_MAX)) + lsad * mv_bits(mx - px, my - py);
        for (int st = 2; st >= 1; st >>= 1) {
            const int ox = mx, oy = my;
            for (int dy = -st; dy <= st; dy += st)
                for (int dx = -st; dx <= st; dx += st) {
                    if (!dx && !dy) continue;
                    const double c = static_cast<double>(window_sad(cur, ref, L, ox + dx, oy + dy, INT64_MAX)) +
                                     lsad * mv_bits(ox + dx - px, oy + dy - py);
                    if (c < sc) { sc = c; mx = ox + dx; my = oy + dy; }
                }
        }
        // 動き場は周囲のブロックにも効くため、変更には窓の SAD 改善が 25% 以上あることも求める
        // (遮蔽が解けた背景などの一貫しない動きを送らない)
        L.changed = (mx != px || my != py) && sc + lsad * 1.0 < c0 * 0.75;
        L.mvx = L.changed ? mx : px;
        L.mvy = L.changed ? my : py;
        L.side = 2;
        if (L.changed) {
            // どちら側が動いたか: 半窓ごとに、線の動きと大域動きの SAD を比べる
            const int64_t pm = window_sad(cur, ref, L, mx, my, INT64_MAX, 1), pg = window_sad(cur, ref, L, px, py, INT64_MAX, 1);
            const int64_t nm = window_sad(cur, ref, L, mx, my, INT64_MAX, -1), ng = window_sad(cur, ref, L, px, py, INT64_MAX, -1);
            const bool pos = pm * 10 < pg * 9, neg = nm * 10 < ng * 9;
            L.side = pos && !neg ? 1 : (neg && !pos ? 0 : 2);
        }
    }
    // 孤立した変化を落とす: 24 画素以内に、似た動き (差 2 画素以内) で変わった別の線がない線は雑音とみなす
    std::vector<char> keep(lines.size(), 0);
    for (size_t i = 0; i < lines.size(); ++i) {
        if (!lines[i].changed) continue;
        for (size_t j = 0; j < lines.size() && !keep[i]; ++j) {
            if (j == i || !lines[j].changed) continue;
            const int dx = lines[i].x - lines[j].x, dy = lines[i].y - lines[j].y;
            if (dx * dx + dy * dy > 24 * 24) continue;
            keep[i] = std::abs(lines[i].mvx - lines[j].mvx) <= 8 && std::abs(lines[i].mvy - lines[j].mvy) <= 8;
        }
    }
    for (size_t i = 0; i < lines.size(); ++i)
        if (lines[i].changed && !keep[i]) {
            lines[i].changed = false;
            lines[i].mvx = lines[i].bx; lines[i].mvy = lines[i].by;
            lines[i].side = 2;
        }
}

void set_line_base(std::vector<LineSeg>& lines, const GlobalModel& gm) {
    for (LineSeg& L : lines) gm.mv_at(L.x, L.y, L.bx, L.by);
}

void code_line_motion(SymIO& io, CMModel& m, std::vector<LineSeg>& lines, const GlobalModel& gm) {
    // 変わった線だけを列挙して送る: 個数、続いて各線の番号の差 (直前に送った線からの間隔)、動きの差分、動く側。
    // 列挙されない線は大域動きのまま (線ごとのフラグを送らないので、変化が少ないフレームではほぼ 0 ビット)
    uint32_t nch = 0;
    for (const LineSeg& L : lines) nch += L.changed ? 1u : 0u;
    nch = io.uint(m, 10, 0, nch);
    if (nch > lines.size()) throw std::runtime_error("corrupt stream: line count");
    std::vector<char> chg(lines.size(), 0);
    if (io.enc) for (size_t i = 0; i < lines.size(); ++i) chg[i] = lines[i].changed;
    int prev = -1;
    int pdx = 0, pdy = 0;  // 直前に変わった線の差分 (差分の予測に使う: 動物体の線は同じ向きに動く)
    uint32_t pgap = 0;
    for (uint32_t k = 0; k < nch; ++k) {
        int idx = 0;
        if (io.enc) { idx = prev + 1; while (!chg[static_cast<size_t>(idx)]) ++idx; }
        const uint32_t gap = io.uint(m, 11, std::min<uint32_t>(pgap, 7), static_cast<uint32_t>(idx - prev - 1));
        idx = prev + 1 + static_cast<int>(gap);
        if (idx >= static_cast<int>(lines.size())) throw std::runtime_error("corrupt stream: line index");
        pgap = gap;
        prev = idx;
        LineSeg& L = lines[static_cast<size_t>(idx)];
        L.changed = true;
        chg[static_cast<size_t>(idx)] = 1;
        const int px = L.bx, py = L.by;
        const int dx = pdx + io.sint(m, 1, 0, L.mvx - px - pdx);
        const int dy = pdy + io.sint(m, 2, 0, L.mvy - py - pdy);
        if (std::abs(dx) > 4096 || std::abs(dy) > 4096) throw std::runtime_error("corrupt stream: line motion");
        L.mvx = px + dx;
        L.mvy = py + dy;
        pdx = dx; pdy = dy;
        const int both = io.bit(m, 3, 0, 0, L.side == 2);
        L.side = both ? 2 : io.bit(m, 4, 0, 0, L.side == 1);
    }
    for (size_t i = 0; i < lines.size(); ++i)
        if (!chg[i]) {
            LineSeg& L = lines[i];
            L.changed = false;
            L.mvx = L.bx; L.mvy = L.by;
            L.side = 2;
        }
}

void build_line_field(const std::vector<LineSeg>& lines, const GlobalModel& gm, int w4, int h4, std::vector<int16_t>& field) {
    field.assign(static_cast<size_t>(w4) * h4 * 2, 0);
    // 線をセルに登録 (近傍探索用)
    const int cw = (w4 * 4 + kLineCell - 1) / kLineCell, ch = (h4 * 4 + kLineCell - 1) / kLineCell;
    std::vector<std::vector<int>> cell(static_cast<size_t>(cw) * ch);
    // 線の現フレームでの位置 (変わった線は参照位置から動きの分だけずらす。動きは pred(p) = ref(p + mv))
    std::vector<int> lx(lines.size()), ly(lines.size());
    auto fdiv4 = [](int v) { return v >= 0 ? v / 4 : -((-v + 3) / 4); };
    for (size_t i = 0; i < lines.size(); ++i) {
        lx[i] = lines[i].x - fdiv4(lines[i].mvx);
        ly[i] = lines[i].y - fdiv4(lines[i].mvy);
        const int cx = std::clamp(lx[i] / kLineCell, 0, cw - 1), cy = std::clamp(ly[i] / kLineCell, 0, ch - 1);
        cell[static_cast<size_t>(cy) * cw + cx].push_back(static_cast<int>(i));
    }
    constexpr int kR2 = 24 * 24;    // 影響半径の 2 乗
    constexpr int kSoft = 64;       // 距離 0 付近の重みの上限 (w = 2^16 / (d² + kSoft))
    for (int by = 0; by < h4; ++by)
        for (int bx = 0; bx < w4; ++bx) {
            const int x = bx * 4 + 2, y = by * 4 + 2;
            int gx, gy;
            gm.mv_at(x, y, gx, gy);
            // 大域動きを弱い重みの候補として含める (線から離れるほど大域動きに戻る)
            int64_t ws = 65536 / (kR2 + kSoft), sx = ws * gx, sy = ws * gy;
            const int ccx = x / kLineCell, ccy = y / kLineCell;
            for (int cy = std::max(0, ccy - 2); cy <= std::min(ch - 1, ccy + 2); ++cy)
                for (int cx = std::max(0, ccx - 2); cx <= std::min(cw - 1, ccx + 2); ++cx)
                    for (int i : cell[static_cast<size_t>(cy) * cw + cx]) {
                        const LineSeg& L = lines[static_cast<size_t>(i)];
                        const int dx = lx[static_cast<size_t>(i)] - x, dy = ly[static_cast<size_t>(i)] - y;
                        const int d2 = dx * dx + dy * dy;
                        if (d2 > kR2) continue;
                        if (L.side != 2) {
                            // 動く側でない画素 (線から 2 画素より離れた反対側) には効かせない
                            const int64_t dn = -(static_cast<int64_t>(dx) * L.gx + static_cast<int64_t>(dy) * L.gy);  // (p - c)·n
                            const int64_t nn = static_cast<int64_t>(L.gx) * L.gx + static_cast<int64_t>(L.gy) * L.gy;
                            const bool on_pos = dn > 0;
                            if (on_pos != (L.side == 1) && dn * dn > 4 * nn) continue;
                        }
                        const int64_t w = 65536 / (d2 + kSoft);
                        ws += w; sx += w * L.mvx; sy += w * L.mvy;
                    }
            auto rdiv = [](int64_t a, int64_t b) { return a >= 0 ? (a + b / 2) / b : -((-a + b / 2) / b); };
            field[(static_cast<size_t>(by) * w4 + bx) * 2] = static_cast<int16_t>(std::clamp<int64_t>(rdiv(sx, ws), -16000, 16000));
            field[(static_cast<size_t>(by) * w4 + bx) * 2 + 1] = static_cast<int16_t>(std::clamp<int64_t>(rdiv(sy, ws), -16000, 16000));
        }
}

// ---------------- 帯域画像の線予測 ----------------
namespace {
constexpr double kEdgeW[4] = {0.35, 0.7, 1.2, 2.0};  // 理想エッジの傾斜の幅 (画素)
constexpr int kBlT = 8, kBlN = 6;                       // 線の影響範囲: 接線方向 ±8, 法線方向 ±6

// 線 L を (位置補正 dq/4 画素, 鋭さ s) で y に適用する (または適用した場合の値を f に渡す)
template <class F>
void band_line_apply(const std::vector<double>& y, int W, int H, const LineSeg& L, int dq, int s, F&& f) {
    const double gn = std::hypot(static_cast<double>(L.gx), static_cast<double>(L.gy));
    if (gn <= 0) return;
    const double nx = L.gx / gn, ny = L.gy / gn, tx = -ny, ty = nx;
    auto smp = [&](double x, double yy) {
        x = std::clamp(x, 0.0, W - 1.0); yy = std::clamp(yy, 0.0, H - 1.0);
        const int x0 = static_cast<int>(x), y0 = static_cast<int>(yy);
        const int x1 = std::min(x0 + 1, W - 1), y1 = std::min(y0 + 1, H - 1);
        const double fx = x - x0, fy = yy - y0;
        return (y[y0 * W + x0] * (1 - fx) + y[y0 * W + x1] * fx) * (1 - fy) + (y[y1 * W + x0] * (1 - fx) + y[y1 * W + x1] * fx) * fy;
    };
    const double a = smp(L.x - 4 * nx, L.y - 4 * ny), b = smp(L.x + 4 * nx, L.y + 4 * ny);
    const double mid = 0.5 * (a + b), half = 0.5 * (b - a), w = kEdgeW[s], d0 = dq / 4.0;
    for (int yy = L.y - kBlT - 1; yy <= L.y + kBlT + 1; ++yy)
        for (int xx = L.x - kBlT - 1; xx <= L.x + kBlT + 1; ++xx) {
            if (xx < 0 || yy < 0 || xx >= W || yy >= H) continue;
            const double ex = xx - L.x, ey = yy - L.y;
            const double dt = ex * tx + ey * ty, dn = ex * nx + ey * ny;
            if (std::abs(dt) > kBlT || std::abs(dn) > kBlN) continue;
            const double om = (1.0 - std::abs(dt) / (kBlT + 1)) * (std::abs(dn) <= kBlN / 2 ? 1.0 : (kBlN - std::abs(dn)) / (kBlN / 2.0));
            const double T = mid + half * std::clamp((dn - d0) / w, -1.0, 1.0);
            const size_t i = static_cast<size_t>(yy) * W + xx;
            f(i, y[i] + om * (T - y[i]));
        }
}
}  // namespace

void code_band_lines(SymIO& io, CMModel& m, const std::vector<double>& y0, int W, int H, const Plane* org, double off,
                     double lambda, int bit_depth, std::vector<double>& y1) {
    y1 = y0;
    Plane tmp(W, H);
    for (size_t i = 0; i < y0.size(); ++i) tmp.v[i] = static_cast<int32_t>(std::lround(y0[i] + off));
    // ぼけた LL 合成画像のエッジは弱いので下限を下げる
    std::vector<LineSeg> lines = extract_lines(tmp, W, H, bit_depth, 32);
    int prev = 0;
    for (const LineSeg& L : lines) {
        int on = 0, dq = 0, s = 1;
        if (io.enc && org) {
            // 線の近傍の SSE (原画との差) を、適用しない場合と各 (dq, s) で比べる
            double base = 0;
            band_line_apply(y1, W, H, L, 0, 1, [&](size_t i, double) { const double e = y1[i] + off - org->v[i]; base += e * e; });
            double best = base - lambda * 7.0;  // 位置・鋭さ・フラグの符号量 (約 7 ビット) に見合う改善だけ採る
            for (int s2 = 0; s2 < 4; ++s2)
                for (int d2 = -4; d2 <= 4; ++d2) {
                    double e2 = 0;
                    band_line_apply(y1, W, H, L, d2, s2, [&](size_t i, double v) { const double e = v + off - org->v[i]; e2 += e * e; });
                    if (e2 < best) { best = e2; on = 1; dq = d2; s = s2; }
                }
        }
        on = io.bit(m, 20, static_cast<uint32_t>(prev), 0, on);
        prev = on;
        if (!on) continue;
        dq = io.sint(m, 21, 0, dq);
        s = static_cast<int>(io.uint(m, 22, 0, static_cast<uint32_t>(s)));
        if (std::abs(dq) > 8 || s > 3) throw std::runtime_error("corrupt stream: band line");
        std::vector<std::pair<size_t, double>> upd;
        band_line_apply(y1, W, H, L, dq, s, [&](size_t i, double v) { upd.push_back({i, v}); });
        for (const auto& u : upd) y1[u.first] = u.second;
    }
}

}  // namespace fvc
