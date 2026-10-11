#include "lines.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

namespace fvc {

std::vector<LineSeg> extract_lines(const Plane& ref, int vw, int vh, int bit_depth) {
    std::vector<LineSeg> out;
    const int W = std::min(vw, ref.w), H = std::min(vh, ref.h);
    const int32_t thr = 96 << std::max(0, bit_depth - 8);  // |gx|+|gy| の下限 (平坦なセルには線を置かない)
    auto at = [&](int x, int y) { return ref.at(std::clamp(x, 0, W - 1), std::clamp(y, 0, H - 1)); };
    for (int cy = 0; cy < H; cy += kLineCell)
        for (int cx = 0; cx < W; cx += kLineCell) {
            LineSeg best;
            int32_t bm = thr - 1;
            // セルの内側 (端 2 画素を除く) で Sobel 勾配が最大の点
            for (int y = cy + 2; y < std::min(cy + kLineCell - 2, H - 1); ++y)
                for (int x = cx + 2; x < std::min(cx + kLineCell - 2, W - 1); ++x) {
                    const int32_t gx = (at(x + 1, y - 1) + 2 * at(x + 1, y) + at(x + 1, y + 1)) - (at(x - 1, y - 1) + 2 * at(x - 1, y) + at(x - 1, y + 1));
                    const int32_t gy = (at(x - 1, y + 1) + 2 * at(x, y + 1) + at(x + 1, y + 1)) - (at(x - 1, y - 1) + 2 * at(x, y - 1) + at(x + 1, y - 1));
                    const int32_t m = std::abs(gx) + std::abs(gy);
                    if (m > bm) { bm = m; best.x = x; best.y = y; best.gx = gx; best.gy = gy; }
                }
            if (bm >= thr) out.push_back(best);
        }
    return out;
}

namespace {
constexpr int kWin = 8;  // 動き推定の窓 (線上の点を中心に 2*kWin 四方)

int64_t window_sad(const Plane& cur, const Plane& ref, int x0, int y0, int mvx, int mvy, int64_t bound) {
    // 1/4 画素は双線形 (推定用の近似)
    const int ix = mvx >> 2, iy = mvy >> 2, fx = mvx & 3, fy = mvy & 3;
    const int w00 = (4 - fx) * (4 - fy), w10 = fx * (4 - fy), w01 = (4 - fx) * fy, w11 = fx * fy;
    auto r = [&](int x, int y) { return ref.at(std::clamp(x, 0, ref.w - 1), std::clamp(y, 0, ref.h - 1)); };
    int64_t sad = 0;
    for (int y = y0 - kWin; y < y0 + kWin && sad < bound; ++y) {
        const int cy = std::clamp(y, 0, cur.h - 1);
        for (int x = x0 - kWin; x < x0 + kWin; ++x) {
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
        int px, py;
        gm.mv_at(L.x, L.y, px, py);
        const double c0 = static_cast<double>(window_sad(cur, ref, L.x, L.y, px, py, INT64_MAX)) + lsad * 1.0;
        // 整数探索: 予測 (大域動き) の周りと 0 の周り
        int bx = (px + 2) >> 2, by = (py + 2) >> 2;
        double bc = 1e300;
        auto test = [&](int ix, int iy) {
            const double c = static_cast<double>(window_sad(cur, ref, L.x, L.y, ix * 4, iy * 4, static_cast<int64_t>(std::min(bc, 1e15)) + 1)) +
                             lsad * mv_bits(ix * 4 - px, iy * 4 - py);
            if (c < bc) { bc = c; bx = ix; by = iy; }
        };
        const int cx = bx, cy = by;
        for (int dy = -range; dy <= range; ++dy)
            for (int dx = -range; dx <= range; ++dx) test(cx + dx, cy + dy);
        test(0, 0);
        // 1/2, 1/4 画素
        int mx = bx * 4, my = by * 4;
        double sc = static_cast<double>(window_sad(cur, ref, L.x, L.y, mx, my, INT64_MAX)) + lsad * mv_bits(mx - px, my - py);
        for (int st = 2; st >= 1; st >>= 1) {
            const int ox = mx, oy = my;
            for (int dy = -st; dy <= st; dy += st)
                for (int dx = -st; dx <= st; dx += st) {
                    if (!dx && !dy) continue;
                    const double c = static_cast<double>(window_sad(cur, ref, L.x, L.y, ox + dx, oy + dy, INT64_MAX)) +
                                     lsad * mv_bits(ox + dx - px, oy + dy - py);
                    if (c < sc) { sc = c; mx = ox + dx; my = oy + dy; }
                }
        }
        // 動き場は周囲のブロックにも効くため、変更には窓の SAD 改善が 1 割以上あることも求める
        L.changed = (mx != px || my != py) && sc + lsad * 1.0 < c0 * 0.9;
        L.mvx = L.changed ? mx : px;
        L.mvy = L.changed ? my : py;
    }
}

void code_line_motion(SymIO& io, CMModel& m, std::vector<LineSeg>& lines, const GlobalModel& gm) {
    // 文脈: 左と上のセルの線が変わったか (動物体の線はまとまって変わる)
    int cw = 1, ch = 1;
    for (const LineSeg& L : lines) { cw = std::max(cw, L.x / kLineCell + 1); ch = std::max(ch, L.y / kLineCell + 1); }
    std::vector<int8_t> st(static_cast<size_t>(cw) * ch, 0);  // 0: 線なし/未変更, 1: 変更
    int pdx = 0, pdy = 0;  // 直前に変わった線の差分 (差分の予測に使う: 動物体の線は同じ向きに動く)
    for (LineSeg& L : lines) {
        int px, py;
        gm.mv_at(L.x, L.y, px, py);
        const int cx = L.x / kLineCell, cy = L.y / kLineCell;
        const uint32_t ctx = static_cast<uint32_t>((cx > 0 ? st[static_cast<size_t>(cy) * cw + cx - 1] : 0) +
                                                   2 * (cy > 0 ? st[static_cast<size_t>(cy - 1) * cw + cx] : 0));
        L.changed = io.bit(m, 0, ctx, 0, L.changed) != 0;
        st[static_cast<size_t>(cy) * cw + cx] = L.changed ? 1 : 0;
        if (!L.changed) { L.mvx = px; L.mvy = py; continue; }
        const int dx = pdx + io.sint(m, 1, 0, L.mvx - px - pdx);
        const int dy = pdy + io.sint(m, 2, 0, L.mvy - py - pdy);
        if (std::abs(dx) > 4096 || std::abs(dy) > 4096) throw std::runtime_error("corrupt stream: line motion");
        L.mvx = px + dx;
        L.mvy = py + dy;
        pdx = dx; pdy = dy;
    }
}

void build_line_field(const std::vector<LineSeg>& lines, const GlobalModel& gm, int w4, int h4, std::vector<int16_t>& field) {
    field.assign(static_cast<size_t>(w4) * h4 * 2, 0);
    // 線をセルに登録 (近傍探索用)
    const int cw = (w4 * 4 + kLineCell - 1) / kLineCell, ch = (h4 * 4 + kLineCell - 1) / kLineCell;
    std::vector<std::vector<int>> cell(static_cast<size_t>(cw) * ch);
    for (size_t i = 0; i < lines.size(); ++i) {
        const int cx = std::min(lines[i].x / kLineCell, cw - 1), cy = std::min(lines[i].y / kLineCell, ch - 1);
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
                        const int dx = L.x - x, dy = L.y - y;
                        const int d2 = dx * dx + dy * dy;
                        if (d2 > kR2) continue;
                        const int64_t w = 65536 / (d2 + kSoft);
                        ws += w; sx += w * L.mvx; sy += w * L.mvy;
                    }
            auto rdiv = [](int64_t a, int64_t b) { return a >= 0 ? (a + b / 2) / b : -((-a + b / 2) / b); };
            field[(static_cast<size_t>(by) * w4 + bx) * 2] = static_cast<int16_t>(std::clamp<int64_t>(rdiv(sx, ws), -16000, 16000));
            field[(static_cast<size_t>(by) * w4 + bx) * 2 + 1] = static_cast<int16_t>(std::clamp<int64_t>(rdiv(sy, ws), -16000, 16000));
        }
}

}  // namespace fvc
