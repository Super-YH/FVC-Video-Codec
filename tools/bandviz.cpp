// 帯域画像の解析ツール: PQMF 帯域を可視化し、帯域間・時間方向の相関統計を出力する
//   bandviz in.y4m frame M out_prefix
//   → out_prefix_X.pgm (現フレーム帯域), _P.pgm (前フレーム帯域), _D.pgm (時間差分), 統計を標準出力
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "fvc/frame.hpp"
#include "fvc/pqmf.hpp"

using namespace fvc;
using Bands = std::vector<std::vector<double>>;

static double corr(const std::vector<double>& a, const std::vector<double>& b) {
    double ma = 0, mb = 0;
    for (size_t i = 0; i < a.size(); ++i) { ma += a[i]; mb += b[i]; }
    ma /= a.size(); mb /= b.size();
    double sab = 0, saa = 0, sbb = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double x = a[i] - ma, y = b[i] - mb;
        sab += x * y; saa += x * x; sbb += y * y;
    }
    return (saa > 0 && sbb > 0) ? sab / std::sqrt(saa * sbb) : 0.0;
}

// 帯域画像を M x M のモザイクにして PGM 出力 (帯域ごとに |v| を 99% 点で正規化、LL はそのまま)
static void write_mosaic(const std::string& path, const Bands& b, int M, int bw, int bh, bool signed_view) {
    const int W = bw * M, H = bh * M;
    std::vector<unsigned char> img(static_cast<size_t>(W) * H, 0);
    for (int k = 0; k < M * M; ++k) {
        std::vector<double> a(b[k].size());
        for (size_t i = 0; i < a.size(); ++i) a[i] = std::abs(b[k][i]);
        std::vector<double> s = a;
        std::nth_element(s.begin(), s.begin() + s.size() * 99 / 100, s.end());
        const double sc = std::max(1e-6, s[s.size() * 99 / 100]);
        const int ox = (k % M) * bw, oy = (k / M) * bh;
        double mn = 1e30, mx = -1e30;
        if (k == 0) for (double v : b[0]) { mn = std::min(mn, v); mx = std::max(mx, v); }
        for (int y = 0; y < bh; ++y)
            for (int x = 0; x < bw; ++x) {
                const double v = b[k][static_cast<size_t>(y) * bw + x];
                double g;
                if (k == 0 && !signed_view) g = (v - mn) / std::max(1e-6, mx - mn) * 255.0;
                else if (signed_view) g = 128.0 + 127.0 * std::clamp(v / sc, -1.0, 1.0);
                else g = 255.0 * std::min(1.0, std::abs(v) / sc);
                img[static_cast<size_t>(oy + y) * W + ox + x] = static_cast<unsigned char>(std::clamp(g, 0.0, 255.0));
            }
    }
    FILE* f = std::fopen(path.c_str(), "wb");
    std::fprintf(f, "P5\n%d %d\n255\n", W, H);
    std::fwrite(img.data(), 1, img.size(), f);
    std::fclose(f);
}

int main(int argc, char** argv) {
    if (argc < 5) { std::fprintf(stderr, "usage: bandviz in.y4m frame M out_prefix\n"); return 2; }
    const int fi = std::atoi(argv[2]), M = std::atoi(argv[3]);
    const std::string pre = argv[4];
    Y4MReader rd;
    if (!rd.open(argv[1])) return 1;
    Frame f, prev, cur;
    for (int i = 0; i <= fi; ++i) {
        if (!rd.read(f)) return 1;
        if (i == fi - 1) prev = f;
        if (i == fi) cur = f;
    }
    const Plane& Y = cur.p[0];
    const int W = Y.w / (M * 8) * (M * 8), H = Y.h / (M * 8) * (M * 8), bw = W / M, bh = H / M;
    auto to_vec = [&](const Plane& p) {
        std::vector<double> v(static_cast<size_t>(W) * H);
        for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) v[static_cast<size_t>(y) * W + x] = p.at(x, y) - 128.0;
        return v;
    };
    Pqmf2D fb(M, M);
    Bands X, P;
    fb.analyze(to_vec(Y), W, H, X);
    fb.analyze(to_vec(prev.p[0]), W, H, P);
    Bands D(M * M);
    for (int k = 0; k < M * M; ++k) { D[k].resize(X[k].size()); for (size_t i = 0; i < X[k].size(); ++i) D[k][i] = X[k][i] - P[k][i]; }
    write_mosaic(pre + "_X.pgm", X, M, bw, bh, false);
    write_mosaic(pre + "_P.pgm", P, M, bw, bh, false);
    write_mosaic(pre + "_D.pgm", D, M, bw, bh, true);
    write_mosaic(pre + "_Xs.pgm", X, M, bw, bh, true);

    double etot = 0;
    std::vector<double> e(M * M, 0.0), ed(M * M, 0.0);
    for (int k = 0; k < M * M; ++k) {
        for (double v : X[k]) e[k] += v * v;
        for (double v : D[k]) ed[k] += v * v;
        if (k) etot += e[k];
    }
    std::printf("band  energy%%(HF)  E(D)/E(X)  rho_t  rho_t_best(shift)  rho_x  rho_x_mirror  rho_|x|  rho_|t|\n");
    for (int k = 0; k < M * M; ++k) {
        const int kx = k % M, ky = k / M;
        // 時間相関: 同位置 / 帯域標本 ±2 シフトの最良
        const double rt = corr(X[k], P[k]);
        double best = rt; int bsx = 0, bsy = 0;
        for (int sy = -2; sy <= 2; ++sy)
            for (int sx = -2; sx <= 2; ++sx) {
                std::vector<double> a, b;
                for (int y = 2; y < bh - 2; ++y)
                    for (int x = 2; x < bw - 2; ++x) {
                        a.push_back(X[k][static_cast<size_t>(y) * bw + x]);
                        b.push_back(P[k][static_cast<size_t>(y + sy) * bw + x + sx]);
                    }
                const double r = corr(a, b);
                if (r > best) { best = r; bsx = sx; bsy = sy; }
            }
        // 帯域間相関: 左 (なければ上) の帯域。値 / 鏡像補正 / 振幅包絡
        double rx = 0, rxm = 0, rabs = 0, rtabs = 0;
        const int rk = kx > 0 ? k - 1 : (ky > 0 ? k - M : -1);
        if (rk >= 0) {
            std::vector<double> m(X[rk].size()), ax(X[k].size()), ar(X[k].size());
            for (int y = 0; y < bh; ++y)
                for (int x = 0; x < bw; ++x) {
                    const size_t i = static_cast<size_t>(y) * bw + x;
                    m[i] = (((kx > 0) ? x : y) & 1) ? -X[rk][i] : X[rk][i];
                    ax[i] = std::abs(X[k][i]);
                    ar[i] = std::abs(X[rk][i]);
                }
            rx = corr(X[k], X[rk]);
            rxm = corr(X[k], m);
            rabs = corr(ax, ar);
        }
        {
            std::vector<double> ax(X[k].size()), ap(X[k].size());
            for (size_t i = 0; i < ax.size(); ++i) { ax[i] = std::abs(X[k][i]); ap[i] = std::abs(P[k][i]); }
            rtabs = corr(ax, ap);
        }
        std::printf("%2d(%d,%d) %8.3f   %8.3f   %6.3f   %6.3f (%+d,%+d)    %6.3f  %6.3f   %6.3f  %6.3f\n", k, kx, ky,
                    k ? 100.0 * e[k] / etot : 0.0, e[k] > 0 ? ed[k] / e[k] : 0.0, rt, best, bsx, bsy, rx, rxm, rabs, rtabs);
    }
    return 0;
}
