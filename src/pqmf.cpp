#include "fvc/pqmf.hpp"

#include <cmath>
#include <stdexcept>

namespace fvc {

namespace {
constexpr double kPi = 3.14159265358979323846;

double bessel_i0(double x) {
    double s = 1.0, t = 1.0;
    for (int k = 1; k < 64; ++k) {
        t *= (x / (2.0 * k)) * (x / (2.0 * k));
        s += t;
        if (t < 1e-17 * s) break;
    }
    return s;
}

// カイザー窓付き理想低域フィルタ (遮断 wc)
std::vector<double> kaiser_lowpass(int N, double wc, double beta) {
    std::vector<double> p(N);
    const double c = (N - 1) / 2.0, i0b = bessel_i0(beta);
    for (int n = 0; n < N; ++n) {
        const double t = n - c;
        const double s = (std::abs(t) < 1e-12) ? wc / kPi : std::sin(wc * t) / (kPi * t);
        const double r = 2.0 * n / (N - 1) - 1.0;
        p[n] = s * bessel_i0(beta * std::sqrt(std::max(0.0, 1.0 - r * r))) / i0b;
    }
    return p;
}

// Lin-Vaidyanathan 基準: g = p*p の 2M 間引き成分の非ゼロ部最大値
double pr_error(const std::vector<double>& p, int M) {
    const int N = static_cast<int>(p.size());
    std::vector<double> g(2 * N - 1, 0.0);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) g[i + j] += p[i] * p[j];
    const int c = N - 1;
    double worst = 0.0;
    for (int n = 2 * M; c + n < 2 * N - 1; n += 2 * M)
        worst = std::max(worst, std::abs(g[c + n]) / std::abs(g[c]));
    return worst;
}
}  // namespace

Pqmf1D::Pqmf1D(int M, int m, double beta) : M_(M), N_(2 * m * M) {
    if (M < 1 || M > 16 || m < 1) throw std::invalid_argument("Pqmf1D: bad parameters");
    if (M == 1) {
        p_ = {1.0};
        N_ = 1;
        h_ = f_ = {{1.0}};
        gain_ = 1.0;
        return;
    }
    // 遮断周波数 wc を黄金分割探索で最適化 (目標 wc ≈ π/(2M))
    double lo = 0.5 * kPi / (2 * M), hi = 1.5 * kPi / (2 * M);
    const double gr = (std::sqrt(5.0) - 1) / 2;
    double a = hi - gr * (hi - lo), b = lo + gr * (hi - lo);
    double fa = pr_error(kaiser_lowpass(N_, a, beta), M);
    double fb = pr_error(kaiser_lowpass(N_, b, beta), M);
    for (int it = 0; it < 80; ++it) {
        if (fa < fb) { hi = b; b = a; fb = fa; a = hi - gr * (hi - lo); fa = pr_error(kaiser_lowpass(N_, a, beta), M); }
        else         { lo = a; a = b; fa = fb; b = lo + gr * (hi - lo); fb = pr_error(kaiser_lowpass(N_, b, beta), M); }
    }
    p_ = kaiser_lowpass(N_, 0.5 * (lo + hi), beta);
    build_filters();
}

void Pqmf1D::build_filters() {
    h_.assign(M_, std::vector<double>(N_));
    f_.assign(M_, std::vector<double>(N_));
    const double c = (N_ - 1) / 2.0;
    for (int k = 0; k < M_; ++k) {
        const double th = ((k & 1) ? -1.0 : 1.0) * kPi / 4.0;
        for (int n = 0; n < N_; ++n) {
            const double a = kPi / M_ * (k + 0.5) * (n - c);
            h_[k][n] = 2.0 * p_[n] * std::cos(a + th);
            f_[k][n] = 2.0 * p_[n] * std::cos(a - th);
        }
    }
    // インパルス応答から全体利得を数値校正
    gain_ = 1.0;
    const int L = M_ * ((2 * N_) / M_ + 2);
    std::vector<double> x(L, 0.0), y;
    x[0] = 1.0;
    std::vector<std::vector<double>> u;
    analyze(x, u);
    synthesize(u, y);
    gain_ = 1.0 / y[0];
}

void Pqmf1D::analyze(const std::vector<double>& x, std::vector<std::vector<double>>& out) const {
    const int L = static_cast<int>(x.size());
    if (L % M_) throw std::invalid_argument("analyze: length not multiple of M");
    const int K = L / M_;
    out.assign(M_, std::vector<double>(K, 0.0));
    for (int k = 0; k < M_; ++k)
        for (int q = 0; q < K; ++q) {
            double s = 0.0;
            for (int n = 0; n < N_; ++n) {
                int idx = (q * M_ - n) % L;
                if (idx < 0) idx += L;
                s += h_[k][n] * x[idx];
            }
            out[k][q] = s;
        }
}

void Pqmf1D::synthesize(const std::vector<std::vector<double>>& in, std::vector<double>& y) const {
    const int K = static_cast<int>(in[0].size()), L = K * M_;
    std::vector<double> t(L, 0.0);
    for (int k = 0; k < M_; ++k)
        for (int q = 0; q < K; ++q) {
            const double v = in[k][q];
            for (int n = 0; n < N_; ++n) t[(q * M_ + n) % L] += f_[k][n] * v;
        }
    // 全体遅延 N-1 を補償
    y.assign(L, 0.0);
    const int d = (N_ - 1) % L;
    for (int n = 0; n < L; ++n) y[n] = gain_ * t[(n + d) % L];
}

double Pqmf1D::synth_norm(int k) const {
    double e = 0.0;
    for (double v : f_[k]) e += v * v;
    return std::sqrt(gain_ * gain_ * e / M_);
}

Pqmf2D::Pqmf2D(int Mx, int My, int m, double beta) : fx_(Mx, m, beta), fy_(My, m, beta) {}

void Pqmf2D::analyze(const std::vector<double>& img, int W, int H,
                     std::vector<std::vector<double>>& bands) const {
    const int Mx = fx_.bands(), My = fy_.bands(), w = W / Mx, h = H / My;
    // 行方向
    std::vector<std::vector<double>> tmp(Mx, std::vector<double>(w * H));
    std::vector<double> row(W);
    std::vector<std::vector<double>> u;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) row[x] = img[y * W + x];
        fx_.analyze(row, u);
        for (int k = 0; k < Mx; ++k)
            for (int x = 0; x < w; ++x) tmp[k][y * w + x] = u[k][x];
    }
    // 列方向
    bands.assign(Mx * My, std::vector<double>(w * h));
    std::vector<double> col(H);
    for (int kx = 0; kx < Mx; ++kx)
        for (int x = 0; x < w; ++x) {
            for (int y = 0; y < H; ++y) col[y] = tmp[kx][y * w + x];
            fy_.analyze(col, u);
            for (int ky = 0; ky < My; ++ky)
                for (int y = 0; y < h; ++y) bands[ky * Mx + kx][y * w + x] = u[ky][y];
        }
}

void Pqmf2D::synthesize(const std::vector<std::vector<double>>& bands, int W, int H,
                        std::vector<double>& img) const {
    const int Mx = fx_.bands(), My = fy_.bands(), w = W / Mx, h = H / My;
    std::vector<std::vector<double>> tmp(Mx, std::vector<double>(w * H));
    std::vector<std::vector<double>> u(My, std::vector<double>(h));
    std::vector<double> col;
    for (int kx = 0; kx < Mx; ++kx)
        for (int x = 0; x < w; ++x) {
            for (int ky = 0; ky < My; ++ky)
                for (int y = 0; y < h; ++y) u[ky][y] = bands[ky * Mx + kx][y * w + x];
            fy_.synthesize(u, col);
            for (int y = 0; y < H; ++y) tmp[kx][y * w + x] = col[y];
        }
    img.assign(W * H, 0.0);
    std::vector<std::vector<double>> v(Mx, std::vector<double>(w));
    std::vector<double> row;
    for (int y = 0; y < H; ++y) {
        for (int kx = 0; kx < Mx; ++kx)
            for (int x = 0; x < w; ++x) v[kx][x] = tmp[kx][y * w + x];
        fx_.synthesize(v, row);
        for (int x = 0; x < W; ++x) img[y * W + x] = row[x];
    }
}

double psnr(const std::vector<double>& a, const std::vector<double>& b, double peak) {
    double se = 0.0;
    for (size_t i = 0; i < a.size(); ++i) se += (a[i] - b[i]) * (a[i] - b[i]);
    if (se == 0.0) return 999.0;
    return 10.0 * std::log10(peak * peak * a.size() / se);
}

}  // namespace fvc
