#include "fvc/transform.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <mutex>
#include <stdexcept>

namespace fvc {

namespace {
constexpr double kPi = 3.14159265358979323846;
}

namespace {
std::vector<double> build_matrix(TxType t, int N) {
    std::vector<double> m(static_cast<size_t>(N) * N);
    for (int k = 0; k < N; ++k)
        for (int n = 0; n < N; ++n) {
            double v = 0;
            switch (t) {
            case TxType::DCT2:
                v = std::sqrt((k == 0 ? 1.0 : 2.0) / N) * std::cos(kPi * (2 * n + 1) * k / (2.0 * N));
                break;
            case TxType::DST7:
                v = std::sqrt(4.0 / (2 * N + 1)) * std::sin(kPi * (2 * n + 1) * (k + 1) / (2.0 * N + 1));
                break;
            case TxType::IDTX:
                v = (k == n) ? 1.0 : 0.0;
                break;
            }
            m[static_cast<size_t>(k) * N + n] = v;
        }
    return m;
}

int log2i(int n) { int l = 0; while ((1 << l) < n) ++l; return l; }

struct MatrixCache {
    // 2 冪サイズはロックなしの固定表、それ以外はロック付き map
    std::vector<double> pow2[3][11];
    std::mutex mu;
    std::map<std::pair<int, int>, std::vector<double>> other;
    MatrixCache() {
        for (int t = 0; t < 3; ++t)
            for (int l = 0; l <= 10; ++l) pow2[t][l] = build_matrix(static_cast<TxType>(t), 1 << l);
    }
};
MatrixCache& cache() { static MatrixCache c; return c; }

// 1D 順変換 (in/out はストライド付き)。DCT-II は偶奇対称で積和を半減。
void fwd1d(TxType t, const double* T, int N, const double* in, int is, double* out, int os) {
    if (t == TxType::IDTX) { for (int k = 0; k < N; ++k) out[k * os] = in[k * is]; return; }
    if (t == TxType::DCT2 && N >= 4) {
        const int h = N / 2;
        double e[512], o[512];
        for (int n = 0; n < h; ++n) { e[n] = in[n * is] + in[(N - 1 - n) * is]; o[n] = in[n * is] - in[(N - 1 - n) * is]; }
        for (int k = 0; k < N; ++k) {
            const double* row = T + static_cast<size_t>(k) * N;
            const double* v = (k & 1) ? o : e;
            double sum = 0;
            for (int n = 0; n < h; ++n) sum += row[n] * v[n];
            out[k * os] = sum;
        }
        return;
    }
    for (int k = 0; k < N; ++k) {
        const double* row = T + static_cast<size_t>(k) * N;
        double sum = 0;
        for (int n = 0; n < N; ++n) sum += row[n] * in[n * is];
        out[k * os] = sum;
    }
}

// 1D 逆変換。kmax 以降の係数は 0 として枝刈り。
void inv1d(TxType t, const double* T, int N, const double* in, int is, double* out, int os, int kmax) {
    if (t == TxType::IDTX) { for (int n = 0; n < N; ++n) out[n * os] = in[n * is]; return; }
    if (t == TxType::DCT2 && N >= 4) {
        const int h = N / 2;
        for (int n = 0; n < h; ++n) {
            double e = 0, o = 0;
            for (int k = 0; k < kmax; k += 2) e += T[static_cast<size_t>(k) * N + n] * in[k * is];
            for (int k = 1; k < kmax; k += 2) o += T[static_cast<size_t>(k) * N + n] * in[k * is];
            out[n * os] = e + o;
            out[(N - 1 - n) * os] = e - o;
        }
        return;
    }
    for (int n = 0; n < N; ++n) {
        double sum = 0;
        for (int k = 0; k < kmax; ++k) sum += T[static_cast<size_t>(k) * N + n] * in[k * is];
        out[n * os] = sum;
    }
}
}  // namespace

const std::vector<double>& tx_matrix(TxType t, int N) {
    MatrixCache& c = cache();
    const int l = log2i(N);
    if ((1 << l) == N && l <= 10) return c.pow2[static_cast<int>(t)][l];
    std::lock_guard<std::mutex> lk(c.mu);
    auto key = std::make_pair(static_cast<int>(t), N);
    auto it = c.other.find(key);
    if (it != c.other.end()) return it->second;
    return c.other.emplace(key, build_matrix(t, N)).first->second;
}

void forward_2d(TxType th, TxType tv, const double* in, int w, int h, double* out) {
    const double* Th = tx_matrix(th, w).data();
    const double* Tv = tx_matrix(tv, h).data();
    std::vector<double> t(static_cast<size_t>(w) * h);
    for (int y = 0; y < h; ++y) fwd1d(th, Th, w, in + static_cast<size_t>(y) * w, 1, t.data() + static_cast<size_t>(y) * w, 1);
    for (int x = 0; x < w; ++x) fwd1d(tv, Tv, h, t.data() + x, w, out + x, w);
}

void inverse_2d(TxType th, TxType tv, const double* in, int w, int h, double* out) {
    const double* Th = tx_matrix(th, w).data();
    const double* Tv = tx_matrix(tv, h).data();
    // 非ゼロ係数の範囲 (行 rmax, 列 cmax) で枝刈り
    int rmax = 0, cmax = 0;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            if (in[static_cast<size_t>(y) * w + x] != 0.0) { rmax = std::max(rmax, y + 1); cmax = std::max(cmax, x + 1); }
    if (rmax == 0) { std::fill(out, out + static_cast<size_t>(w) * h, 0.0); return; }
    std::vector<double> t(static_cast<size_t>(w) * h, 0.0);
    for (int x = 0; x < cmax; ++x) inv1d(tv, Tv, h, in + x, w, t.data() + x, w, tv == TxType::IDTX ? h : rmax);
    for (int y = 0; y < h; ++y)
        inv1d(th, Th, w, t.data() + static_cast<size_t>(y) * w, 1, out + static_cast<size_t>(y) * w, 1, th == TxType::IDTX ? w : cmax);
}

TnsFilter tns_design(const double* c, int n, int max_order, int qbits) {
    TnsFilter f;
    f.qbits = qbits;
    // 自己相関 (周波数方向) → Levinson-Durbin → PARCOR
    std::vector<double> R(max_order + 1, 0.0);
    for (int l = 0; l <= max_order; ++l)
        for (int k = l; k < n; ++k) R[l] += c[k] * c[k - l];
    if (R[0] <= 0) return f;
    R[0] *= 1.0 + 1e-9;
    std::vector<double> a(max_order + 1, 0.0), tmp;
    double err = R[0];
    std::vector<double> parcor;
    for (int i = 1; i <= max_order; ++i) {
        double acc = R[i];
        for (int j = 1; j < i; ++j) acc += a[j] * R[i - j];
        const double k = -acc / err;
        parcor.push_back(k);
        tmp = a;
        for (int j = 1; j < i; ++j) a[j] = tmp[j] + k * tmp[i - j];
        a[i] = k;
        err *= (1 - k * k);
        if (err <= 0) break;
    }
    // arcsin 量子化: q = round(asin(r) * (2^(b-1)-0.5) / (π/2))
    const double sc = ((1 << (qbits - 1)) - 0.5) / (kPi / 2);
    const int32_t qmax = (1 << (qbits - 1)) - 1;
    for (double r : parcor) {
        int32_t q = static_cast<int32_t>(std::lround(std::asin(r) * sc));
        q = std::max(-qmax, std::min(qmax, q));
        f.parcor_q.push_back(q);
    }
    while (!f.parcor_q.empty() && f.parcor_q.back() == 0) f.parcor_q.pop_back();
    f.order = static_cast<int>(f.parcor_q.size());
    return f;
}

std::vector<double> tns_lpc_from_parcor(const TnsFilter& f) {
    const double sc = ((1 << (f.qbits - 1)) - 0.5) / (kPi / 2);
    std::vector<double> a(f.order + 1, 0.0), tmp;
    a[0] = 1.0;
    for (int i = 1; i <= f.order; ++i) {
        const double k = std::sin(f.parcor_q[i - 1] / sc);
        tmp = a;
        for (int j = 1; j < i; ++j) a[j] = tmp[j] + k * tmp[i - j];
        a[i] = k;
    }
    return a;
}

void tns_analysis(const TnsFilter& f, const double* c, int n, double* e) {
    const auto a = tns_lpc_from_parcor(f);
    for (int k = 0; k < n; ++k) {
        double s = c[k];
        for (int i = 1; i <= f.order && k - i >= 0; ++i) s += a[i] * c[k - i];
        e[k] = s;
    }
}

void tns_synthesis(const TnsFilter& f, const double* e, int n, double* c) {
    const auto a = tns_lpc_from_parcor(f);
    for (int k = 0; k < n; ++k) {
        double s = e[k];
        for (int i = 1; i <= f.order && k - i >= 0; ++i) s -= a[i] * c[k - i];
        c[k] = s;
    }
}

}  // namespace fvc
