#include "fvc/quant.hpp"

#include <cmath>
#include <stdexcept>

namespace fvc {

void nearest_zn(const double* x, int n, double* y) {
    for (int i = 0; i < n; ++i) y[i] = std::nearbyint(x[i]);
}

void nearest_dn(const double* x, int n, double* y) {
    long long sum = 0;
    int worst = 0;
    double wdist = -1.0;
    for (int i = 0; i < n; ++i) {
        y[i] = std::nearbyint(x[i]);
        sum += static_cast<long long>(y[i]);
        const double d = std::abs(x[i] - y[i]);
        if (d > wdist) { wdist = d; worst = i; }
    }
    if (sum & 1) {
        // 誤差最大の座標を逆方向へ丸め直す
        y[worst] += (x[worst] > y[worst]) ? 1.0 : -1.0;
    }
}

void nearest_e8(const double* x, double* y) {
    double a[8], b[8], xs[8];
    nearest_dn(x, 8, a);
    for (int i = 0; i < 8; ++i) xs[i] = x[i] - 0.5;
    nearest_dn(xs, 8, b);
    for (int i = 0; i < 8; ++i) b[i] += 0.5;
    double da = 0, db = 0;
    for (int i = 0; i < 8; ++i) {
        da += (x[i] - a[i]) * (x[i] - a[i]);
        db += (x[i] - b[i]) * (x[i] - b[i]);
    }
    for (int i = 0; i < 8; ++i) y[i] = (da <= db) ? a[i] : b[i];
}

static int32_t floordiv2(int32_t v) { return (v >= 0) ? v / 2 : -((-v + 1) / 2); }

LatticeIndex lattice_to_index(Lattice L, const double* y, int n) {
    LatticeIndex idx;
    switch (L) {
    case Lattice::Zn:
        for (int i = 0; i < n; ++i) idx.z.push_back(static_cast<int32_t>(std::lround(y[i])));
        break;
    case Lattice::Dn: {
        int64_t s = 0;
        for (int i = 0; i < n - 1; ++i) {
            const int32_t v = static_cast<int32_t>(std::lround(y[i]));
            idx.z.push_back(v);
            s += v;
        }
        const int32_t last = static_cast<int32_t>(std::lround(y[n - 1]));
        const int32_t par = static_cast<int32_t>(s & 1);  // last ≡ par (mod 2)
        idx.z.push_back(floordiv2(last - par));
        break;
    }
    case Lattice::E8: {
        if (n != 8) throw std::invalid_argument("E8 requires n==8");
        int32_t k[8];
        for (int i = 0; i < 8; ++i) k[i] = static_cast<int32_t>(std::lround(2.0 * y[i]));
        idx.coset = k[0] & 1;
        // 2y = c + 2z, z ∈ D8 (和偶数) — Σk ≡ 0 (mod 4) と同値
        int32_t z[8];
        for (int i = 0; i < 8; ++i) z[i] = (k[i] - idx.coset) / 2;
        int64_t s = 0;
        for (int i = 0; i < 7; ++i) { idx.z.push_back(z[i]); s += z[i]; }
        // E8 = D8 ∪ (D8 + ½) で D8+½ は z_i = y_i - ½ が D8 → Σz 偶数 (両コセット共通)
        const int32_t par = static_cast<int32_t>(s & 1);
        idx.z.push_back(floordiv2(z[7] - par));
        break;
    }
    }
    return idx;
}

void index_to_lattice(Lattice L, const LatticeIndex& idx, int n, double* y) {
    switch (L) {
    case Lattice::Zn:
        for (int i = 0; i < n; ++i) y[i] = idx.z[i];
        break;
    case Lattice::Dn: {
        int64_t s = 0;
        for (int i = 0; i < n - 1; ++i) { y[i] = idx.z[i]; s += idx.z[i]; }
        y[n - 1] = 2.0 * idx.z[n - 1] + static_cast<double>(s & 1);
        break;
    }
    case Lattice::E8: {
        int64_t s = 0;
        for (int i = 0; i < 7; ++i) { y[i] = idx.z[i] + 0.5 * idx.coset; s += idx.z[i]; }
        y[7] = 2.0 * idx.z[7] + static_cast<double>(s & 1) + 0.5 * idx.coset;
        break;
    }
    }
}

void lattice_quantize(Lattice L, const double* x, int n, double step, double* xq, LatticeIndex* idx) {
    std::vector<double> t(n), q(n);
    for (int i = 0; i < n; ++i) t[i] = x[i] / step;
    switch (L) {
    case Lattice::Zn: nearest_zn(t.data(), n, q.data()); break;
    case Lattice::Dn: nearest_dn(t.data(), n, q.data()); break;
    case Lattice::E8: nearest_e8(t.data(), q.data()); break;
    }
    if (idx) *idx = lattice_to_index(L, q.data(), n);
    for (int i = 0; i < n; ++i) xq[i] = q[i] * step;
}

int32_t NonlinearSQ::quantize(double x) const {
    const double v = std::pow(std::abs(x), gamma) / step;
    const int32_t q = static_cast<int32_t>(std::floor(v + 1.0 - theta));
    return x < 0 ? -q : q;
}

double NonlinearSQ::dequantize(int32_t q) const {
    if (q == 0) return 0.0;
    const double m = (std::abs(q) + delta) * step;
    const double r = std::pow(std::max(m, 0.0), 1.0 / gamma);
    return q < 0 ? -r : r;
}

RandomProjection::RandomProjection(int m, int n, uint64_t seed) : m_(m), n_(n), P_(m * n), Ginv_(m * m) {
    if (m <= 0 || n <= 0 || m > n) throw std::invalid_argument("RandomProjection: need 0<m<=n");
    SplitMix64 rng(seed);
    const double s = 1.0 / std::sqrt(static_cast<double>(m));
    for (auto& v : P_) v = (rng.next() >> 63) ? s : -s;
    // G = P P^T を Gauss-Jordan で逆行列化
    std::vector<double> A(m * 2 * m, 0.0);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            double g = 0;
            for (int k = 0; k < n; ++k) g += P_[i * n + k] * P_[j * n + k];
            A[i * 2 * m + j] = g;
        }
        A[i * 2 * m + m + i] = 1.0;
    }
    for (int c = 0; c < m; ++c) {
        int piv = c;
        for (int r = c + 1; r < m; ++r)
            if (std::abs(A[r * 2 * m + c]) > std::abs(A[piv * 2 * m + c])) piv = r;
        if (std::abs(A[piv * 2 * m + c]) < 1e-12) throw std::runtime_error("RandomProjection: singular");
        for (int j = 0; j < 2 * m; ++j) std::swap(A[c * 2 * m + j], A[piv * 2 * m + j]);
        const double d = A[c * 2 * m + c];
        for (int j = 0; j < 2 * m; ++j) A[c * 2 * m + j] /= d;
        for (int r = 0; r < m; ++r) {
            if (r == c) continue;
            const double f = A[r * 2 * m + c];
            for (int j = 0; j < 2 * m; ++j) A[r * 2 * m + j] -= f * A[c * 2 * m + j];
        }
    }
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < m; ++j) Ginv_[i * m + j] = A[i * 2 * m + m + j];
}

void RandomProjection::project(const double* x, double* y) const {
    for (int i = 0; i < m_; ++i) {
        double s = 0;
        for (int k = 0; k < n_; ++k) s += P_[i * n_ + k] * x[k];
        y[i] = s;
    }
}

void RandomProjection::reconstruct(const double* y, double* x) const {
    std::vector<double> w(m_, 0.0);
    for (int i = 0; i < m_; ++i)
        for (int j = 0; j < m_; ++j) w[i] += Ginv_[i * m_ + j] * y[j];
    for (int k = 0; k < n_; ++k) {
        double s = 0;
        for (int i = 0; i < m_; ++i) s += P_[i * n_ + k] * w[i];
        x[k] = s;
    }
}

double pre_forward(PreNonlin t, double x, double a) {
    switch (t) {
    case PreNonlin::Identity: return x;
    case PreNonlin::Power: return std::copysign(std::pow(std::abs(x), a), x);
    case PreNonlin::Asinh: return std::asinh(x / a);
    case PreNonlin::SignedLog: return std::copysign(std::log1p(std::abs(x) / a), x);
    }
    return x;
}

double pre_inverse(PreNonlin t, double y, double a) {
    switch (t) {
    case PreNonlin::Identity: return y;
    case PreNonlin::Power: return std::copysign(std::pow(std::abs(y), 1.0 / a), y);
    case PreNonlin::Asinh: return a * std::sinh(y);
    case PreNonlin::SignedLog: return std::copysign(a * std::expm1(std::abs(y)), y);
    }
    return y;
}

}  // namespace fvc
