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

const std::vector<double>& tx_matrix(TxType t, int N) {
    static std::map<std::pair<int, int>, std::vector<double>> cache;
    static std::mutex mu;
    std::lock_guard<std::mutex> lk(mu);
    auto key = std::make_pair(static_cast<int>(t), N);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    std::vector<double> m(N * N);
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
            m[k * N + n] = v;
        }
    return cache.emplace(key, std::move(m)).first->second;
}

static void apply_rows(const std::vector<double>& T, bool inv, const double* in, int w, int h, double* out) {
    std::vector<double> r(w);
    for (int y = 0; y < h; ++y) {
        for (int k = 0; k < w; ++k) {
            double s = 0;
            for (int n = 0; n < w; ++n) s += (inv ? T[n * w + k] : T[k * w + n]) * in[y * w + n];
            r[k] = s;
        }
        for (int k = 0; k < w; ++k) out[y * w + k] = r[k];
    }
}

static void apply_cols(const std::vector<double>& T, bool inv, const double* in, int w, int h, double* out) {
    std::vector<double> c(h);
    for (int x = 0; x < w; ++x) {
        for (int k = 0; k < h; ++k) {
            double s = 0;
            for (int n = 0; n < h; ++n) s += (inv ? T[n * h + k] : T[k * h + n]) * in[n * w + x];
            c[k] = s;
        }
        for (int k = 0; k < h; ++k) out[k * w + x] = c[k];
    }
}

void forward_2d(TxType th, TxType tv, const double* in, int w, int h, double* out) {
    std::vector<double> t(w * h);
    apply_rows(tx_matrix(th, w), false, in, w, h, t.data());
    apply_cols(tx_matrix(tv, h), false, t.data(), w, h, out);
}

void inverse_2d(TxType th, TxType tv, const double* in, int w, int h, double* out) {
    std::vector<double> t(w * h);
    apply_cols(tx_matrix(tv, h), true, in, w, h, t.data());
    apply_rows(tx_matrix(th, w), true, t.data(), w, h, out);
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
