#include "dictionary.hpp"

#include <algorithm>
#include <cmath>

namespace fvc {

constexpr int Dictionary::kSizes[2];

Dictionary::Dictionary() : entries_(kCapacity) {
    // 予約項目 (規範生成): 各サイズについて
    //  - 低次 2D DCT 基底 (u,v) ∈ [0,4)^2 \ {(0,0)} : 15
    //  - ステップエッジ 16 方向 × 3 オフセット : 48
    //  - チェッカ / 縞 (周期 2,4) : 6
    // 振幅は ±64 (平均 0 に正規化)
    constexpr double kPi = 3.14159265358979323846;
    int id = 0;
    for (int s : kSizes) {
        auto finish = [&](std::vector<double> v) {
            double m = 0;
            for (double x : v) m += x;
            m /= v.size();
            double mx = 1e-9;
            for (double& x : v) { x -= m; mx = std::max(mx, std::abs(x)); }
            std::vector<int32_t> d(v.size());
            for (size_t i = 0; i < v.size(); ++i) d[i] = static_cast<int32_t>(std::lround(v[i] / mx * 64.0));
            add_reserved(id++, s, std::move(d));
        };
        for (int v = 0; v < 4; ++v)
            for (int u = 0; u < 4; ++u) {
                if (!u && !v) continue;
                std::vector<double> b(s * s);
                for (int y = 0; y < s; ++y)
                    for (int x = 0; x < s; ++x)
                        b[y * s + x] = std::cos(kPi * (2 * x + 1) * u / (2.0 * s)) * std::cos(kPi * (2 * y + 1) * v / (2.0 * s));
                finish(b);
            }
        for (int a = 0; a < 16; ++a)
            for (int o = -1; o <= 1; ++o) {
                const double th = kPi * a / 16.0, c = std::cos(th), sn = std::sin(th);
                std::vector<double> b(s * s);
                for (int y = 0; y < s; ++y)
                    for (int x = 0; x < s; ++x) {
                        const double d = (x - (s - 1) / 2.0) * c + (y - (s - 1) / 2.0) * sn - o * s / 4.0;
                        b[y * s + x] = d > 0 ? 1.0 : (d < 0 ? -1.0 : 0.0);
                    }
                finish(b);
            }
        for (int per : {2, 4}) {
            std::vector<double> c(s * s), h(s * s), v(s * s);
            for (int y = 0; y < s; ++y)
                for (int x = 0; x < s; ++x) {
                    c[y * s + x] = (((x / (per / 2)) + (y / (per / 2))) & 1) ? 1 : -1;
                    h[y * s + x] = ((y / (per / 2)) & 1) ? 1 : -1;
                    v[y * s + x] = ((x / (per / 2)) & 1) ? 1 : -1;
                }
            finish(c); finish(h); finish(v);
        }
    }
}

void Dictionary::add_reserved(int id, int s, std::vector<int32_t> d) {
    entries_[id].s = s;
    entries_[id].d = std::move(d);
    by_size_[s == 8 ? 0 : 1].push_back(id);
}

void Dictionary::index_remove(int id) {
    auto& v = by_size_[entries_[id].s == 8 ? 0 : 1];
    v.erase(std::remove(v.begin(), v.end(), id), v.end());
}

int Dictionary::add_from(const Plane& p, int x, int y, int s) {
    const int id = next_dyn_;
    next_dyn_ = next_dyn_ + 1 >= kCapacity ? kReserved : next_dyn_ + 1;
    if (entries_[id].s) index_remove(id);
    Entry& e = entries_[id];
    e.s = s;
    e.d.resize(static_cast<size_t>(s) * s);
    int64_t sum = 0;
    for (int j = 0; j < s; ++j)
        for (int i = 0; i < s; ++i) {
            const int32_t v = p.at(std::min(x + i, p.w - 1), std::min(y + j, p.h - 1));
            e.d[j * s + i] = v;
            sum += v;
        }
    const int32_t mean = static_cast<int32_t>(sum / (static_cast<int64_t>(s) * s));
    for (auto& v : e.d) v -= mean;  // 平均 0 で保持
    by_size_[s == 8 ? 0 : 1].push_back(id);
    return id;
}

void Dictionary::reset_dynamic() {
    for (int id = kReserved; id < kCapacity; ++id)
        if (entries_[id].s) { index_remove(id); entries_[id] = Entry{}; }
    next_dyn_ = kReserved;
}

}  // namespace fvc
