// FVC 内部 (符号器のみ): 格子丸め SPSA-Adam (近傍継承つき) — 整数格子上の少数変数の最小化
//
//  Adam を符号器の探索に合わせて変えたもの:
//   1. SPSA 勾配: 全座標を ±c (符号は Rademacher) で同時に揺らし、2 回の評価 f(x + cΔ), f(x − cΔ) から
//      勾配 g_i = (f+ − f−) / (2 c Δ_i) を推定する (次元によらず 2 回)
//   2. 格子丸め + 信頼領域: Adam の正規化歩幅 lr · m̂ / (√v̂ + ε) を整数に丸めて動かし、改善したときだけ採る。
//      改善しない・歩幅が 0 に丸まるときは lr と c を半分にし、lr < 0.5 で終了 (離散格子の自然な収束判定)
//   3. 近傍継承: 隣のブロックで最後に得た (m, v) を減衰させて初期値にする。動きは空間的に連続なので、
//      最初から正しい向きに進み、数歩で収束する
//   4. ヒューリスティック初期化 (ハイブリッド): 呼び出し側が安価な推定 (予測子・粗推定・近傍・大域動き等) で
//      開始点を作り、最良の開始点から始める。開始点がすでに目標値以下なら探索しない
//   5. 可変学習率: 成功で ×1.5 (上限 16)、失敗で ×0.5 (Rprop 風)。初期値は近傍の最後の歩幅から
//   6. 早期打ち切り: 目標値に達したら、または歩幅 < 1 で 2 回続けて失敗したら終了 (ReduceLROnPlateau + patience)
//  機械学習の学習高速化の手法をそのまま適用したもの:
//   7. NAdam: Nesterov の先読み運動量 β1·m̂ + (1−β1)·g/(1−β1^t) で歩幅を決める (少ない歩数で収束)
//   8. Lookahead (Zhang+ 2019): 遅い重み φ を持ち、k = 3 歩ごとに φ と現在点の中点を試して φ を更新
//      (SPSA の雑音で振動するのを抑える)
//   9. 勾配クリップ: |g| を √v̂ の 3 倍で頭打ち (外れ値の勾配で飛ばない)
//  10. 評価のメモ化: 同じ点を二度評価しない
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <unordered_map>

namespace fvc {

template <int D>
struct AdamState {
    std::array<double, D> m{}, v{};
    double last_step = 0;  // 最後に成功した移動の大きさ (次のブロックの初期学習率に使う)
    bool valid = false;
};

// f: std::array<int, D> -> double (小さいほど良い)。lo/hi: 各座標の範囲。
// starts: 開始点 (n 個)。inherit: 近傍の状態 (nullptr 可)。out_state: 最終状態 (次のブロックへ継承)
template <int D, class F>
std::array<int, D> lattice_adam(F&& f, const std::array<int, D>* starts, int n, const std::array<int, D>& lo,
                                const std::array<int, D>& hi, double lr0, const AdamState<D>* inherit,
                                AdamState<D>* out_state, uint64_t seed, int max_iter = 24, double target = -1e300) {
    auto clampv = [&](std::array<int, D> x) {
        for (int i = 0; i < D; ++i) x[i] = std::clamp(x[i], lo[i], hi[i]);
        return x;
    };
    std::unordered_map<uint64_t, double> memo;
    auto key = [&](const std::array<int, D>& x) {
        uint64_t k = 1469598103934665603ull;
        for (int i = 0; i < D; ++i) k = (k ^ static_cast<uint64_t>(static_cast<uint32_t>(x[i]))) * 1099511628211ull;
        return k;
    };
    auto eval = [&](const std::array<int, D>& x) {
        const uint64_t k = key(x);
        auto it = memo.find(k);
        if (it != memo.end()) return it->second;
        const double v = f(x);
        memo.emplace(k, v);
        return v;
    };
    std::array<int, D> x = clampv(starts[0]);
    double fx = eval(x);
    for (int k = 1; k < n; ++k) {
        const std::array<int, D> y = clampv(starts[k]);
        const double fy = eval(y);
        if (fy < fx) { fx = fy; x = y; }
    }
    if (fx <= target) return x;  // 開始点で十分
    AdamState<D> st;
    double lr = lr0;
    if (inherit && inherit->valid) {
        for (int i = 0; i < D; ++i) { st.m[i] = 0.5 * inherit->m[i]; st.v[i] = 0.5 * inherit->v[i]; }
        if (inherit->last_step > 0) lr = std::clamp(1.5 * inherit->last_step, 1.0, lr0 * 2);
    }
    constexpr double b1 = 0.8, b2 = 0.95, eps = 1e-9;
    double c = std::max(1.0, lr * 0.5);
    int fails = 0, succ = 0;
    std::array<int, D> phi = x;  // Lookahead の遅い重み
    uint64_t rs = seed * 0x9E3779B97F4A7C15ull + 0x632BE59BD9B4E019ull;
    auto rnd = [&]() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; };
    for (int t = 1; t <= max_iter && lr >= 0.5 && fx > target && fails < 2; ++t) {
        std::array<int, D> d{};
        const int ci = std::max(1, static_cast<int>(std::lround(c)));
        for (int i = 0; i < D; ++i) d[i] = (rnd() >> 63) ? 1 : -1;
        std::array<int, D> xp = x, xm = x;
        for (int i = 0; i < D; ++i) { xp[i] += ci * d[i]; xm[i] -= ci * d[i]; }
        xp = clampv(xp); xm = clampv(xm);
        const double fp = eval(xp), fm = eval(xm);
        // 揺らした点そのものが良ければ採る (評価の再利用)
        if (fp < fx || fm < fx) {
            if (fp <= fm) { fx = fp; x = xp; } else { fx = fm; x = xm; }
            st.last_step = ci;
        }
        double num = 0;
        const double bc2p = 1 - std::pow(b2, std::max(t - 1, 1));
        std::array<double, D> gs{};
        for (int i = 0; i < D; ++i) {
            double g = (fp - fm) / (2.0 * ci * d[i]);
            if (t > 1) {  // 勾配クリップ
                const double lim = 3.0 * std::sqrt(st.v[i] / bc2p) + 1e-9;
                g = std::clamp(g, -lim, lim);
            }
            gs[i] = g;
            st.m[i] = b1 * st.m[i] + (1 - b1) * g;
            st.v[i] = b2 * st.v[i] + (1 - b2) * g * g;
            num += std::abs(st.m[i]);
        }
        const double bc1 = 1 - std::pow(b1, t), bc2 = 1 - std::pow(b2, t);
        std::array<int, D> y = x;
        bool any = false;
        for (int i = 0; i < D; ++i) {
            // NAdam: Nesterov の先読み運動量
            const double mn = b1 * st.m[i] / bc1 + (1 - b1) * gs[i] / bc1;
            const double s = lr * mn / (std::sqrt(st.v[i] / bc2) + eps);
            const int si = static_cast<int>(std::lround(-s));
            y[i] += si;
            any |= si != 0;
        }
        y = clampv(y);
        if (any && num > 0) {
            const double fy = eval(y);
            if (fy < fx) {
                double mv = 0;
                for (int i = 0; i < D; ++i) mv = std::max(mv, static_cast<double>(std::abs(y[i] - x[i])));
                fx = fy; x = y; st.last_step = mv;
                lr = std::min(lr * 1.5, 16.0);  // 成功: 歩幅を広げる
                fails = 0;
                if (++succ % 3 == 0) {
                    // Lookahead: 遅い重みと現在点の中点を試し、遅い重みを現在点へ引き寄せる
                    std::array<int, D> mid;
                    for (int i = 0; i < D; ++i) mid[i] = (phi[i] + x[i]) / 2;
                    mid = clampv(mid);
                    const double fmid = eval(mid);
                    if (fmid < fx) { fx = fmid; x = mid; }
                    phi = x;
                }
                continue;
            }
        }
        lr *= 0.5;  // 失敗: 歩幅と揺らぎを縮める
        c = std::max(1.0, c * 0.5);
        if (lr < 1.0) ++fails;
    }
    if (out_state) { *out_state = st; out_state->valid = true; }
    return x;
}

}  // namespace fvc
