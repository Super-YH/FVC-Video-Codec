// FVC リファレンス部品の単体テスト (外部依存なし)
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "fvc/codec.hpp"
#include "fvc/color.hpp"
#include "fvc/entropy.hpp"
#include "fvc/pqmf.hpp"
#include "fvc/quant.hpp"
#include "fvc/transform.hpp"

using namespace fvc;

static int g_fail = 0;
#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #cond); ++g_fail; } \
    } while (0)

static void test_color() {
    for (int r = 0; r < 256; r += 5)
        for (int g = 0; g < 256; g += 7)
            for (int b = 0; b < 256; b += 3) {
                int32_t y, co, cg, r2, g2, b2;
                rgb_to_ycocg_r(r, g, b, y, co, cg);
                ycocg_r_to_rgb(y, co, cg, r2, g2, b2);
                CHECK(r == r2 && g == g2 && b == b2);
            }
    std::vector<int32_t> L(64), C(64), P(64);
    for (int i = 0; i < 64; ++i) { L[i] = i * 3 + 10; C[i] = (L[i] * 3) / 4 - 7; }
    auto p = fit_chroma_pred(L.data(), C.data(), 64);
    apply_chroma_pred(p, L.data(), 64, P.data());
    int maxe = 0;
    for (int i = 0; i < 64; ++i) maxe = std::max(maxe, std::abs(C[i] - P[i]));
    CHECK(maxe <= 2);
    std::printf("color: ok (CfL alpha_q=%d max|res|=%d)\n", p.alpha_q, maxe);
}

static void test_pqmf() {
    const int W = 128, H = 64;
    std::vector<double> img(W * H), rec;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            img[y * W + x] = 128 + 60 * std::sin(x * 0.11) * std::cos(y * 0.07) + ((x * 31 + y * 17) % 23);
    for (int M : {2, 4, 8, 16}) {
        Pqmf2D fb(M, M);
        std::vector<std::vector<double>> bands;
        fb.analyze(img, W, H, bands);
        fb.synthesize(bands, W, H, rec);
        const double q = psnr(img, rec, 255.0);
        std::printf("pqmf: %2dx%-2d = %3d bands, PSNR %.1f dB\n", M, M, M * M, q);
        CHECK(q > 50.0);
        // ロスレス経路: 残差 r = x - round(x^) を別途符号化すれば完全再構成
        int maxr = 0;
        for (int i = 0; i < W * H; ++i) maxr = std::max(maxr, static_cast<int>(std::abs(std::lround(img[i] - rec[i]))));
        CHECK(maxr <= 2);
    }
}

static void test_lattice() {
    SplitMix64 rng(1);
    double mse[3] = {0, 0, 0};
    const int T = 20000;
    for (int t = 0; t < T; ++t) {
        double x[8], y[8], z[8];
        for (double& v : x) v = (rng.uniform() - 0.5) * 20.0;
        const Lattice Ls[3] = {Lattice::Zn, Lattice::Dn, Lattice::E8};
        for (int l = 0; l < 3; ++l) {
            LatticeIndex idx;
            lattice_quantize(Ls[l], x, 8, 1.0, y, &idx);
            index_to_lattice(Ls[l], idx, 8, z);
            for (int i = 0; i < 8; ++i) CHECK(std::abs(y[i] - z[i]) < 1e-9);
            for (int i = 0; i < 8; ++i) mse[l] += (x[i] - y[i]) * (x[i] - y[i]);
        }
    }
    // 正規化二次モーメント G = E|e|^2 / (n V^(2/n)); V(Z8)=1, V(D8)=2, V(E8)=1
    const double vol[3] = {1.0, 2.0, 1.0};
    const char* nm[3] = {"Z8", "D8", "E8"};
    double G[3];
    for (int l = 0; l < 3; ++l) {
        G[l] = mse[l] / T / 8.0 / std::pow(vol[l], 2.0 / 8.0);
        std::printf("lattice %s: G=%.4f\n", nm[l], G[l]);
    }
    CHECK(std::abs(G[0] - 1.0 / 12) < 0.003);
    CHECK(std::abs(G[2] - 0.0717) < 0.003);
    CHECK(G[2] < G[1] && G[1] < G[0]);

    NonlinearSQ sq;
    sq.gamma = 0.75; sq.step = 2.0;
    for (double x = -100; x <= 100; x += 0.37) {
        const double r = sq.dequantize(sq.quantize(x));
        CHECK(std::abs(r - x) <= std::pow(std::abs(x) + 1.0, 1.0 - sq.gamma) * sq.step * 2.0 + 1e-9);
    }
    for (auto t : {PreNonlin::Identity, PreNonlin::Power, PreNonlin::Asinh, PreNonlin::SignedLog})
        for (double x = -50; x < 50; x += 1.3) CHECK(std::abs(pre_inverse(t, pre_forward(t, x, 0.7), 0.7) - x) < 1e-6);

    RandomProjection rp(8, 32, 1234);
    std::vector<double> v(32), y(8), r(32), y2(8);
    for (double& e : v) e = rng.uniform() - 0.5;
    rp.project(v.data(), y.data());
    rp.reconstruct(y.data(), r.data());
    rp.project(r.data(), y2.data());
    for (int i = 0; i < 8; ++i) CHECK(std::abs(y[i] - y2[i]) < 1e-9);  // P P^+ = I
    std::printf("quant: ok\n");
}

static void test_transform() {
    for (int w : {4, 8, 16, 32, 64})
        for (int h : {4, 16, 128}) {
            std::vector<double> a(w * h), c(w * h), b(w * h);
            for (int i = 0; i < w * h; ++i) a[i] = std::sin(i * 0.3) * 50 + i % 7;
            for (TxType th : {TxType::DCT2, TxType::DST7, TxType::IDTX}) {
                forward_2d(th, TxType::DCT2, a.data(), w, h, c.data());
                inverse_2d(th, TxType::DCT2, c.data(), w, h, b.data());
                double e = 0;
                for (int i = 0; i < w * h; ++i) e = std::max(e, std::abs(a[i] - b[i]));
                CHECK(e < 1e-8);
            }
        }
    // TNS 往復
    std::vector<double> c(64), e(64), r(64);
    for (int k = 0; k < 64; ++k) c[k] = 100 * std::pow(0.9, k) * std::cos(k * 0.4);
    auto f = tns_design(c.data(), 64, 8, 4);
    tns_analysis(f, c.data(), 64, e.data());
    tns_synthesis(f, e.data(), 64, r.data());
    double ec = 0, ee = 0, md = 0;
    for (int k = 0; k < 64; ++k) { ec += c[k] * c[k]; ee += e[k] * e[k]; md = std::max(md, std::abs(r[k] - c[k])); }
    CHECK(md < 1e-8);
    CHECK(ee < ec);
    std::printf("transform: ok (TNS order=%d, pred gain %.1f dB)\n", f.order, 10 * std::log10(ec / ee));
}

static void test_entropy() {
    SplitMix64 rng(7);
    std::vector<int32_t> vals(50000);
    // ラプラス分布的な整数列 (前値と相関)
    int32_t prev = 0;
    for (auto& v : vals) {
        const double u = rng.uniform();
        int32_t m = static_cast<int32_t>(-std::log(1 - u) * 3.0);
        v = (rng.next() & 1) ? m : -m;
        if (rng.uniform() < 0.3) v = prev;
        prev = v;
    }
    vals.push_back(2000000000);
    vals.push_back(-2000000000);
    CMModel me, md;
    EntropyWriter w;
    prev = 0;
    for (int32_t v : vals) {
        w.sint(me, static_cast<uint32_t>(std::min(std::abs(prev), 15)), prev < 0, v);
        prev = v;
    }
    auto bytes = w.finish();
    EntropyReader r(bytes.data(), bytes.size());
    prev = 0;
    bool ok = true;
    for (int32_t v : vals) {
        const int32_t d = r.sint(md, static_cast<uint32_t>(std::min(std::abs(prev), 15)), prev < 0);
        if (d != v) { ok = false; break; }
        prev = d;
    }
    CHECK(ok);
    std::printf("entropy: %zu symbols -> %zu bytes (%.3f bits/sym), roundtrip %s\n", vals.size(), bytes.size(),
                bytes.size() * 8.0 / vals.size(), ok ? "ok" : "NG");
}

static void test_codec() {
    VideoInfo info;
    info.width = 83; info.height = 61; info.chroma = ChromaFormat::C420; info.ct = ColorTransform::Identity;
    Frame f;
    for (int p = 0; p < 3; ++p) {
        const int w = p ? info.chroma_w() : info.width, h = p ? info.chroma_h() : info.height;
        Plane pl(w, h);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                pl.at(x, y) = std::clamp(static_cast<int>(128 + 70 * std::sin(x * 0.13 + p) * std::cos(y * 0.09) + ((x * 13 + y * 7) % 11)), 0, 255);
        f.p.push_back(pl);
    }
    struct Case { bool lossy, l2; int pqmf, qp; Preset pr; };
    const Case cases[] = {{false, false, 0, 0, Preset::Medium}, {true, true, 0, 30, Preset::Medium}, {true, false, 0, 30, Preset::Faster},
                          {true, false, 0, 40, Preset::Placebo}, {true, true, 2, 30, Preset::Medium}, {true, false, 1, 25, Preset::Medium}};
    for (const Case& c : cases) {
        EncoderConfig cfg;
        cfg.lossy_layer = c.lossy; cfg.l2_lossless = c.l2; cfg.pqmf_log2 = c.pqmf; cfg.qp = c.qp; cfg.preset = c.pr;
        Encoder enc(info, cfg);
        auto s = enc.sequence_header();
        Frame rec, dec;
        for (int k = 0; k < 2; ++k) { auto u = enc.encode(f, &rec); s.insert(s.end(), u.begin(), u.end()); }
        auto e = enc.end_of_stream(); s.insert(s.end(), e.begin(), e.end());
        Decoder d(s);
        CHECK(d.ok());
        int n = 0;
        while (d.next(dec)) {
            ++n;
            for (int p = 0; p < 3; ++p) {
                CHECK(dec.p[p].v == rec.p[p].v);                       // 符号器と復号器の再構成一致
                if (!c.lossy || c.l2) CHECK(dec.p[p].v == f.p[p].v);    // 可逆
            }
        }
        CHECK(n == 2);
        std::printf("codec: lossy=%d l2=%d pqmf=%d qp=%d -> %zu bytes, PSNR-Y %.2f\n", c.lossy, c.l2, c.pqmf, c.qp, s.size(),
                    plane_psnr(f.p[0], rec.p[0], 8));
    }
}

int main() {
    test_color();
    test_pqmf();
    test_lattice();
    test_transform();
    test_entropy();
    test_codec();
    if (g_fail) { std::printf("%d failure(s)\n", g_fail); return 1; }
    std::printf("all tests passed\n");
    return 0;
}
