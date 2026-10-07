// fvc コマンドラインツール
//   fvc enc [options] input.(y4m|ppm) output.fvc
//   fvc dec input.fvc output.(y4m|ppm)
// options: -q QP  --preset faster|fast|medium|slow|placebo  --lossless  --l2  --pqmf N  --frames N
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>

#include "fvc/codec.hpp"

using namespace fvc;

static bool ends_with(const std::string& s, const char* suf) {
    const size_t n = std::strlen(suf);
    return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
}

static int usage() {
    std::fprintf(stderr,
                 "usage:\n  fvc enc [-q QP] [--preset faster|fast|medium|slow|placebo] [--lossless] [--l2] [--pqmf N]"
                 " [--frames N] in.(y4m|ppm) out.fvc\n  fvc dec in.fvc out.(y4m|ppm)\n");
    return 2;
}

static int cmd_enc(int argc, char** argv) {
    EncoderConfig cfg;
    int max_frames = 1 << 30;
    std::string in, out;
    for (int i = 0; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-q" && i + 1 < argc) cfg.qp = std::atoi(argv[++i]);
        else if (a == "--lossless") { cfg.lossy_layer = false; }
        else if (a == "--l2") cfg.l2_lossless = true;
        else if (a == "--psy") cfg.psy = true;
        else if (a == "--ibc") cfg.ibc = 1;
        else if (a == "--no-ibc") cfg.ibc = 0;
        else if (a.rfind("--e8=", 0) == 0) cfg.e8 = std::atoi(a.c_str() + 5);
        else if (a.rfind("--tns=", 0) == 0) cfg.tns = std::atoi(a.c_str() + 6);
        else if (a.rfind("--cfl=", 0) == 0) cfg.cfl = std::atoi(a.c_str() + 6);
        else if (a == "--pqmf" && i + 1 < argc) cfg.pqmf_log2 = std::atoi(argv[++i]);
        else if (a == "--frames" && i + 1 < argc) max_frames = std::atoi(argv[++i]);
        else if (a == "--preset" && i + 1 < argc) {
            const std::string p = argv[++i];
            if (p == "faster") cfg.preset = Preset::Faster;
            else if (p == "fast") cfg.preset = Preset::Fast;
            else if (p == "medium") cfg.preset = Preset::Medium;
            else if (p == "slow") cfg.preset = Preset::Slow;
            else if (p == "placebo") cfg.preset = Preset::Placebo;
            else return usage();
        } else if (in.empty()) in = a;
        else if (out.empty()) out = a;
        else return usage();
    }
    if (in.empty() || out.empty()) return usage();

    VideoInfo info;
    Y4MReader y4m;
    Frame single;
    const bool is_ppm = ends_with(in, ".ppm");
    if (is_ppm) {
        if (!read_ppm_ycocg(in, single, info)) { std::fprintf(stderr, "cannot read %s\n", in.c_str()); return 1; }
    } else {
        if (!y4m.open(in)) { std::fprintf(stderr, "cannot read %s\n", in.c_str()); return 1; }
        info = y4m.info();
    }
    Encoder enc(info, cfg);
    std::vector<uint8_t> stream = enc.sequence_header();
    Frame f, rec;
    int n = 0;
    double psnr_sum[3] = {0, 0, 0};
    const auto t0 = std::chrono::steady_clock::now();
    while (n < max_frames) {
        if (is_ppm) { if (n) break; f = single; }
        else if (!y4m.read(f)) break;
        const auto u = enc.encode(f, &rec);
        stream.insert(stream.end(), u.begin(), u.end());
        for (int p = 0; p < 3; ++p) psnr_sum[p] += plane_psnr(f.p[p], rec.p[p], info.bit_depth);
        ++n;
    }
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const auto e = enc.end_of_stream();
    stream.insert(stream.end(), e.begin(), e.end());
    std::ofstream(out, std::ios::binary).write(reinterpret_cast<const char*>(stream.data()), static_cast<std::streamsize>(stream.size()));
    const double pixels = static_cast<double>(info.width) * info.height * n;
    std::printf("frames=%d bytes=%zu bpp=%.4f PSNR(Y/C1/C2)=%.3f/%.3f/%.3f fps=%.3f\n", n, stream.size(),
                stream.size() * 8.0 / std::max(1.0, pixels), psnr_sum[0] / std::max(1, n), psnr_sum[1] / std::max(1, n),
                psnr_sum[2] / std::max(1, n), n / std::max(sec, 1e-9));
    return 0;
}

static int cmd_dec(int argc, char** argv) {
    if (argc != 2) return usage();
    std::ifstream is(argv[0], std::ios::binary);
    if (!is) { std::fprintf(stderr, "cannot read %s\n", argv[0]); return 1; }
    std::vector<uint8_t> s((std::istreambuf_iterator<char>(is)), std::istreambuf_iterator<char>());
    Decoder dec(s);
    if (!dec.ok()) { std::fprintf(stderr, "bad stream\n"); return 1; }
    const std::string out = argv[1];
    Frame f;
    int n = 0;
    if (ends_with(out, ".ppm")) {
        if (!dec.next(f)) { std::fprintf(stderr, "no frame\n"); return 1; }
        if (dec.info().ct != ColorTransform::YCoCgR) { std::fprintf(stderr, "ppm output needs YCoCg stream\n"); return 1; }
        write_ppm_ycocg(out, f, dec.info());
        n = 1;
    } else {
        Y4MWriter w;
        if (!w.open(out, dec.info())) return 1;
        while (dec.next(f)) { w.write(f); ++n; }
    }
    std::printf("decoded %d frame(s)\n", n);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) return usage();
    try {
        if (!std::strcmp(argv[1], "enc")) return cmd_enc(argc - 2, argv + 2);
        if (!std::strcmp(argv[1], "dec")) return cmd_dec(argc - 2, argv + 2);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return usage();
}
