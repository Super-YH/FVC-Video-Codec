// fvc コマンドラインツール
//   fvc enc [options] input.(y4m|ppm) output.fvc
//   fvc dec input.fvc output.(y4m|ppm)
// options: -q QP  --preset faster|fast|medium|slow|placebo  --lossless  --l2  --pqmf N  --frames N
#include <chrono>
#include <cmath>
#include <ctime>
#include <map>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
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
                 " [--frames N]\n      [--keyint N] [--bframes N] [--refs N] [--no-copy] [--psy] [--ibc|--no-ibc]\n      [--e8=0|1] [--tns=0|1] [--cfl=0|1] [--dict=0|1] [--shapes=0|1] [--fir=0|1] [--lf=0|1]\n      [--tiles C R] [--threads N] [--cqp N] [--aqp=0|1] [--rect=0|1] [--tmvp=0|1] [--alf=0|1] [--mts=0|1] [--cdef=0|1] [--tune psy|psnr] [--band-coder blocks|samples] [--psy-rd X]\n      [--ssim TARGET] [-v] [--quiet] in.(y4m|ppm) out.fvc\n  fvc dec in.fvc out.(y4m|ppm)\n");
    return 2;
}

static int cmd_enc(int argc, char** argv) {
    EncoderConfig cfg;
    int max_frames = 1 << 30;
    bool verbose = false, quiet = false;
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
        else if (a == "--keyint" && i + 1 < argc) cfg.keyint = std::atoi(argv[++i]);
        else if (a == "--bframes" && i + 1 < argc) cfg.bframes = std::atoi(argv[++i]);
        else if (a == "--refs" && i + 1 < argc) cfg.refs = std::atoi(argv[++i]);
        else if (a == "--no-copy") cfg.copy_frames = false;
        else if (a == "-v") verbose = true;
        else if (a == "--quiet") quiet = true;
        else if (a == "--ssim" && i + 1 < argc) cfg.target_ssim = std::atof(argv[++i]);
        else if (a == "--tiles" && i + 2 < argc) { cfg.tile_cols = std::atoi(argv[++i]); cfg.tile_rows = std::atoi(argv[++i]); }
        else if (a == "--threads" && i + 1 < argc) cfg.threads = std::atoi(argv[++i]);
        else if (a == "--cqp" && i + 1 < argc) cfg.chroma_qp_offset = std::atoi(argv[++i]);
        else if (a.rfind("--aqp=", 0) == 0) cfg.aqp = std::atoi(a.c_str() + 6);
        else if (a.rfind("--rect=", 0) == 0) cfg.rect = std::atoi(a.c_str() + 7);
        else if (a.rfind("--tmvp=", 0) == 0) cfg.tmvp = std::atoi(a.c_str() + 7);
        else if (a.rfind("--alf=", 0) == 0) cfg.alf = std::atoi(a.c_str() + 6);
        else if (a.rfind("--mts=", 0) == 0) cfg.mts = std::atoi(a.c_str() + 6);
        else if (a.rfind("--cdef=", 0) == 0) cfg.cdef = std::atoi(a.c_str() + 7);
        else if (a == "--band-coder" && i + 1 < argc) cfg.band_samples = std::string(argv[++i]) == "samples";
        else if (a == "--tune" && i + 1 < argc) cfg.tune_psnr = std::string(argv[++i]) == "psnr";
        else if (a == "--psy-rd" && i + 1 < argc) cfg.psy_strength = std::atof(argv[++i]);
        else if (a.rfind("--dict=", 0) == 0) cfg.dict = std::atoi(a.c_str() + 7);
        else if (a.rfind("--shapes=", 0) == 0) cfg.shapes = std::atoi(a.c_str() + 9);
        else if (a.rfind("--fir=", 0) == 0) cfg.fir = std::atoi(a.c_str() + 6);
        else if (a.rfind("--lf=", 0) == 0) cfg.loop_filter = std::atoi(a.c_str() + 5);
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
    cfg.keep_recon = false;
    Encoder enc(info, cfg);
    const char* kPreset[] = {"faster", "fast", "medium", "slow", "placebo"};
    int total = is_ppm ? 1 : y4m_frame_count(in, info);
    if (total > 0) total = std::min(total, max_frames);
    const double fps = static_cast<double>(info.fps_num) / std::max(1, info.fps_den);
    if (!quiet) {
        // 開始時の設定表示 (LAME 風)
        std::fprintf(stderr, "FVC encoder (C++17 reference)\n");
        std::fprintf(stderr, "Encoding %s\n      to %s\n", in.c_str(), out.c_str());
        std::fprintf(stderr, "Input : %dx%d %s %d-bit %s, %.3f fps%s\n", info.width, info.height,
                     info.chroma == ChromaFormat::C420 ? "4:2:0" : "4:4:4", info.bit_depth,
                     info.ct == ColorTransform::YCoCgR ? "YCoCg-R" : "YCbCr", fps,
                     total > 0 ? (", " + std::to_string(total) + " frames").c_str() : "");
        if (!cfg.lossy_layer)
            std::fprintf(stderr, "Mode  : lossless (MED + CM)\n");
        else if (cfg.target_ssim > 0)
            std::fprintf(stderr, "Mode  : target SSIM %.4f (%.2f dB), per-frame QP search\n", cfg.target_ssim,
                         -10.0 * std::log10(std::max(1e-9, 1.0 - cfg.target_ssim)));
        else
            std::fprintf(stderr, "Mode  : constant QP %d (I-5 / P-2 / B+1+depth)%s\n", cfg.qp, cfg.l2_lossless ? " + L2 lossless" : "");
        std::fprintf(stderr, "Preset: %s  bframes=%s  keyint=%s  pqmf=%d  psy=%s  threads=%s\n",
                     kPreset[static_cast<int>(cfg.preset)],
                     cfg.bframes >= 0 ? std::to_string(cfg.bframes).c_str() : "auto",
                     cfg.keyint > 0 ? std::to_string(cfg.keyint).c_str() : "inf", cfg.pqmf_log2, cfg.psy ? "on" : "off",
                     cfg.threads > 0 ? std::to_string(cfg.threads).c_str() : "auto");
        std::fprintf(stderr, "\n    Frame          |  CPU time/estim | REAL time/estim | play/CPU |    ETA  |   kbps | PSNR-Y |  SSIM\n");
    }
    // ストリーミング出力: ユニットが出来た順にファイルへ書き、都度フラッシュする
    std::ofstream os(out, std::ios::binary);
    if (!os) { std::fprintf(stderr, "cannot write %s\n", out.c_str()); return 1; }
    size_t stream_size = 0;
    auto emit = [&](const std::vector<uint8_t>& u) {
        if (u.empty()) return;
        os.write(reinterpret_cast<const char*>(u.data()), static_cast<std::streamsize>(u.size()));
        os.flush();
        stream_size += u.size();
    };
    emit(enc.sequence_header());
    Frame f;
    int n = 0;
    const auto t0 = std::chrono::steady_clock::now();
    const std::clock_t c0 = std::clock();
    auto hms = [](double sec) {
        char b[32];
        const int t = static_cast<int>(sec + 0.5);
        if (t >= 3600) std::snprintf(b, sizeof b, "%d:%02d:%02d", t / 3600, t / 60 % 60, t % 60);
        else std::snprintf(b, sizeof b, "%d:%02d", t / 60, t % 60);
        return std::string(b);
    };
    size_t shown = 0;
    auto progress = [&](bool final) {
        const auto& st = enc.stats();
        if (quiet || (!final && st.size() == shown)) return;
        shown = st.size();
        const double real = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const double cpu = static_cast<double>(std::clock() - c0) / CLOCKS_PER_SEC;
        const int done = static_cast<int>(st.size());
        const double frac = total > 0 ? std::min(1.0, static_cast<double>(done) / total) : 0.0;
        double ps = 0, ss = 0;
        size_t bytes = 0;
        for (const auto& x : st) { ps += x.psnr[0]; ss += x.ssim; bytes += x.bytes; }
        const double kbps = done ? bytes * 8.0 / 1000.0 / (done / fps) : 0.0;
        char pos[48];
        if (total > 0) std::snprintf(pos, sizeof pos, "%6d/%-6d (%3d%%)", done, total, static_cast<int>(frac * 100));
        else std::snprintf(pos, sizeof pos, "%6d          ", done);
        const double est_c = frac > 0 ? cpu / frac : 0, est_r = frac > 0 ? real / frac : 0;
        std::fprintf(stderr, "\r%s| %7s/%7s | %7s/%7s | %7.4fx | %7s | %6.1f | %6.2f | %.4f%s", pos, hms(cpu).c_str(),
                     hms(est_c).c_str(), hms(real).c_str(), hms(est_r).c_str(), cpu > 0 ? (done / fps) / cpu : 0.0,
                     hms(std::max(0.0, est_r - real)).c_str(), kbps, done ? ps / done : 0.0, done ? ss / done : 0.0,
                     final ? "\n" : "");
        std::fflush(stderr);
    };
    while (n < max_frames) {
        if (is_ppm) { if (n) break; f = single; }
        else if (!y4m.read(f)) break;
        emit(enc.encode(f));
        ++n;
        progress(false);
    }
    emit(enc.flush());
    progress(true);
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    os.close();

    // ---- 集計
    const auto& stats = enc.stats();
    double ps[3] = {0, 0, 0}, ss = 0;
    int cnt[4] = {0, 0, 0, 0}, trials = 0;
    size_t tbytes[4] = {0, 0, 0, 0};
    std::map<int, int> qh[4];
    BlockUsage u;
    for (const auto& st : stats) {
        if (verbose)
            std::printf("  poc=%4d %c qp=%2d bytes=%7zu PSNR-Y=%6.2f SSIM=%.4f%s\n", st.poc, "IPBC"[static_cast<int>(st.type)],
                        st.qp, st.bytes, st.psnr[0], st.ssim, st.trials > 1 ? (" trials=" + std::to_string(st.trials)).c_str() : "");
        for (int p = 0; p < 3; ++p) ps[p] += st.psnr[p];
        ss += st.ssim;
        const int t = static_cast<int>(st.type);
        ++cnt[t];
        tbytes[t] += st.bytes;
        ++qh[t][st.qp];
        trials += st.trials;
        u.add(st.usage);
    }
    const double pixels = static_cast<double>(info.width) * info.height * n;
    if (!quiet && n > 0) {
        const char* tn[4] = {"I", "P", "B", "COPY"};
        std::printf("\nFrame types          count     share    avg bytes   bytes share   QP histogram\n");
        for (int t = 0; t < 4; ++t) {
            if (!cnt[t]) continue;
            std::string hist;
            for (const auto& kv : qh[t]) hist += std::to_string(kv.first) + ":" + std::to_string(kv.second) + " ";
            std::printf("  %-6s          %8d   %6.1f%%   %10.0f   %10.1f%%   %s\n", tn[t], cnt[t], 100.0 * cnt[t] / n,
                        static_cast<double>(tbytes[t]) / cnt[t], 100.0 * tbytes[t] / std::max<size_t>(1, stream_size), hist.c_str());
        }
        const double area = static_cast<double>(u.intra + u.inter + u.ibc + u.dict);
        if (area > 0) {
            auto pc = [&](uint64_t v) { return 100.0 * v / area; };
            auto bar = [&](uint64_t v) { return std::string(static_cast<size_t>(pc(v) / 2.5 + 0.5), '*'); };
            std::printf("\nBlock modes (luma area)\n");
            std::printf("  intra   %6.2f%%  %s\n", pc(u.intra), bar(u.intra).c_str());
            std::printf("  inter   %6.2f%%  %s\n", pc(u.inter), bar(u.inter).c_str());
            std::printf("    merge %6.2f%%  (skip %.2f%%)  bi-pred %.2f%%  rect %.2f%%\n", pc(u.merge), pc(u.skip), pc(u.bi), pc(u.rect));
            std::printf("  ibc     %6.2f%%  %s\n", pc(u.ibc), bar(u.ibc).c_str());
            std::printf("  dict    %6.2f%%  %s\n", pc(u.dict), bar(u.dict).c_str());
            std::printf("  tools   TNS %.2f%%  E8-VQ %.2f%%  CfL(chroma area) %llu px\n", pc(u.tns), pc(u.e8),
                        static_cast<unsigned long long>(u.cfl));
            std::printf("\nBlock sizes (luma area)\n");
            for (int i = 0; i < 7; ++i) {
                if (!u.size[i]) continue;
                const int sz = 4 << i;
                std::printf("  %3dx%-3d %6.2f%%  %s\n", sz, sz, pc(u.size[i]), bar(u.size[i]).c_str());
            }
        }
        std::printf("\nAverage PSNR  Y %.3f  Cb/Co %.3f  Cr/Cg %.3f dB   SSIM %.5f (%.2f dB)\n", ps[0] / n, ps[1] / n, ps[2] / n,
                    ss / n, -10.0 * std::log10(std::max(1e-9, 1.0 - ss / n)));
        std::printf("Size %zu bytes  %.1f kbps  %.4f bpp   speed %.3f fps (%.3fx realtime)%s\n\n", stream_size,
                    stream_size * 8.0 / 1000.0 / std::max(1e-9, n / fps), stream_size * 8.0 / std::max(1.0, pixels),
                    n / std::max(sec, 1e-9), (n / fps) / std::max(sec, 1e-9),
                    cfg.target_ssim > 0 ? ("   QP search encodes " + std::to_string(trials)).c_str() : "");
    }
    // 機械可読な 1 行要約
    std::printf("frames=%d (I%d P%d B%d C%d) bytes=%zu bpp=%.4f kbps=%.1f PSNR(Y/C1/C2)=%.3f/%.3f/%.3f SSIM=%.5f fps=%.3f\n", n,
                cnt[0], cnt[1], cnt[2], cnt[3], stream_size, stream_size * 8.0 / std::max(1.0, pixels),
                stream_size * 8.0 / 1000.0 / std::max(1e-9, n / fps), ps[0] / std::max(1, n), ps[1] / std::max(1, n),
                ps[2] / std::max(1, n), ss / std::max(1, n), n / std::max(sec, 1e-9));
    return 0;
}

static int cmd_dec(int argc, char** argv) {
    if (argc != 2) return usage();
    // ストリーミング復号: 入力を逐次読み、フレームが揃い次第出力する ("-" で標準入力)
    std::ifstream ifs;
    std::istream* is = &std::cin;
    if (std::string(argv[0]) != "-") {
        ifs.open(argv[0], std::ios::binary);
        if (!ifs) { std::fprintf(stderr, "cannot read %s\n", argv[0]); return 1; }
        is = &ifs;
    }
    Decoder dec(*is);
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
