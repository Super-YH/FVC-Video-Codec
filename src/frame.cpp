#include "fvc/frame.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <sstream>

#include "fvc/color.hpp"

namespace fvc {

Y4MReader::~Y4MReader() { if (fp_) std::fclose(fp_); }
Y4MWriter::~Y4MWriter() { if (fp_) std::fclose(fp_); }

static bool read_line(FILE* fp, std::string& s) {
    s.clear();
    int c;
    while ((c = std::fgetc(fp)) != EOF && c != '\n') s.push_back(static_cast<char>(c));
    return c != EOF || !s.empty();
}

bool Y4MReader::open(const std::string& path) {
    fp_ = std::fopen(path.c_str(), "rb");
    if (!fp_) return false;
    std::string hdr;
    if (!read_line(fp_, hdr) || hdr.rfind("YUV4MPEG2", 0) != 0) return false;
    std::istringstream is(hdr);
    std::string tok;
    info_.chroma = ChromaFormat::C420;
    info_.ct = ColorTransform::Identity;
    while (is >> tok) {
        switch (tok[0]) {
        case 'W': info_.width = std::stoi(tok.substr(1)); break;
        case 'H': info_.height = std::stoi(tok.substr(1)); break;
        case 'F': std::sscanf(tok.c_str() + 1, "%d:%d", &info_.fps_num, &info_.fps_den); break;
        case 'C':
            if (tok.rfind("C444", 0) == 0) info_.chroma = ChromaFormat::C444;
            else if (tok.rfind("C420", 0) == 0) info_.chroma = ChromaFormat::C420;
            else return false;
            {  // 高ビット深度 (C420p10 等) は未対応
                const size_t pp = tok.find('p', 4);
                if (pp != std::string::npos && pp + 1 < tok.size() && std::isdigit(static_cast<unsigned char>(tok[pp + 1]))) return false;
            }
            break;
        default: break;
        }
    }
    return info_.width > 0 && info_.height > 0;
}

bool Y4MReader::read(Frame& f) {
    std::string line;
    if (!read_line(fp_, line) || line.rfind("FRAME", 0) != 0) return false;
    f.p.clear();
    for (int i = 0; i < 3; ++i) {
        const int w = i ? info_.chroma_w() : info_.width, h = i ? info_.chroma_h() : info_.height;
        Plane pl(w, h);
        std::vector<uint8_t> buf(static_cast<size_t>(w) * h);
        if (std::fread(buf.data(), 1, buf.size(), fp_) != buf.size()) return false;
        for (size_t k = 0; k < buf.size(); ++k) pl.v[k] = buf[k];
        f.p.push_back(std::move(pl));
    }
    return true;
}

bool Y4MWriter::open(const std::string& path, const VideoInfo& info) {
    info_ = info;
    fp_ = std::fopen(path.c_str(), "wb");
    if (!fp_) return false;
    std::fprintf(fp_, "YUV4MPEG2 W%d H%d F%d:%d Ip A1:1 %s\n", info.width, info.height, info.fps_num,
                 info.fps_den, info.chroma == ChromaFormat::C444 ? "C444" : "C420jpeg");
    return true;
}

bool Y4MWriter::write(const Frame& f) {
    std::fputs("FRAME\n", fp_);
    struct Flush { FILE* f; ~Flush() { std::fflush(f); } } fl{fp_};  // ストリーミング: フレームごとに出力
    for (const Plane& pl : f.p) {
        std::vector<uint8_t> buf(pl.v.size());
        for (size_t k = 0; k < buf.size(); ++k) buf[k] = static_cast<uint8_t>(std::min(255, std::max(0, pl.v[k])));
        if (std::fwrite(buf.data(), 1, buf.size(), fp_) != buf.size()) return false;
    }
    return true;
}

static bool ppm_token(FILE* fp, int& v) {
    int c;
    do {
        c = std::fgetc(fp);
        if (c == '#') while (c != '\n' && c != EOF) c = std::fgetc(fp);
    } while (c != EOF && std::isspace(c));
    if (c == EOF) return false;
    v = 0;
    while (c != EOF && std::isdigit(c)) { v = v * 10 + (c - '0'); c = std::fgetc(fp); }
    return true;
}

bool read_ppm_ycocg(const std::string& path, Frame& f, VideoInfo& info) {
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) return false;
    char m[2];
    int w, h, mx;
    bool ok = std::fread(m, 1, 2, fp) == 2 && m[0] == 'P' && m[1] == '6' && ppm_token(fp, w) && ppm_token(fp, h) &&
              ppm_token(fp, mx) && mx == 255;
    std::vector<uint8_t> buf;
    if (ok) {
        buf.resize(static_cast<size_t>(w) * h * 3);
        ok = std::fread(buf.data(), 1, buf.size(), fp) == buf.size();
    }
    std::fclose(fp);
    if (!ok) return false;
    info = VideoInfo{};
    info.width = w; info.height = h; info.bit_depth = 8;
    info.chroma = ChromaFormat::C444; info.ct = ColorTransform::YCoCgR;
    f.p.assign(3, Plane(w, h));
    for (size_t k = 0; k < static_cast<size_t>(w) * h; ++k)
        rgb_to_ycocg_r(buf[3 * k], buf[3 * k + 1], buf[3 * k + 2], f.p[0].v[k], f.p[1].v[k], f.p[2].v[k]);
    return true;
}

bool write_ppm_ycocg(const std::string& path, const Frame& f, const VideoInfo& info) {
    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) return false;
    std::fprintf(fp, "P6\n%d %d\n255\n", info.width, info.height);
    std::vector<uint8_t> buf(static_cast<size_t>(info.width) * info.height * 3);
    for (size_t k = 0; k < static_cast<size_t>(info.width) * info.height; ++k) {
        int32_t r, g, b;
        ycocg_r_to_rgb(f.p[0].v[k], f.p[1].v[k], f.p[2].v[k], r, g, b);
        buf[3 * k] = static_cast<uint8_t>(std::min(255, std::max(0, r)));
        buf[3 * k + 1] = static_cast<uint8_t>(std::min(255, std::max(0, g)));
        buf[3 * k + 2] = static_cast<uint8_t>(std::min(255, std::max(0, b)));
    }
    const bool ok = std::fwrite(buf.data(), 1, buf.size(), fp) == buf.size();
    std::fclose(fp);
    return ok;
}

void plane_range(const VideoInfo& info, int plane, int32_t& lo, int32_t& hi) {
    const int32_t mx = (1 << info.bit_depth) - 1;
    if (info.ct == ColorTransform::YCoCgR && plane > 0) { lo = -mx; hi = mx; }
    else { lo = 0; hi = mx; }
}

double plane_psnr(const Plane& a, const Plane& b, int bit_depth) {
    double se = 0;
    for (size_t k = 0; k < a.v.size(); ++k) { const double d = a.v[k] - b.v[k]; se += d * d; }
    if (se == 0) return 999.0;
    const double peak = (1 << bit_depth) - 1;
    return 10.0 * std::log10(peak * peak * a.v.size() / se);
}

double plane_ssim(const Plane& a, const Plane& b, int bit_depth) {
    const double L = (1 << bit_depth) - 1, c1 = (0.01 * L) * (0.01 * L), c2 = (0.03 * L) * (0.03 * L);
    double sum = 0;
    int n = 0;
    for (int y = 0; y + 8 <= a.h; y += 4)
        for (int x = 0; x + 8 <= a.w; x += 4) {
            double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
            for (int j = 0; j < 8; ++j)
                for (int i = 0; i < 8; ++i) {
                    const double va = a.at(x + i, y + j), vb = b.at(x + i, y + j);
                    sa += va; sb += vb; saa += va * va; sbb += vb * vb; sab += va * vb;
                }
            const double ma = sa / 64, mb = sb / 64;
            const double va = saa / 64 - ma * ma, vb = sbb / 64 - mb * mb, cov = sab / 64 - ma * mb;
            sum += ((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2));
            ++n;
        }
    return n ? sum / n : 1.0;
}

int y4m_frame_count(const std::string& path, const VideoInfo& info) {
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) return -1;
    std::string hdr;
    int c;
    while ((c = std::fgetc(fp)) != EOF && c != '\n') hdr.push_back(static_cast<char>(c));
    const long start = std::ftell(fp);
    std::fseek(fp, 0, SEEK_END);
    const long end = std::ftell(fp);
    std::fclose(fp);
    const long fsz = 6L + info.width * static_cast<long>(info.height) + 2L * info.chroma_w() * info.chroma_h();
    return fsz > 0 ? static_cast<int>((end - start) / fsz) : -1;
}

}  // namespace fvc
