// FVC: フレーム/プレーン表現と入出力 (Y4M 420/444, PPM P6)
#pragma once
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace fvc {

struct Plane {
    int w = 0, h = 0;
    std::vector<int32_t> v;
    Plane() = default;
    Plane(int w_, int h_, int32_t fill = 0) : w(w_), h(h_), v(static_cast<size_t>(w_) * h_, fill) {}
    int32_t& at(int x, int y) { return v[static_cast<size_t>(y) * w + x]; }
    int32_t at(int x, int y) const { return v[static_cast<size_t>(y) * w + x]; }
};

enum class ChromaFormat : uint8_t { C444 = 0, C420 = 1 };
enum class ColorTransform : uint8_t { YCoCgR = 0, Identity = 1 };

struct Frame {
    std::vector<Plane> p;  // 3 プレーン
};

struct VideoInfo {
    int width = 0, height = 0, bit_depth = 8;
    ChromaFormat chroma = ChromaFormat::C420;
    ColorTransform ct = ColorTransform::Identity;
    int fps_num = 30, fps_den = 1;
    int chroma_w() const { return chroma == ChromaFormat::C420 ? (width + 1) / 2 : width; }
    int chroma_h() const { return chroma == ChromaFormat::C420 ? (height + 1) / 2 : height; }
};

// Y4M (8bit, C420*/C444)。フレームは YUV のまま (ct=Identity)。
class Y4MReader {
public:
    bool open(const std::string& path);
    bool read(Frame& f);
    const VideoInfo& info() const { return info_; }
    ~Y4MReader();
private:
    FILE* fp_ = nullptr;
    VideoInfo info_;
};

class Y4MWriter {
public:
    bool open(const std::string& path, const VideoInfo& info);
    bool write(const Frame& f);
    ~Y4MWriter();
private:
    FILE* fp_ = nullptr;
    VideoInfo info_;
};

// PPM P6 (8bit RGB) <-> YCoCg-R 444 フレーム
bool read_ppm_ycocg(const std::string& path, Frame& f, VideoInfo& info);
bool write_ppm_ycocg(const std::string& path, const Frame& f, const VideoInfo& info);

// プレーンの値域 (クリップ用)
void plane_range(const VideoInfo& info, int plane, int32_t& lo, int32_t& hi);

double plane_psnr(const Plane& a, const Plane& b, int bit_depth);

}  // namespace fvc
