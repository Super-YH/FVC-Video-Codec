// FVC: 余弦変調 疑似QMF (PQMF) フィルタバンク。2D 分離型で最大 16x16 = 256 帯域。
#pragma once
#include <vector>

namespace fvc {

class Pqmf1D {
public:
    // M: 帯域数 (2..16), taps_per_band m: 原型長 N = 2*m*M
    Pqmf1D(int M, int m = 6, double kaiser_beta = 9.0);

    int bands() const { return M_; }
    int length() const { return N_; }
    const std::vector<double>& prototype() const { return p_; }

    // 循環拡張による解析。x.size() は M の倍数。out[k] は長さ L/M。
    void analyze(const std::vector<double>& x, std::vector<std::vector<double>>& out) const;
    // 合成。遅延 (N-1) を循環シフトで補償済み。
    void synthesize(const std::vector<std::vector<double>>& in, std::vector<double>& y) const;

private:
    int M_, N_;
    std::vector<double> p_;
    std::vector<std::vector<double>> h_, f_;  // 解析/合成フィルタ
    double gain_ = 1.0;
    void build_filters();
};

// 2D 分離型。画像は row-major (W x H)。W は Mx, H は My の倍数。
// 帯域 (kx, ky) は bands[ky*Mx + kx] に (W/Mx) x (H/My) で格納。
class Pqmf2D {
public:
    Pqmf2D(int Mx, int My, int m = 6, double kaiser_beta = 9.0);
    void analyze(const std::vector<double>& img, int W, int H,
                 std::vector<std::vector<double>>& bands) const;
    void synthesize(const std::vector<std::vector<double>>& bands, int W, int H,
                    std::vector<double>& img) const;
    int bands_x() const { return fx_.bands(); }
    int bands_y() const { return fy_.bands(); }

private:
    Pqmf1D fx_, fy_;
};

double psnr(const std::vector<double>& a, const std::vector<double>& b, double peak);

}  // namespace fvc
