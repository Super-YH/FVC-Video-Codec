// FVC 内部: アフィン図形レイヤ (仕様 §4)
#pragma once
#include <cstdint>
#include <vector>

#include "block_coder.hpp"
#include "fvc/frame.hpp"

namespace fvc {

struct Shape {
    int type = 0;     // 0: ガウス, 1: 楕円, 2: 矩形, 3: 三角形
    int cx = 0, cy = 0;     // 中心 (1/4 画素)
    int l1 = 16, l2 = 16;   // 対数スケール: 半径 = 2^(l/8) 画素
    int theta = 0;          // 回転: 2π/64 単位
    int amp = 0;            // 振幅 (整数画素値)
    int gx = 0, gy = 0;     // 正準座標での線形傾き (amp/16 単位)
    int soft = 4;           // エッジ幅 σ = 2^(soft/4 - 2) 画素 (ガウス以外)
};

// 図形を加算描画 (S += Σ m_s v_s)
void render_shapes(const std::vector<Shape>& shapes, Plane& S);
// 図形リストの構文: 個数 + 直前図形からの差分 (§4.3)
// tref (前フレームの図形リスト) があれば、i 番目の図形を tref[i] からの差分 (動き + パラメータ変化) で送れる (§7.4)
void code_shapes(SymIO& io, CMModel& m, std::vector<Shape>& shapes, const std::vector<Shape>* tref = nullptr);
// 符号器: 原画 (平均除去) にガウス図形を貪欲に当てはめる
std::vector<Shape> fit_shapes(const Plane& org, int max_shapes);
double shapes_bits_estimate(const std::vector<Shape>& shapes);

}  // namespace fvc
