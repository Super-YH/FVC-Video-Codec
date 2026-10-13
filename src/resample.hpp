// FVC 内部: 縮小符号化 (超低レート用) の 2:1 縮小と 1:2 拡大
//  縮小して符号化し、復号後に元の大きさへ拡大する (JPEG 階層モード 1992, H.263 Annex P 1998 と同じ考え方)。
//  フィルタは Lanczos-3 窓付き sinc (Duchon 1979) の整数 (Q14) 係数。拡大は復号器の後処理 (ループ外)
#pragma once
#include "fvc/frame.hpp"

namespace fvc {

// 2:1 縮小 (出力 (w+1)/2 x (h+1)/2)
Plane downsample2(const Plane& in);
// 1:2 拡大 (出力 w x h、w, h は元の大きさ)
Plane upsample2(const Plane& in, int w, int h);

}  // namespace fvc
