# FVC Video Codec

仕様書: [docs/FVC_SPEC.md](docs/FVC_SPEC.md)

C++17 リファレンス部品 (色変換 / PQMF / 格子VQ・量子化 / 変換・TNS / CM+rANS):

```sh
cmake -S . -B build && cmake --build build -j && ./build/fvc_tests
```

## CLI (段階2: I フレーム)

```sh
./build/fvc enc -q 32 --preset medium in.y4m out.fvc     # 非可逆 (Y4M 420/444 8bit, または PPM → YCoCg-R)
./build/fvc enc --lossless in.ppm out.fvc                # 純ロスレス
./build/fvc enc -q 32 --l2 --pqmf 2 in.ppm out.fvc       # PQMF 16帯域 + L2 ロスレス層 (完全可逆)
./build/fvc dec out.fvc rec.y4m
```

### プリセット (lena 512x512 4:2:0, 1 フレーム, 参考値)

| preset | 有効ツール | qp32 bpp / PSNR-Y | 速度 |
|---|---|---|---|
| faster | 8..32 ブロック, DC/Planar/H/V, CfL | 0.332 / 35.69 | 4.1 fps |
| fast | 8..64, 35 モード, CfL | 0.301 / 35.75 | 1.4 fps |
| medium | 4..64, 35 モード, CfL | 0.297 / 35.74 | 0.9 fps |
| slow | + TNS, IBC | 0.297 / 35.74 | 0.3 fps |
| placebo | + E8 格子 VQ, 全モード RD | 0.296 / 35.79 | 0.04 fps |

追加オプション: `--psy` (ノイズ補完), `--ibc/--no-ibc`, `--e8=0|1`, `--tns=0|1`, `--cfl=0|1`

## 動画 (段階 4〜8)

```sh
./build/fvc enc --preset medium -q 32 --keyint 64 --bframes 1 -v in.y4m out.fvc   # I/P/B/COPY
```

追加オプション: `--refs N`, `--no-copy`, `--dict=0|1`, `--shapes=0|1`, `--fir=0|1`, `-v` (フレーム別統計)

## エンコード情報の表示と目標 SSIM

既定で LAME 風の表示を行います (`--quiet` で抑制、`-v` でフレームごとの QP/サイズ/PSNR/SSIM)。

- 開始時: 入力形式、モード (固定 QP / 目標 SSIM / ロスレス)、プリセット
- 進捗: フレーム数と割合、CPU/実時間と推定、再生速度比、残り時間、kbps、平均 PSNR-Y、平均 SSIM
- 終了時: フレーム種別ごとの枚数・平均サイズ・QP 分布、ブロックモード (イントラ/インター/マージ/スキップ/双予測/長方形/IBC/辞書) とブロックサイズの面積比、平均 PSNR/SSIM、ビットレート、速度

```sh
./build/fvc enc --preset fast --ssim 0.95 in.y4m out.fvc   # 各フレームの輝度 SSIM ≥ 0.95 となる最大 QP を探索
```
