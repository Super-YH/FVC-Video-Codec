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
