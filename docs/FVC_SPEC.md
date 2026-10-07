# FVC (Flexible Video Codec) 技術仕様書 v0.1

本書は、YCoCg 色空間・PQMF 帯域分解・アフィン図形表現・可変ブロック変換・格子ベクトル量子化・パラメトリック補完・辞書参照・コンテキストミキシング + rANS を組み合わせた映像/画像コーデック **FVC** の、C++ で実装可能な粒度の仕様である。

- 規範 (normative) 部分: 復号処理・ビットストリーム構文。復号器はこれに完全に従うこと。
- 参考 (informative) 部分: 符号器の探索・最適化。品質/速度のトレードオフで自由に変えてよい。
- 本リポジトリの `include/fvc/` と `src/` に、各章の中核アルゴリズムのリファレンス実装がある (§14)。

---

## 0. 記法と共通定義

| 記号 | 意味 |
|---|---|
| $x[n],\ x[i,j]$ | 1D/2D 信号 (2D は行 $i$, 列 $j$、row-major) |
| $W,H$ | フレーム幅・高さ (画素) |
| $\lfloor\cdot\rfloor,\ \mathrm{round}(\cdot)$ | 床関数, 最近接整数 (0.5 は偶数丸め: `std::nearbyint`) |
| $a \gg b$ | 算術右シフト (負数は $-\infty$ 方向へ丸め) |
| $\mathrm{clip}(v,a,b)$ | $\min(\max(v,a),b)$ |
| $J = D + \lambda R$ | RD コスト (D: 歪み (SSE 等), R: ビット数) |
| $\mathcal{Q}_\Lambda(\cdot)$ | 格子 $\Lambda$ の最近傍点量子化 |
| `u(n)` / `ae(v)` | 構文: 固定長 n ビット (バイパス) / CM+rANS 符号化要素 |

**決定性**: 復号結果は全プラットフォームでビット一致しなければならない。浮動小数で定義された処理 (PQMF, 変換, 図形描画, 非線形変換) は、規範上は **固定小数点版** を用いる (§13.3)。本書の実数式は意味の定義であり、固定小数点化の規則は §13.3 で与える。

---

## 1. 全体構成

### 1.1 処理パイプライン (符号器)

```
入力 RGB ──► [§2 YCoCg-R] ──► Y, Co, Cg
                    │
                    ├─► [§2.3 色予測] Co,Cg の残差化 (選択)
                    ▼
            [§3 PQMF 帯域分解 (複数フィルタ候補→最良選択)]
                    │ 帯域画像 B_k (最大256) / 帯域間差分 / 合成残差 r_pqmf
                    ▼
     ┌──────────────┼───────────────────────────┐
     ▼              ▼                           ▼
[§4 アフィン図形] [§5 ブロック分割+変換]   [§6 コピー/辞書(§10)]
     │              │ DCT/DST→帯域分割→パラメトリック補完
     │              │ →非線形変換→ 格子VQ / ランダム投影 / 非線形SQ
     └──────┬───────┴───────────┬───────────────┘
            ▼                   ▼
   [§7 インター予測 P/B/Copy, ゲイン/パン, FIR動き, パラメトリック残差]
            ▼
   [§8 L1 残差 → L2 残差 (ロスレス可)]
            ▼
   [§9 ループフィルタ: デブロック + 周波数ゲイン + 画素ゲイン]
            ▼
   [§11 2値化 → CM (PPMd型文脈+ミキサ+APM) → rANS] ──► ビットストリーム
```

### 1.2 データ階層

```
Sequence
 └ GOP / RAP 区間
    └ Frame (I / P / B / COPY)
       └ Plane (Y, Co, Cg  ※ 4:4:4 / 4:2:0 / 4:2:2 を SPS で指定)
          └ Band layer: PQMF 帯域画像 B_{p,k}  (k = ky*Mx + kx)
             └ Tile (独立復号単位, 最大 4096x4096)
                └ Region: Shape-list (§4) と Block-tree (§5) の重ね合わせ
```

各 Plane はまず「帯域なし (Mx=My=1)」でも符号化できる。その場合 §3 はバイパスされ、§4〜§9 は画素領域に直接適用される。

### 1.3 再構成の加法モデル (規範)

帯域 $k$ の再構成は、各ツールの寄与の **加算** で定義する:

$$
\hat B_k = \underbrace{P_k}_{\text{予測 (§6,§7)}} + \underbrace{S_k}_{\text{図形 (§4)}} + \underbrace{T_k}_{\text{変換係数 (§5)}} + \underbrace{F_k}_{\text{パラメトリック補完 (§5.6,§7.6)}}
$$

各項は領域ごとに存在しても無くてもよく、無い項は 0。画素への合成:

$$
\hat X_p = \mathrm{Syn}_{\mathcal F}\big(\{\hat B_{p,k}\}_k\big) + \hat r^{\,\mathrm{pqmf}}_p,\qquad
\hat X^{\mathrm{final}}_p = \mathrm{LF}(\hat X_p) + \hat e^{(1)}_p + \hat e^{(2)}_p
$$

($\mathrm{LF}$: §9 ループフィルタ、$e^{(1)},e^{(2)}$: §8 の L1/L2 残差)。

---

## 2. 色空間と色予測

### 2.1 YCoCg-R (可逆, 規範)

入力ビット深度 $b$ の RGB に対し:

$$
\begin{aligned}
C_o &= R - B, & t &= B + (C_o \gg 1),\\
C_g &= G - t, & Y &= t + (C_g \gg 1).
\end{aligned}
$$

逆変換:

$$
t = Y - (C_g \gg 1),\quad G = C_g + t,\quad B = t - (C_o \gg 1),\quad R = B + C_o.
$$

$Y$ は $b$ ビット、$C_o, C_g$ は $b+1$ ビット (符号付き)。整数演算のみで完全可逆 (実装: `fvc::rgb_to_ycocg_r`)。

SPS の `color_transform` で `0: YCoCg-R, 1: 恒等 (入力が既に YUV/単色), 2: 非可逆 YCoCg (浮動、高ビット深度用)` を選ぶ。

### 2.2 クロマサブサンプリング

`chroma_format ∈ {444, 422, 420}`。ダウンサンプルは符号器の自由 (informative)。復号側アップサンプルは規範: 4 タップ FIR $[-1, 9, 9, -1]/16$ を分離型で適用、境界は対称拡張。

### 2.3 色予測 (CfL 型, 選択的残差摘出)

ブロック/帯域領域 $\Omega$ ごとに、再構成済みの輝度 $\hat Y$ (サブサンプル位置に合わせて平均ダウンサンプル) から色差を予測する:

$$
\tilde C[i] = \left(\alpha_q\,(\hat Y[i]-\bar Y)\right) \gg 6 \;+\; \beta,\qquad
\bar Y = \left\lfloor \tfrac{1}{|\Omega|}\sum_{i\in\Omega}\hat Y[i] \right\rfloor
$$

- $\alpha_q \in [-128,127]$ (精度 1/64)、$\beta$ は整数オフセット。
- 符号器は最小二乗 $\alpha^* = \frac{\sum (Y-\bar Y)(C-\bar C)}{\sum (Y-\bar Y)^2}$, $\beta^*=\bar C$ を量子化 (`fit_chroma_pred`)。
- $\beta$ は隣接ブロックの $\beta$ からの差分で送る。$\alpha_q$ は隣接 $\alpha_q$ を文脈に CM 符号化。
- **追加モード**: $C_g$ は $\hat Y$ に加え、既復号の $\hat C_o$ も説明変数とする 2 変数モデル $\tilde C_g = (\alpha_1(\hat Y-\bar Y) + \alpha_2(\hat C_o - \bar C_o))\gg 6 + \beta$ を選択可。

**残差摘出フラグ** `chroma_pred_mode` (領域単位):

| 値 | 意味 |
|---|---|
| 0 | 色予測なし ($C$ をそのまま後段へ) |
| 1 | 予測のみ (残差は捨てる: $\hat C = \tilde C$)。低ビットレート向け |
| 2 | 予測 + 残差摘出 ($R_C = C - \tilde C$ を後段 §3〜§8 で符号化) |
| 3 | 予測 + 残差摘出 + 残差は L2 ロスレス層に直行 |

---

## 3. PQMF 帯域分解

### 3.1 余弦変調 PQMF (規範)

1 次元 $M$ 帯域 ($M \in \{1,2,4,8,16\}$、2D 分離で最大 $16\times16=256$ 帯域、$M_x \ne M_y$ も可)。原型低域フィルタ $p[n]$ ($N = 2mM$ タップ, $m$ = 重なり係数) から:

$$
h_k[n] = 2p[n]\cos\!\left(\frac{\pi}{M}\left(k+\tfrac12\right)\left(n-\tfrac{N-1}{2}\right) + \theta_k\right),\quad
f_k[n] = 2p[n]\cos\!\left(\frac{\pi}{M}\left(k+\tfrac12\right)\left(n-\tfrac{N-1}{2}\right) - \theta_k\right)
$$

$$
\theta_k = (-1)^k \frac{\pi}{4},\qquad k = 0,\dots,M-1.
$$

**解析** (間引き $M$、長さ $L$ の信号を循環拡張):

$$
u_k[q] = \sum_{n=0}^{N-1} h_k[n]\; x[(qM - n) \bmod L]
$$

**合成** (遅延 $N-1$ を補償):

$$
y[n] = g\sum_{k=0}^{M-1}\sum_{q} u_k[q]\, f_k[(n + N - 1 - qM) \bmod L]
$$

$g$ は全体利得の正規化定数 (インパルス応答から決定)。

**原型フィルタ設計** (Lin–Vaidyanathan のカイザー窓法): 

$$
p[n] = \frac{\sin(\omega_c(n-c))}{\pi(n-c)}\cdot \frac{I_0\!\left(\beta\sqrt{1-(2n/(N-1)-1)^2}\right)}{I_0(\beta)},\quad c=\tfrac{N-1}{2}
$$

$\omega_c$ は次の準完全再構成誤差を最小化するよう黄金分割探索で決める:

$$
\omega_c^* = \arg\min_{\omega_c}\ \max_{n\neq 0} \frac{|g_{pp}[c' + 2Mn]|}{|g_{pp}[c']|},\quad g_{pp} = p * p,\ c'=N-1.
$$

規範上は、符号器が選べる **フィルタ集合** $\mathcal{F}$ の各係数を固定小数点テーブル (Q1.30) として規定する (§13.3)。初期集合:

| filter_id | m | β | 用途 | 参考 PSNR (8bit, 往復) |
|---|---|---|---|---|
| 0 | 6 | 9 | 標準 (高分離) | ≈ 68–72 dB |
| 1 | 4 | 6 | 短タップ (低遅延・低リンギング) | ≈ 55–62 dB |
| 2 | 8 | 9 | 長タップ (最大分離) | ≈ 65–75 dB |
| 3 | — | — | Haar/整数リフティング (可逆, $M=2^j$ の木構造) | ∞ (完全可逆) |

(参考値はリファレンス実装 `tests/test_main.cpp` で実測。)

PQMF は準完全再構成なので、ロスレス/高品質時は §3.4 の合成残差で誤差を回収する。

### 3.2 複数フィルタ同時適用とパラメータ化

符号器は **各タイル・各プレーン** について $\mathcal F$ の全候補 (または部分集合) で同時に分解し、

$$
f^* = \arg\min_{f\in\mathcal F,\ (M_x,M_y)} \Big( D\big(X,\ \mathrm{Syn}_f(\mathcal{Q}(\mathrm{Ana}_f X))\big) + \lambda R_f \Big)
$$

を選ぶ。簡易版は「帯域係数の $\ell_1$ ノルム + 合成残差エネルギー」の最小化 (高速推定)。選択結果は `pqmf_filter_id`, `log2_Mx`, `log2_My` として送る。

**帯域ごとの部分木分解** (`band_split_flag`): 帯域 $k$ を更に $2\times2$ / $1\times2$ / $2\times1$ に再分解でき、非一様分解 (ウェーブレットパケット的) を表現できる。合計帯域数は 256 以下。

### 3.3 帯域間差分 (Band-difference) 

帯域 $k$ を、参照帯域 $j$ (同プレーン or 他プレーン) からの予測差分で表すことができる:

$$
D_k = B_k - \left(a_{kj}\,\mathcal{T}_{kj}(\hat B_j) + b_{kj}\right)
$$

- $\mathcal{T}_{kj}$ ∈ {恒等, 左右反転 (隣接帯域のスペクトル鏡像: $(-1)^n$ 変調), 上下反転, 両反転}。PQMF の奇数帯域は鏡像スペクトルになるため反転が有効。
- $a_{kj}$ は Q4.8 固定小数、$b_{kj}$ は整数。最小二乗で求める。
- `band_ref_mode` で {なし, 同プレーン, クロスプレーン (Y→Co 等)} を指定。

### 3.4 合成残差 (残差摘出)

$$
r^{\mathrm{pqmf}} = X - \mathrm{round}\left(\mathrm{Syn}_{f^*}(\hat B)\right)
$$

`pqmf_residual_mode`: 0=破棄, 1=§5 で非可逆符号化, 2=§8 L2 層でロスレス符号化。リファレンス実装では非量子化時 $|r^{\mathrm{pqmf}}|\le 2$ (8bit) を確認済み。

---

## 4. アフィン図形の重ね合わせ表現

帯域画像および残差を、パラメトリックな図形 (プリミティブ) の加算合成で表す。

### 4.1 プリミティブの定義

図形 $s$ はパラメータ $\Theta_s = (\tau_s, A_s, \mathbf{t}_s, \mathbf{a}_s, \sigma_s, \mathrm{blend}_s)$:

- $\tau_s$: 形状タイプ
- $A_s \in \mathbb{R}^{2\times2}$: アフィン行列, $\mathbf t_s\in\mathbb R^2$: 平行移動
- $\mathbf a_s$: 振幅プロファイルのパラメータ
- $\sigma_s$: エッジのソフトネス
- blend: {加算, 置換, 乗算}

正準座標 $\mathbf u = A_s^{-1}(\mathbf x - \mathbf t_s)$ における基底形状関数 $\phi_\tau(\mathbf u)$:

| τ | 名称 | $\phi_\tau(\mathbf u)$ (エッジ前) |
|---|---|---|
| 0 | 楕円 (単位円) | $d(\mathbf u) = \lVert\mathbf u\rVert - 1$ |
| 1 | 矩形 (単位正方形) | $d = \max(|u_1|,|u_2|) - 1$ |
| 2 | 三角形 | 3 辺の符号付き距離の最大 |
| 3 | ガウス | $\exp(-\tfrac12\lVert\mathbf u\rVert^2)$ (エッジ不要) |
| 4 | 多角形 (頂点数 ≤ 16, 正準座標で頂点を送る) | 凸: 辺距離の最大 / 非凸: 巻数 |
| 5 | ガボール | $\exp(-\tfrac12\lVert\mathbf u\rVert^2)\cos(\omega u_1+\varphi)$ |
| 6 | ストローク (ベジェ曲線, 制御点 ≤ 4) | 曲線までの距離 − 半幅 |
| 7 | 辞書パッチ (§10 の項目をアフィン変形) | 双三次補間したパッチ値 |

符号付き距離 $d$ を持つ形状のマスク:

$$
m_s(\mathbf x) = \mathrm{sigmoid}\!\left(-\frac{d(\mathbf u)}{\sigma_s}\right) \quad(\sigma_s\to0\ \text{で硬いエッジ})
$$

振幅プロファイル ($\mathbf a_s$):

$$
v_s(\mathbf u) = a_0 + a_1 u_1 + a_2 u_2 \;\;(+\ a_3 u_1^2 + a_4 u_1u_2 + a_5 u_2^2\ \text{: 2次オプション})
$$

図形の寄与と、領域 $\Omega$ における図形層の再構成:

$$
S(\mathbf x) = \bigoplus_{s=1}^{N_s} m_s(\mathbf x)\, v_s(A_s^{-1}(\mathbf x-\mathbf t_s))
$$

$\oplus$ は blend に従い順に適用 (加算: $S \leftarrow S + m v$, 置換: $S \leftarrow (1-m)S + m v$, 乗算: $S\leftarrow S(1 + m v)$)。

### 4.2 アフィン行列のパラメータ化

数値安定・量子化しやすい形として、対数スケール・回転・せん断で表す:

$$
A = R(\theta)\begin{pmatrix} e^{\ell_1} & 0 \\ 0 & e^{\ell_2}\end{pmatrix}\begin{pmatrix}1 & \kappa\\ 0 & 1\end{pmatrix}
$$

量子化: $\theta$: $2\pi/256$ 刻み、$\ell_{1,2}$: $1/16$ 刻み (サイズ $2^{\ell}$ 画素)、$\kappa$: $1/32$ 刻み、$\mathbf t$: 1/8 画素、$\sigma$: $\log_2$ で 1/4 刻み、$\mathbf a$: 帯域ステップ $\Delta_k$ 単位。

### 4.3 参照差分符号化 (図形間予測)

図形は **既に符号化された図形を参照** してパラメータ差分で記録する:

$$
\Theta_s = \Theta_{\mathrm{ref}(s)} \boxplus \delta\Theta_s
$$

- `ref_idx`: 0 = 参照なし (絶対値, 予測は空間近傍の中央値), 1..K = 直近 K (=64) 個の図形リスト (MRU 順)、K+1.. = 他帯域の同位置図形 (帯域間相関)、辞書 §10 の図形項目。
- $\boxplus$: 位置・振幅は加算、$\theta$ は mod $2\pi$ 加算、$\ell$ は加算 (=スケール乗算)。
- 他帯域を参照する場合は帯域のサブサンプル比で位置・スケールを自動換算: $\mathbf t' = \mathbf t \cdot (M_{\mathrm{ref}}/M_{\mathrm{cur}})$, $\ell' = \ell + \log(M_{\mathrm{ref}}/M_{\mathrm{cur}})$。
- `copy_flags` (ビットマスク) で「タイプ/行列/位置/振幅」それぞれについて差分 0 (完全コピー) を 1 ビットで指定。

### 4.4 符号器の当てはめ (informative)

1. **初期化**: 残差 $E = B - (\text{他ツールの寄与})$ に対し、マッチング追跡。候補生成は (a) 局所極値周辺のガウス、(b) エッジ検出 + 主成分分析による楕円/矩形、(c) 辞書パッチ。
2. **精密化**: $L(\Theta) = \sum_{\mathbf x}(E(\mathbf x) - S_\Theta(\mathbf x))^2 + \lambda R(\Theta)$ を Adam / Gauss–Newton で最適化 ($\sigma$ を大→小へアニーリング)。
3. **受理判定**: 追加による $\Delta J < 0$ のときのみ採用。最大図形数は領域あたり 4096。

### 4.5 規範描画

復号器は固定小数点で描画する: $\mathbf u$ は Q16.16、sigmoid は 1024 エントリ LUT ($d/\sigma \in [-8,8)$)、exp は 1024 エントリ LUT。描画は図形のバウンディングボックス ($d < 8\sigma$ の範囲) に限定。

---

## 5. ブロック分割・変換・量子化

### 5.1 ブロック分割

帯域画像 (あるいは画素領域) を最大 **1024×1024** の CTU に分割し、再帰的に分割する。ブロックサイズ $w,h \in \{4, 8, 16, \dots, 1024\}$ (2 の冪) に加え、`arbitrary_rect` モードで 4 の倍数の任意長方形 (例 12×20) を許す。

分割ノードの種類 `split_mode`:

| 値 | 分割 |
|---|---|
| 0 | 葉 |
| 1 | 4 分木 (QT) |
| 2 / 3 | 2 分木 水平 / 垂直 (BT) |
| 4 / 5 | 3 分木 水平 / 垂直 (1:2:1) |
| 6 / 7 | 非対称 2 分 水平 / 垂直 (1:3, 3:1 は `asym_pos`) |
| 8 | 任意位置 2 分 (分割位置を 4 画素単位で送る) |
| 9 | 図形マスク分割 (§4 の図形 1 個を境界として 2 領域に分割) |

制約: 葉の最小サイズ 4×4、縦横比 ≤ 64:1、深さ ≤ 12。

### 5.2 変換

葉ブロック (残差 $e$, $w\times h$) に分離型変換:

$$
C = T_v\, e\, T_h^{\top}
$$

$$
\text{DCT-II: } T[k,n] = \sqrt{\tfrac{2-\delta_{k0}}{N}}\cos\frac{\pi(2n+1)k}{2N},\qquad
\text{DST-VII: } T[k,n] = \sqrt{\tfrac{4}{2N+1}}\sin\frac{\pi(2n+1)(k+1)}{2N+1}
$$

`tx_h, tx_v ∈ {DCT2, DST7, IDTX(恒等)}`。DST-VII は 4〜64 まで、DCT-II は 4〜1024 まで。1024 点は高速アルゴリズム (Loeffler 型バタフライ / FFT 経由) で実装する。規範は整数近似行列 (§13.3)。

### 5.3 係数の複数バンド・パーティション分割

変換係数 $C[u,v]$ を周波数の **バンド** に分割し、さらにバンドを **パーティション** (ベクトル化単位) に分ける。

**放射状バンド** (既定): 正規化周波数 $\rho(u,v) = \sqrt{(u/w)^2 + (v/h)^2}$ に対し境界 $0=\rho_0<\rho_1<\dots<\rho_{N_b}=\sqrt2$ を設定:

$$
\mathcal{B}_b = \{(u,v) : \rho_b \le \rho(u,v) < \rho_{b+1}\}
$$

境界はブロックサイズ別の既定テーブル (対数間隔) から `band_layout_id` で選ぶか、明示送信。
**方向バンド** (`band_dir=1`): 角度 $\arctan(v h/(u w))$ でさらに 2〜4 分割。

各バンド内の係数はジグザグ順に並べ、長さ $L_p \in\{4, 8, 16, 24, 32\}$ のパーティション $\mathbf c_{b,p}$ に分割。

**パーティション連結** (`part_join`): 複数パーティション (同一ブロックの別バンド、別ブロック、別帯域画像、別プレーン) を連結して一つのベクトルとして扱い、同一の量子化器・利得で処理できる。

$$
\mathbf c_{\mathrm{joint}} = [\mathbf c_{b_1,p_1};\ \mathbf c_{b_2,p_2};\ \dots] \in \mathbb{R}^{\sum L}
$$

連結は `join_group_id` で指定し、グループ単位で 1 つの利得 $g$・量子化モードを送る。

### 5.4 利得・形状分離

各パーティション (または連結グループ) を利得と形状に分ける:

$$
g = \lVert \mathbf c\rVert_2,\qquad \mathbf s = \mathbf c / g
$$

$g$ は対数領域で量子化 ($\hat g = 2^{q_g/4}\Delta_b$、$q_g$ はバンド間・ブロック間差分で符号化)。形状 $\mathbf s$ は §5.5 の量子化器で量子化し、復号時に $\hat{\mathbf c} = \hat g\,\hat{\mathbf s}/\lVert\hat{\mathbf s}\rVert$ (エネルギー保存)。`gain_shape=0` の場合は $\mathbf c$ を直接量子化する。

### 5.5 量子化 (格子 VQ・ランダム投影・非線形スカラ量子化の組み合わせのみ)

量子化モード `qmode` はこの 3 種の **合成** だけで構成する:

$$
\hat{\mathbf c} = \mathcal{N}^{-1}\Big( P^{+}\,\mathcal{Q}\big(P\,\mathcal{N}(\mathbf c)\big)\Big)
$$

- $\mathcal N$: §5.7 の非線形前変換 (恒等可)
- $P$: ランダム投影 (使わない場合は $I$)
- $\mathcal Q$: 格子 VQ または 非線形スカラ量子化

#### (a) 格子ベクトル量子化

$\mathcal Q_\Lambda(\mathbf y) = \Delta\cdot \mathrm{NP}_\Lambda(\mathbf y/\Delta)$。格子は $\mathbb Z^n$, $D_n$, $E_8$ (8 次元ごとに分割; 長さが 8 の倍数でなければ端数を $D_n$ / $\mathbb Z^n$)。

**最近傍点アルゴリズム** (Conway–Sloane):

- $\mathbb Z^n$: $y_i = \mathrm{round}(x_i)$
- $D_n = \{\mathbf y\in\mathbb Z^n : \sum y_i \equiv 0 \pmod 2\}$: 各座標を丸め、和が奇数なら丸め誤差最大の座標 $i^*$ を逆方向に丸め直す ($f(x_{i^*})$ を $\pm1$)。
- $E_8 = D_8 \cup (D_8 + \tfrac12\mathbf 1)$: $\mathbf a = \mathrm{NP}_{D_8}(\mathbf x)$, $\mathbf b = \mathrm{NP}_{D_8}(\mathbf x - \tfrac12\mathbf1) + \tfrac12\mathbf1$ のうち近い方。

正規化二次モーメント (リファレンス実測): $G(\mathbb Z^8)=0.0833$, $G(D_8)=0.0759$, $G(E_8)=0.0719$ → $E_8$ はスカラ量子化比 **約 0.65 dB** の粒状利得。

**インデックス化** (整数シンボル列へ):

- $\mathbb Z^n$: $z_i = y_i$。
- $D_n$: $z_i = y_i\ (i<n-1)$, $z_{n-1} = \lfloor (y_{n-1} - \pi)/2\rfloor$, $\pi = (\sum_{i<n-1} y_i)\bmod 2$。(1 ビット節約)
- $E_8$: コセットビット $c = (2y_0)\bmod 2$, $z_i = (2y_i - c)/2$ に対し $\sum z_i$ 偶数なので $D_8$ と同様に $z_7$ を半分にする。

復号は逆写像 (`fvc::index_to_lattice`)。$z_i$ は §11 の符号付き整数 2 値化で CM 符号化 (文脈: バンド番号, 座標位置, 近傍パーティションの $\lVert\mathbf z\rVert_1$)。

#### (b) ランダム投影

シード $s$ (スライスヘッダ + パーティション番号から導出) で $m\times n$ 行列 $P$ を生成 ($m\le n$):

$$
P_{ij} = \pm\frac{1}{\sqrt m}\quad (\text{符号は SplitMix64}(s) \text{ の最上位ビット})
$$

符号器は $\mathbf y = P\mathbf c$ を (a) または (c) で量子化。復号は最小ノルム解

$$
\hat{\mathbf c} = P^{\top}(PP^{\top})^{-1}\hat{\mathbf y}
$$

で再構成する (ノイズ的・高周波バンドで $m/n = 1/4\sim1/2$ に次元削減)。オプション `rp_sparse_recon=1` では、復号器は $\hat{\mathbf c}$ に対しバンド包絡に基づく軟閾値反復 (ISTA, 反復回数 $T\le 8$ を規範固定) を適用:

$$
\mathbf c^{(t+1)} = \mathcal{S}_{\eta}\!\left(\mathbf c^{(t)} + P^{\top}(PP^\top)^{-1}(\hat{\mathbf y} - P\mathbf c^{(t)})\right)
$$

#### (c) 非線形スカラ量子化

べき乗コンパンディング + デッドゾーン:

$$
v = \frac{\mathrm{sgn}(x)|x|^{\gamma}}{\Delta},\quad q = \mathrm{sgn}(v)\left\lfloor |v| + 1 - \theta \right\rfloor,\quad
\hat x = \mathrm{sgn}(q)\left((|q|+\delta)\Delta\right)^{1/\gamma}\ (q\ne0)
$$

$\gamma\in\{1, 0.875, 0.75, 0.625, 0.5\}$, $\theta\in[0.5,1)$ (符号器のデッドゾーン; 規範外), $\delta \in [-0.5, 0]$ (再構成オフセット; 規範, 1/16 刻み)。

**組み合わせ例**:

| qmode | 構成 | 典型用途 |
|---|---|---|
| 0 | 非線形SQ | 低周波 DC/AC |
| 1 | $E_8$ 格子 VQ (利得形状) | 中周波 |
| 2 | ランダム投影 → 非線形SQ | 高周波ノイズ様 |
| 3 | ランダム投影 → $D_n$ VQ | 連結パーティション |
| 4 | 非線形前変換 → $E_8$ VQ | 高ダイナミックレンジ |

量子化ステップ: $\Delta_{b} = \Delta_0 \cdot 2^{(QP - 4)/6}\cdot w_b$ ($QP\in[0,63]$ を 1/4 刻み拡張、$w_b$ はバンド重み行列; 帯域画像 $k$ ごとにも $w_k$ を掛ける)。

### 5.6 パラメトリックなバンド補完

量子化で 0 になった (あるいは送らない) バンドを、パラメータで埋める (`fill_mode`、バンドごと):

1. **ノイズ置換 (PNS)**: $\hat c(u,v) = \hat g_b\,\frac{n(u,v)}{\lVert \mathbf n\rVert}\sqrt{|\mathcal B_b|}\cdot\rho$, $n\sim$ 擬似乱数 (SplitMix64, シード規範), $\rho$ = 包絡傾き補正。
2. **係数コピー (帯域複製)**: 他バンド $b'$ (同ブロック・他ブロック・他帯域画像) の係数をコピー: $\hat{\mathbf c}_b = \hat g_b \cdot \Pi(\hat{\mathbf c}_{b'})/\lVert\cdot\rVert$。$\Pi$ は §7.6 の並び替え。
3. **包絡補間**: 隣接バンドの $\hat g$ から $\log$ 線形補間した利得 + ノイズ。
4. **符号のみ**: 振幅は包絡モデル $|c| \approx g_b \cdot (\rho/\rho_b)^{-\alpha}$, 符号を 1 ビット/係数で送る (`sign_only`)。

### 5.7 非線形前変換 (量子化前に潜らせる)

量子化前に、最大 2 段の単調可逆変換を適用できる (`prenl_type[2]`, `prenl_param[2]`):

| type | 順変換 $\mathcal N(x)$ | 逆変換 |
|---|---|---|
| 0 | $x$ | $y$ |
| 1 (Power) | $\mathrm{sgn}(x)|x|^a$ | $\mathrm{sgn}(y)|y|^{1/a}$ |
| 2 (Asinh) | $\sinh^{-1}(x/a)$ | $a\sinh y$ |
| 3 (SignedLog) | $\mathrm{sgn}(x)\log(1+|x|/a)$ | $\mathrm{sgn}(y)\,a(e^{|y|}-1)$ |
| 4 (PWL) | 区分線形 (節点 ≤ 8、単調増加、送信) | 逆区分線形 |

規範実装は 4096 エントリ LUT + 線形補間 (§13.3)。

---

## 6. イントラコピー と TNS

### 6.0 方向性イントラ予測 (実装済み)

ブロック葉の予測モード: 0 = DC, 1 = Planar, 2〜34 = 角度予測 (HEVC 互換の角度表
$A = \{32,26,21,17,13,9,5,2,0,-2,\dots,-32,\dots,32\}$, 1/32 画素精度の線形補間、負角は逆角度 $\lfloor 8192/A\rceil$ で側辺参照を投影)、35 = CfL (色差のみ、§2.3: DC 予測 + $\alpha_q/8\cdot(\hat Y - \bar Y)$, $\alpha_q\in[-16,16]$)。
モードは MPM 2 候補 (左・上) フラグ + インデックス、それ以外は 0..34 を CM 符号化。

**実装済みツールの実測効果** (lena 512×512 4:2:0, qp32, medium 基準):
CfL −3.4% ビット (色差 +0.9 dB)、33 方向角度 + RD 改善で段階2比 −14%、
TNS ≈ 0 (符号化時間 2 倍; slow 以上で有効)、E8 格子 VQ ≈ 0 (低レートではデッドゾーン付きスカラと同等; placebo のみ)、IBC: 自然画像では ≈ 0 (スクリーンコンテンツ向け)。

### 6.1 パラメトリックコピー (帯域内・帯域間・色残差間)

対象ブロック/図形 $\Omega$ を、既復号領域のソースから変形してコピー:

$$
\hat B_{\mathrm{dst}}(\mathbf x) = a\cdot\mathcal W\!\left(\hat B_{\mathrm{src}}\right)(\mathbf x) + b + \mathbf g^\top(\mathbf x - \mathbf x_0)
$$

- **ソース**: `src_space` ∈ {同帯域, 他帯域 (帯域インデックス), 他プレーン (Y/Co/Cg/色残差), 辞書 §10, 参照フレーム (§7)}。
- **変形** $\mathcal W$: 変位ベクトル (1/4 画素; 帯域間ではスケール換算)、8 種の二面体変換 (回転 90°×k, 反転)、オプションでアフィン $A$ (§4.2 のパラメータ化)、補間は 8 タップ (§7.3 の FIR と共通)。
- **振幅**: $a$ (Q4.8)、$b$、線形傾き $\mathbf g$ (オプション)。
- **周波数領域コピー**: `copy_domain=1` で変換係数間のコピー (ソースブロックの係数 $\hat C_{\mathrm{src}}$ をバンド選択付きでコピー、例: 低周波のみコピーして高周波は §5 で送る)。

ベクトル予測: 近傍ブロックのコピーベクトル候補リスト (最大 6) からインデックス + 差分。

### 6.2 TNS (時間/空間ノイズ整形 → 周波数領域の残差整形)

変換係数列 (ブロック内、ジグザグ or 行/列方向; `tns_dir`) $c[k]$ に対し周波数方向の LPC フィルタを適用し、量子化ノイズを空間領域で信号包絡に沿って整形する。

**符号器** (解析, FIR):

$$
e[k] = c[k] + \sum_{i=1}^{p} a_i\,c[k-i],\qquad p\le 8
$$

**復号器** (合成, IIR):

$$
\hat c[k] = \hat e[k] - \sum_{i=1}^{p} a_i\,\hat c[k-i]
$$

係数は自己相関 $R[l] = \sum_k c[k]c[k-l]$ から Levinson–Durbin で PARCOR $r_i$ を求め、arcsin 量子化:

$$
q_i = \mathrm{round}\!\left(\sin^{-1}(r_i)\cdot\frac{2^{b-1}-\tfrac12}{\pi/2}\right),\quad b=4,\qquad \hat r_i = \sin\!\left(q_i \frac{\pi/2}{2^{b-1}-\tfrac12}\right)
$$

$\hat r_i$ から step-up 再帰で $a_i$ を復元 ($a^{(i)}_j = a^{(i-1)}_j + \hat r_i a^{(i-1)}_{i-j}$)。$|\hat r_i| < 1$ なので合成フィルタは常に安定。TNS は `tns_start_band`〜`tns_end_band` の範囲のみに適用でき、最大 3 区間。

---

## 7. インター予測 (P / B / COPY フレーム)

### 7.1 フレーム種別

| frame_type | 内容 |
|---|---|
| I | §2〜§6 のみ |
| P | 参照リスト L0 (最大 16 枚) |
| B | L0 + L1 (双予測、重み付き) |
| COPY | 参照フレームを丸ごとコピー。グローバルゲイン・パン・グローバル動きのパラメータのみ (残差なし or 低次元パラメトリック残差のみ) |

参照ピクチャは再構成済みの **帯域画像単位** でも保持し、帯域ごとに予測できる (`inter_domain` = 画素 / 帯域)。

### 7.2 グローバルパラメータ (ゲイン・パン・グローバル動き)

フレーム (またはタイル) 単位:

$$
P_{\mathrm{glob}}(\mathbf x) = g_p\cdot \hat X_{\mathrm{ref}}\!\left(\mathcal G(\mathbf x)\right) + o_p
$$

- $\mathcal G$: 並進 (パン) / 相似 / アフィン / 射影 (8 パラメータ):
  $\mathcal G(\mathbf x) = \frac{(h_{11}x+h_{12}y+h_{13},\ h_{21}x+h_{22}y+h_{23})}{h_{31}x+h_{32}y+1}$
- $g_p, o_p$: プレーン別ゲイン/オフセット (フェード, 明滅)。さらに帯域別ゲイン $g_{p,k}$ (`band_gain_flag`) で周波数依存のフェード (フォーカス変化) を表現。
- **パン探索** (informative): 位相相関で並進候補 → 特徴点 RANSAC でアフィン/射影 → $\min_{\Theta}\sum (X-P_{\mathrm{glob}})^2$ を Gauss–Newton で精密化。ゲインは $g = \mathrm{Cov}(X,P)/\mathrm{Var}(P)$。

### 7.3 ブロック動き補償と低次元 FIR 動き

ブロック単位の動きベクトル $\mathbf d = (d_x,d_y)$ (1/16 画素) に加え、**低次元 FIR カーネル** で動きとぼけを同時に表す:

$$
P(\mathbf x) = \sum_{\mathbf m \in \mathcal K} h[\mathbf m]\ \hat X_{\mathrm{ref}}\big(\lfloor\mathbf x + \mathbf d\rfloor + \mathbf m\big)
$$

$\mathcal K$ はサポート (`fir_shape`: 分離型 $2\times4$ タップ / 3×3 / 5×5 十字)。カーネル $h$ は低次元パラメータ $\boldsymbol\psi$ から生成:

$$
h = h_{\mathrm{frac}}(\mathbf d - \lfloor \mathbf d\rfloor) \;*\; h_{\mathrm{blur}}(\sigma_x,\sigma_y,\rho) \;+\; \sum_{j=1}^{J} \psi_j\,\mathbf b_j
$$

- $h_{\mathrm{frac}}$: 分数画素補間 (8 タップ DCT-IF、規範テーブル)
- $h_{\mathrm{blur}}$: 異方性ガウスぼけ (動きぼけ/フォーカス変化)
- $\mathbf b_j$: 規範で定める直交 FIR 基底 ($J\le 4$, 例: 1 次微分 $\partial_x,\partial_y$、ラプラシアン、鞍点) → **動き成分摘出**: 残差の中で局所変位/変形に起因する成分を $\psi_j$ に吸収。
- 符号器は最小二乗 $\min_{\boldsymbol\psi}\lVert X_\Omega - P_{\boldsymbol\psi}\rVert^2$ (線形なので正規方程式で解ける)。

同じ FIR 動き記述を **参照帯域画像** に直接適用できる (帯域ドメインでの MC: 帯域 $k$ のブロックは参照の帯域 $k$ (または隣接帯域の線形結合) から予測):

$$
P_k(\mathbf x) = \sum_{k'\in\mathcal N(k)} \alpha_{kk'} \sum_{\mathbf m} h_{kk'}[\mathbf m]\, \hat B^{\mathrm{ref}}_{k'}(\mathbf x + \mathbf d/M + \mathbf m)
$$

($\mathcal N(k)$: 帯域 $k$ とその隣接帯域 — 間引きによるエイリアシング成分を補償)。

**動きベクトル場の分解**: ブロック MV は $\mathbf d(\mathbf x) = \mathbf d_{\mathrm{glob}}(\mathbf x) + \mathbf d_{\mathrm{affine\,local}} + \delta\mathbf d$ と表し、$\delta\mathbf d$ のみ予測差分符号化 (予測子: 空間近傍 3 + 時間同位置 2 + グローバル 1 の候補リスト, マージモード有り)。

**B 双予測**: $P = \frac{w_0 P_0 + w_1 P_1 + 2^{5}}{2^6}$, $w_0 + w_1 = 64$ (重みインデックス)。OBMC (`obmc_flag`) でブロック境界の予測を窓関数で重ね合わせ可。

### 7.4 アフィン/図形単位の動き

§4 の図形と §5 のブロックにはそれぞれ動きパラメータを付与できる。図形の時間予測は前フレームの対応図形 ($\mathrm{ref}(s)$ に参照フレームの図形リストを使う) からのパラメータ差分 — 図形は「動く」ことで自然に追跡される。

### 7.5 P/B 残差の摘出

$$
R = X - P \quad(P: \text{上記予測の合成})
$$

残差 $R$ はブロック (§5)・図形 (§4)・帯域画像のいずれでも表現でき、さらに次の **パラメトリック手法** で記述できる。

### 7.6 パラメトリック残差記述

帯域画像および、それを含むブロックの残差を次で記述 (複数ツールの **同時使用可**: 例 低周波は係数符号化、高周波は PNS):

1. **ノイズ置き換え** (§5.6-1 と共通)。インターではフレーム間で時間相関のあるノイズ (`noise_temporal_seed`) も選択可 — フィルムグレイン的成分。
2. **係数コピー**: 参照フレームの同位置ブロック (動き補償後) の変換係数を、バンド単位で利得付きコピー: $\hat C_b = a_b\, C^{\mathrm{ref}}_b$。
3. **並び替え**: パーティション内係数の置換 $\Pi$ を、規範定義の置換族 (巡回シフト $k$、反転、偶奇交換、ビット反転 — `perm_id` 4 ビット + パラメータ) から選ぶ。コピーと併用し、テクスチャの位相ずれを低コストで表現。
4. **周波数領域コピー**: ブロック内の低周波バンドを高周波へ複製 (帯域複製, SBR 的): $\hat C(u,v) = g\cdot \hat C(u - \Delta u, v - \Delta v)$ (範囲外 0)、$(\Delta u,\Delta v)$ とバンド利得包絡を送る。
5. **帯域画像単位の記述**: PQMF 帯域 $k$ 全体を「帯域 $k'$ のコピー × 利得包絡 (低解像度利得マップ $g_k(\mathbf x)$, 1/16 解像度、双線形補間)」として表す。

### 7.7 コピーフレームとスキップ

`frame_type=COPY` または ブロック単位 `skip_flag`: 予測をそのまま再構成とする。COPY フレームはグローバルパラメータ + オプションの帯域別ゲインマップのみで、数十バイトで符号化される。

---

## 8. 残差の階層化 (L1 / L2) とロスレス

$$
\begin{aligned}
\hat X^{(0)} &= \text{§2〜§7, §9 による非可逆再構成}\\
e^{(1)} &= X - \hat X^{(0)} &&\text{(L1 残差: 非可逆 or ロスレス)}\\
e^{(2)} &= X - \hat X^{(0)} - \hat e^{(1)} &&\text{(L2 = 最終残差: ロスレス)}
\end{aligned}
$$

- `l1_mode`: 0 = なし、1 = §5 方式 (細かい QP) で非可逆、2 = ロスレス。
- `l2_mode`: 0 = なし、1 = ロスレス。`l2_mode=1` のとき全体として完全可逆 (YCoCg-R と合わせ RGB で可逆)。
- 近可逆モード: `l2_max_err = E` で $|e^{(2)}|\le E$ となるよう $e^{(2)}$ を $\mathrm{round}(e/(2E+1))$ で量子化 (JPEG-LS near-lossless 同等)。

**ロスレス残差符号化** (整数, 画素ラスタ順):

1. 予測: MED (LOCO-I) $\hat e = \mathrm{med}(a, b, a+b-c)$ と、勾配調整予測 (CALIC GAP) を文脈で選択、さらにバイアス補正 $C[\text{ctx}]$。
2. 予測誤差 $\epsilon = e - \hat e$ を §11 の `sint` で CM 符号化。文脈: 局所活動度 $\Delta = |d-b| + |b-c| + |c-a|$ の量子化値 (8 段)、帯域番号、近傍 $|\epsilon|$、$\hat X^{(0)}$ の局所勾配 (非可逆層との相関 — **L1/L2 は再構成画像を文脈として使える**)。

---

## 9. ループフィルタ (デブロック)

3 段構成、すべて規範 (順番固定):

### 9.1 従来型デブロック

ブロック境界 (4 画素グリッド上、変換ブロック/予測ブロック境界) ごとに境界強度
$\mathrm{BS} \in\{0,1,2\}$ (イントラ/係数有無/MV 差 ≥ 1 画素で決定)。
境界をまたぐ画素 $p_3..p_0 | q_0..q_3$ に対し、

$$
d = |p_2 - 2p_1 + p_0| + |q_2 - 2q_1 + q_0| < \beta(QP)
$$

のとき適用。弱フィルタ:

$$
\Delta = \mathrm{clip}\!\left(\frac{9(q_0-p_0) - 3(q_1-p_1) + 8}{16},\ -t_c,\ t_c\right),\quad p_0' = p_0+\Delta,\ q_0' = q_0-\Delta
$$

強フィルタ (大ブロック ≥ 32 は 7 タップ長フィルタ) は HEVC/VVC 同等の係数表を規範とする。$\beta, t_c$ は QP 依存テーブル + スライスオフセット。

### 9.2 周波数領域ゲインフィルタ

再構成 $\hat X$ を重なり窓 (8×8 / 16×16、ホップ 1/2、正弦窓 $w[n]=\sin(\pi(n+\frac12)/N)$ で TDAC 的に完全再構成) で DCT し、帯域ごとのゲインを掛けて逆変換:

$$
\tilde C(u,v) = G_{c(u,v)}\left(\frac{|\hat C(u,v)|^2}{\sigma^2_{c}}\right)\hat C(u,v),\qquad
G(\xi) = \max\!\left(0,\ 1 - \frac{\mu}{\xi}\right)^{\kappa}
$$

($c(u,v)$: 周波数クラス 1〜8、$\sigma^2_c$: 量子化雑音分散推定 $\Delta_c^2/12$、$\mu,\kappa$ はフレームで送るクラス別パラメータ。Wiener 型縮小)。

### 9.3 画素単位ゲインフィルタ (パラメータ側)

フレーム/タイル単位のパラメータとして低解像度の **ゲインマップ** $\Gamma[i,j]$ (グリッド間隔 $2^s$, $s\in\{3..7\}$, Q2.6) を送り、双線形補間した画素ごとのゲイン $\gamma(\mathbf x)$ で、フィルタ前後の差分の適用量を制御する:

$$
\hat X^{\mathrm{LF}}(\mathbf x) = \hat X(\mathbf x) + \gamma(\mathbf x)\left(\mathrm{DF}(\hat X)(\mathbf x) - \hat X(\mathbf x)\right)
$$

($\mathrm{DF}$: 9.1→9.2 適用後)。符号器は各グリッドセルで $\gamma^* = \frac{\sum (X-\hat X)(\mathrm{DF}-\hat X)}{\sum(\mathrm{DF}-\hat X)^2}$ を求めて量子化。さらに **図形枠** (§4 の図形境界帯 $|d(\mathbf u)| < 2\sigma$) ごとに追加ゲイン $\gamma_s$ を持てる (`shape_edge_gain`)。

---

## 10. 辞書 (最大 65536 項目)

### 10.1 構造

```
entry {
  uint16 id;            // 0..65535
  uint8  kind;          // 0: 画素パッチ, 1: 帯域画像パッチ, 2: 変換係数パターン, 3: 図形(群), 4: 残差パッチ
  uint16 w, h;          // 4..1024
  uint8  plane_mask;    // 適用可能プレーン
  int16  data[...];     // kind に応じた値 (Q 形式は kind 依存)
  uint32 use_count, last_use; // LRU 管理 (規範: 符号器/復号器が同一に更新)
}
```

- **予約項目** `id ∈ [0, R)` (R = 4096 を規範で予約): DCT 基底 (4〜32 点の 2D 基底, 計 ~1360)、平坦/線形勾配/放射勾配、ステップエッジ (16 方向 × 4 位置)、規範擬似乱数ノイズパッチ (分布・相関長別)、ガボール (8 方向 × 4 周波数)、チェッカ/縞パターン、§4 の標準図形群。これらはビットストリームに含まず、規範の生成式で両側が生成する。
- **動的項目** `id ∈ [R, 65536)`: ビットストリームで追加・置換・削除。

### 10.2 操作

| op | 内容 |
|---|---|
| ADD_EXPLICIT | データを明示送信して登録 |
| ADD_FROM_RECON | 再構成済み領域 (フレーム/帯域/位置/サイズ) を切り出して登録 (データ送信不要) |
| UPDATE | 既存項目に差分を加算 |
| EVICT | 削除 (省略時は満杯で LRU を自動削除) |
| RESET | 動的項目を全消去 (RAP で必須) |

### 10.3 参照

ブロック・図形・残差・帯域画像のいずれも辞書項目で予測できる:

$$
\hat B_\Omega = a\cdot\mathcal W\!\left(D_{\mathrm{id}}\right) + b \quad(\mathcal W: \text{§6.1 と同じ変形})
$$

複数項目の線形結合 ($\le 4$ 項) も可: $\sum_j a_j \mathcal W_j(D_{\mathrm{id}_j})$。**残差への適用**: 予測残差 $R$ (§7.5) や L1 残差に対しても同じ参照を使える (`dict_apply_residual`)。

`id` の符号化: MRU リスト上の位置 (move-to-front 変換) を CM で符号化 — 頻出項目は数ビット。

---

## 11. エントロピー符号化 (CM + rANS)

### 11.1 方針

すべての構文要素 (フラグ、整数、格子インデックス、図形パラメータ、MV 等) を **2 値化** し、各ビットの確率をコンテキストミキシングで推定し、**2 値 rANS** で符号化する。PPMd 型の可変長文脈 (直前シンボル履歴) もミキサへの入力の一つとして扱う。

### 11.2 2 値化

- フラグ: 1 ビット。
- 非負整数 `uint(v)`: $v' = v+1$, $e = \lfloor\log_2 v'\rfloor$ を単進符号 ($e$ 個の 1 + 終端 0) で、次に $v'$ の下位 $e$ ビットを上位から。上位 3 ビットまで文脈モデル、残りはバイパス ($p=1/2$)。
- 符号付き整数 `sint(v)`: ゼロフラグ → 符号 → `uint(|v|-1)`。
- 列挙値 (分割モード等): 規範の 2 分木 (頻度順) で 2 値化。
- 各ビットは **ノード番号** (2 値化木の位置) を持ち、文脈の一部となる。

### 11.3 確率モデル (CM)

入力モデル $i=1..n$ が確率 $p_i$ を出す。各入力は文脈 (構文要素種別 $\times$ ノード $\times$ 文脈値) のハッシュでテーブル参照する **2 速度適応カウンタ**:

$$
p^{\mathrm{fast}} \leftarrow p^{\mathrm{fast}} + (b - p^{\mathrm{fast}})\,2^{-4},\quad
p^{\mathrm{slow}} \leftarrow p^{\mathrm{slow}} + (b - p^{\mathrm{slow}})\,2^{-7},\quad
p_i = \tfrac12(p^{\mathrm{fast}}+p^{\mathrm{slow}})
$$

入力の種類 (構文要素ごとに規範で列挙):

1. order-0 (ノードのみ)
2. 局所文脈 A (例: 近傍係数の大きさ, 近傍ブロックの同要素値)
3. 局所文脈 B (例: 帯域番号, バンド番号, ブロックサイズ)
4. A×B 結合
5. **PPMd 型履歴文脈**: 同じ構文要素の直前 $k$ 個 ($k=1,2,4$) のシンボル値ハッシュ (order-$k$)。最長一致文脈が存在しない場合は短い次数へエスケープ (= そのテーブルエントリが未使用なら入力値 0 = 中立)。
6. マッチモデル: 直前に同一の文脈列が出現した位置の次のビットを予測 (長い反復パターン用)。

**ミキサ** (ロジスティック混合, 1 層; 重み集合は小さな文脈で選択):

$$
\mathrm{st}(p) = \ln\frac{p}{1-p},\qquad
p_{\mathrm{mix}} = \sigma\Big(\sum_i w_i\,\mathrm{st}(p_i)\Big),\qquad
w_i \leftarrow w_i + \eta\,(b - p_{\mathrm{mix}})\,\mathrm{st}(p_i)
$$

**APM/SSE** (2 次推定): $\mathrm{st}(p_{\mathrm{mix}})$ を 33 区間に分けた補間テーブル (ノード文脈別) で補正し、$p = (p_{\mathrm{mix}} + 3p_{\mathrm{apm}})/4$。

規範: 全演算は整数 (stretch は 4096 エントリ表、squash は $\pm2047$ 範囲の表; 確率 12 ビット, $p\in[1,4095]$)。リファレンス `fvc::CMModel` 参照。

### 11.4 rANS (2 値, 規範)

状態 $x\in[L, 2^8L)$, $L = 2^{23}$, 確率精度 $2^{12}$、バイト単位正規化。ビット $b$、確率 $p_1$ について $(\mathrm{start}, f) = b\ ?\ (0, p_1) : (p_1, 4096 - p_1)$。

**符号化** (シンボルを **逆順** に処理):

$$
\text{while } x \ge \left(\tfrac{L}{2^{12}}\cdot 2^8\right)f:\ \text{emit}(x \bmod 256),\ x \leftarrow \lfloor x/256\rfloor
$$
$$
x \leftarrow \left\lfloor \frac{x}{f}\right\rfloor 2^{12} + (x \bmod f) + \mathrm{start}
$$

終了時に $x$ を 4 バイト出力し、出力バイト列全体を反転 (復号順にする)。

**復号**: $x$ を 4 バイト (ビッグエンディアン) で初期化。各ビット:

$$
s = x \bmod 2^{12},\quad b = [s < p_1],\quad x \leftarrow f\cdot\lfloor x/2^{12}\rfloor + s - \mathrm{start},\quad
\text{while } x<L:\ x \leftarrow 256x + \text{read}()
$$

CM は適応的なので、**符号器は順方向に CM を回して $(b, p_1)$ 列を記録し、ストリーム (タイル) 終端で逆順に rANS 符号化** する。復号器は順方向に CM と rANS を同時に回す (`RansBitEncoder::finish`)。

**並列化**: タイルごとに独立 rANS ストリーム。さらにタイル内で `num_interleaved_streams ∈{1,2,4,8}` の rANS インタリーブ (構文要素クラス別: 係数 / 図形 / MV / その他) を許す。

### 11.5 文脈リセット

CM 状態はタイル先頭で初期化。`cm_inherit=1` の場合、同じ参照構造の直前フレームのタイル終了時状態を継承 (RAP では禁止)。

---

## 12. ビットストリーム構文

### 12.1 ユニット構造

```
Bitstream := { Unit }
Unit      := unit_header payload
unit_header:
  u(8)   unit_type   // 0:SEQ 1:FRAME 2:TILE 3:DICT 4:SEI 5:EOS
  u(32)  payload_size
```

`payload_size` によりユニット単位でスキップ可能。TILE のペイロードは rANS バイト列そのもの。

### 12.2 シーケンスヘッダ (SEQ, 固定長)

| 要素 | 記述子 | 意味 |
|---|---|---|
| magic | u(32) | `'F','V','C','1'` |
| profile, level | u(8), u(8) | |
| width_minus1, height_minus1 | u(16), u(16) | |
| bit_depth_minus8 | u(4) | 8〜16 bit |
| chroma_format | u(2) | 0:400 1:420 2:422 3:444 |
| color_transform | u(2) | §2.1 |
| ctu_log2 | u(4) | 5..10 (32..1024) |
| max_bands_log2 | u(4) | 0..8 |
| pqmf_filter_set_mask | u(8) | 使用可能フィルタ |
| dict_reserved_enable | u(1) | |
| max_ref_frames | u(5) | |
| frame_rate_num/den | u(32), u(32) | |
| tool_enable_flags | u(32) | 各ツールの有効化 (図形, TNS, FIR 動き, …) |

### 12.3 フレームヘッダ (FRAME, ae(v))

```
frame_header() {
  frame_type                      ae(v)  // I/P/B/COPY
  poc_delta                       ae(v)
  if (frame_type != I) {
    num_ref_l0, (B) num_ref_l1    ae(v)
    ref_idx_list[]                ae(v)
    global_motion_model           ae(v)  // 0:none 1:pan 2:sim 3:affine 4:proj
    global_motion_params[]        ae(v)  // 予測: 前フレームのパラメータ
    for (p in planes) { gain_q[p], offset_q[p] } ae(v)
    band_gain_flag                ae(v)
  }
  if (frame_type == COPY) { copy_band_gain_map (opt); return }
  qp_base                         ae(v)
  for (p in planes) {
    chroma_pred_mode_default      ae(v)
    pqmf_filter_id, log2_Mx, log2_My   ae(v)
    band_split_tree()             ae(v)
    pqmf_residual_mode            ae(v)
  }
  l1_mode, l2_mode, l2_max_err    ae(v)
  loop_filter_params()            ae(v)  // β/tc offset, §9.2 (μ,κ)[8], §9.3 gain map
  tile_layout()                   ae(v)
}
```

### 12.4 タイル (TILE)

```
tile_data() {
  for (p in planes) for (k in bands(p)) {
    band_header(k)                // band_ref_mode, a_kj, b_kj, band_qp_delta, fill_mode
    for (ctu in tile) {
      shape_list(ctu, k)          // §4: num_shapes, {ref_idx, copy_flags, δΘ}
      coding_tree(ctu, k)         // §5.1 split_mode 再帰
    }
  }
  pqmf_residual()                 // §3.4
  l1_residual(), l2_residual()    // §8
}

coding_unit() {
  pred_mode                       // INTRA_COPY / INTER / DICT / PARAM_ONLY / SKIP
  pred_params()                   // §6.1 / §7.3 / §10.3
  chroma_pred_mode, alpha_q, beta_delta     // Co,Cg のみ
  cbf
  if (cbf) {
    tx_h, tx_v, tns_params()
    band_layout_id, part_len, join_group_id[]
    for (band b) {
      fill_mode[b]
      if (fill_mode[b] == CODED) {
        qmode[b], prenl[b]
        gain_q[b] (delta)
        for (partition p) lattice_index / sq_levels / rp_levels
      } else fill_params[b]
    }
  }
}
```

### 12.5 辞書ユニット (DICT)

`num_ops`, 各 `{op, id, kind, w, h, (data | recon_ref)}`。フレームより前に置き、そのフレームから有効。

---

## 13. 符号器制御と実装詳細

### 13.1 RD 最適化 (informative)

$\lambda = c\cdot 2^{(QP-12)/3}$ ($c\approx0.57$ I, 0.68 P/B)。ツール選択の探索順 (高速化):

1. グローバル動き/ゲイン推定 → COPY フレーム判定 ($J_{\mathrm{COPY}}$ が閾値以下なら確定)
2. PQMF フィルタ選択 (§3.2: 縮小画像で候補評価 → 上位 2 候補を本評価)
3. CTU ごと: 図形当てはめ (残差エネルギー上位領域のみ) → ブロック分割 (トップダウン早期打ち切り + ボトムアップ併合)
4. 葉ごと: 予測モード → 変換 → バンドごとの qmode/fill_mode を貪欲選択
5. RDOQ: 格子 VQ では $\mathcal{Q}_\Lambda$ の近傍 (Voronoi 隣接 ≤ 240 点 for $E_8$) からレート込みで最良点を選ぶ (トレリスは不要)
6. ループフィルタパラメータは最終再構成で最小二乗推定

レート推定は CM の現在状態から $-\log_2 p$ を積算 (正確・適応的)。

### 13.2 歪み尺度

既定 SSE (YCoCg 各プレーン重み 1 : 0.5 : 0.5)。オプションで帯域別知覚重み $w_k$、SSIM 近似 (局所分散比でスケール)。

### 13.3 固定小数点化 (規範の決定性)

| 処理 | 形式 |
|---|---|
| PQMF 係数 | Q1.30 (int32)、積和は int64、帯域サンプルは Q(b+4).8 |
| DCT/DST | 整数行列 (スケール $2^{6+\log_2 N/2}$)、段間シフトで 16bit 中間精度 |
| 図形描画 | 座標 Q16.16、sigmoid/exp LUT 1024 |
| 非線形前変換 | 4096 エントリ LUT + 線形補間 |
| ランダム投影 | $\pm1$ 行列 (スケールは最後に一括)、$(PP^\top)^{-1}$ は Cholesky を Q16.16 固定小数で (規範アルゴリズム) |
| 格子 VQ | 座標は整数 ×2 表現 ($E_8$ の半整数対応) |
| CM / rANS | 全整数 (§11) |

リファレンス実装 (本リポジトリ) は現段階では double 版。規範固定小数点版は同一 API で差し替える (§14.2)。

### 13.4 計算量の目安

PQMF $M=16, m=6$: 1 画素あたり $2\times 2mM = 384$ MAC (分離 2 次元, ポリフェーズ+DCT-IV 高速化で ≈ $2(2m + \log_2 M)\approx 32$ MAC)。実装ではポリフェーズ分解 $h_k[n] \to$ $2M$ 個のポリフェーズ成分 + 長さ $M$ の DCT-IV に置き換えること。

---

## 14. C++ 実装構成

### 14.1 リポジトリにあるリファレンス実装 (動作確認済み)

| ファイル | 内容 | 対応章 |
|---|---|---|
| `include/fvc/color.hpp`, `src/color.cpp` | YCoCg-R, CfL 色予測 (当てはめ/適用) | §2 |
| `include/fvc/pqmf.hpp`, `src/pqmf.cpp` | PQMF 原型設計 (カイザー + 黄金分割)、1D/2D 解析・合成 (最大 256 帯域)、PSNR | §3 |
| `include/fvc/quant.hpp`, `src/quant.cpp` | $\mathbb Z^n/D_n/E_8$ 最近傍点・インデックス化、非線形 SQ、ランダム投影 (最小ノルム再構成)、非線形前変換、SplitMix64 | §5.5, §5.7 |
| `include/fvc/transform.hpp`, `src/transform.cpp` | DCT-II/DST-VII/恒等 (任意長)、2D 分離変換、TNS (Levinson + arcsin 量子化 + 合成) | §5.2, §6.2 |
| `include/fvc/entropy.hpp`, `src/entropy.cpp` | 2 値 rANS、stretch/squash、CM (4 入力 + バイアス, ミキサ, APM)、uint/sint 2 値化 | §11 |
| `tests/test_main.cpp` | 全部品の往復/精度テスト | — |

ビルド・テスト:

```sh
cmake -S . -B build && cmake --build build -j && ./build/fvc_tests
```

実測結果 (リファレンス):

```
color: ok (CfL alpha_q=48 max|res|=0)
pqmf:  2x2  =   4 bands, PSNR 70.1 dB
pqmf:  4x4  =  16 bands, PSNR 67.8 dB
pqmf:  8x8  =  64 bands, PSNR 71.5 dB
pqmf: 16x16 = 256 bands, PSNR 71.7 dB
lattice Z8: G=0.0833   lattice D8: G=0.0759   lattice E8: G=0.0719
transform: ok (TNS order=6, pred gain 4.6 dB)
entropy: 50002 symbols -> 21420 bytes (3.427 bits/sym), roundtrip ok
```

### 14.2 全体のクラス設計 (今後の実装)

```cpp
namespace fvc {

struct Plane { int w, h; std::vector<int32_t> px; };          // 整数画素
struct BandImage { int plane, kx, ky, w, h; std::vector<int32_t> v; };

struct Shape {                       // §4
  uint8_t type; double theta, l1, l2, kappa; double tx, ty;
  std::array<double,6> amp; double sigma; uint8_t blend;
  int32_t ref_idx; uint8_t copy_flags;
};

struct CodingUnit {                  // §5, §6, §7
  int x, y, w, h;
  enum class Pred : uint8_t { IntraCopy, Inter, Dict, ParamOnly, Skip } pred;
  // 予測パラメータ
  int16_t mv_x, mv_y; std::array<int16_t,4> fir_psi; uint8_t ref_idx;
  uint16_t dict_id; int16_t a_q, b;
  // 変換・量子化
  TxType tx_h, tx_v; TnsFilter tns;
  std::vector<BandCoding> bands;     // qmode, prenl, gain_q, levels / fill_params
};

class SyntaxCoder {                  // 符号器/復号器で同一コードを共有するテンプレート I/F
 public:
  template <class IO> void code_cu(IO& io, CodingUnit& cu, const Context& ctx);
  // IO = EntropyWriter → 書き込み, EntropyReader → 読み込み (値を参照で更新)
};

class Encoder { public: std::vector<uint8_t> encode_frame(const Frame&); };
class Decoder { public: Frame decode_frame(const uint8_t*, size_t); };
}
```

**符号器/復号器の構文共有**: 構文要素の読み書きを `template<class IO> void code(IO&, T& value, ctx)` で一本化する (Writer は値を符号化、Reader は値を書き戻す)。これにより文脈導出の不一致 (最も多いバグ) を構造的に防ぐ。

### 14.3 推奨実装順序

1. ✅ エントロピー (CM+rANS)、色変換、変換、量子化、PQMF (本リポジトリ)
2. ビットストリーム I/O (ユニット、SEQ/FRAME ヘッダ)、`SyntaxCoder` テンプレート
3. I フレーム最小構成: YCoCg → (M=1) → 固定 QT 分割 → DCT → 非線形 SQ → CM → 復号一致テスト
4. PQMF 帯域化 + 合成残差 + L2 ロスレス → **完全可逆モード** 達成 (検証が容易)
5. 格子 VQ / 利得形状 / バンド分割 / パラメトリック補完 / TNS
6. イントラコピー, 色予測, 帯域間差分
7. P/B: ブロック MC → FIR 動き → グローバル動き/ゲイン → COPY フレーム
8. 図形表現 (§4)、辞書 (§10)
9. ループフィルタ 3 段
10. 固定小数点化 (§13.3) とクロスプラットフォーム一致テスト、SIMD 化

各段階で「符号器出力 → 復号器 → 符号器内部再構成とビット一致」の回帰テストを必須とする。

---

## 付録 A. 主要パラメータの値域一覧

| パラメータ | 値域 | 量子化 |
|---|---|---|
| 帯域数 | $M_x,M_y\in\{1,2,4,8,16\}$ (≤256) | — |
| ブロックサイズ | 4〜1024 (2 冪 + 4 の倍数任意長方形) | — |
| QP | 0〜63 (1/4 刻み拡張で 0〜255) | — |
| CfL $\alpha$ | $[-2, 2)$ | 1/64 |
| 帯域間予測 $a_{kj}$ | $[-8, 8)$ | 1/256 |
| 図形 θ / ℓ / κ / t | $[0,2\pi)$ / $[-2, 10]$ / $[-2,2]$ / 画像内 | $2\pi/256$ / 1/16 / 1/32 / 1/8 px |
| MV | ±8192 px | 1/16 px |
| FIR 動き基底係数 $\psi_j$ | $[-1,1)$ | 1/128 |
| TNS 次数 / PARCOR | ≤8 / $(-1,1)$ | arcsin 4 bit |
| 非線形 SQ γ / δ | {1,.875,.75,.625,.5} / $[-0.5,0]$ | — / 1/16 |
| ランダム投影比 $m/n$ | {1/8,1/4,3/8,1/2,3/4} | — |
| 辞書 ID | 0〜65535 (0〜4095 予約) | — |
| ゲインマップ γ | $[0, 4)$ | Q2.6 |

## 付録 B. 用語

- **帯域画像**: PQMF 分解で得られる各サブバンドの 2D 配列。
- **バンド**: 変換係数領域の周波数区分 (帯域画像と区別)。
- **パーティション**: バンド内のベクトル量子化単位。
- **L1/L2 残差**: §8 の階層化残差 (L2 は最終, ロスレス)。
- **PNS**: Perceptual Noise Substitution (ノイズ置き換え)。
- **TNS**: Temporal (ここでは空間) Noise Shaping、周波数方向 LPC による量子化雑音整形。
