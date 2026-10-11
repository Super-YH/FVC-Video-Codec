# FVC TS 001 V0.9.0 (2026-10)

**Technical Specification**

**FVC video coding; Bitstream syntax, decoding process and encoder guidelines**

> This document is a project document of FVC written in the clause structure and style of an ETSI Technical Specification (TS). It is not a document published by ETSI.
> Reference implementation: repository `Super-YH/FVC-Video-Codec`, branch `claude/ecstatic-shannon-p0n2hk`, commit `e822af4`.
> Companion Japanese edition: `docs/FVC_TS_ja.md`. Where the two differ, the behaviour of the reference implementation (`src/`) prevails.

---

## Contents

- Foreword
- Modal verbs terminology
- Introduction
- 1 Scope
- 2 References
- 3 Definitions, symbols and abbreviations
- 4 Overview
- 5 Conventions
- 6 Bitstream structure
- 7 Entropy coding
- 8 Frame decoding process
- 9 Block layer
- 10 Prediction
- 11 Transform and quantization
- 12 Band (PQMF) coding
- 13 Shape layer and dictionary
- 14 In-loop filters
- 15 Lossless layers (L1/L2)
- Annex A (normative): Value ranges and constraints
- Annex B (informative): Encoder guidelines and presets
- Annex C (informative): Performance results
- Annex D (informative): Implementation status and known limitations
- History

---

## Foreword

This Technical Specification (TS) has been produced by the FVC project. It specifies the syntax and semantics of the FVC coded bitstream (hereafter "the scheme") and its normative decoding process, and gives encoder guidelines and evaluation results for information.

Version V0.9.0 indicates that the reference implementation implements the complete normative decoding process, but that some processes (Annex D) still rely on floating-point arithmetic, so bit-exactness across independent implementations is not yet guaranteed.

## Modal verbs terminology

In the present document, the following expressions are used in line with the ETSI Drafting Rules.

| Expression | Meaning |
|---|---|
| shall | Requirement; mandatory for conformance |
| shall not | Prohibition |
| should | Recommendation |
| may | Permission |
| can | Possibility or capability |

## Introduction

The scheme is built around block transform coding and integrates the following elements into a single bitstream:

- the reversible YCoCg-R colour transform and CfL-style chroma prediction;
- cosine-modulated PQMF band decomposition into up to 256 bands, with block coding of the band images;
- a layer of superposed affine shapes (Gaussian, ellipse, rectangle, triangle);
- quadtree partitioning from 4 to 64 samples, with DCT-II / DST-VII / DCT-VIII / identity transforms;
- dead-zone scalar quantization, E8 lattice VQ, and gain–shape coding (including nonlinear scalar quantization, pre-transforms and random projection);
- TNS, noise filling and parametric band filling;
- I / P / B / COPY frames, global motion models (translation to projective), per-band gains, FIR motion and OBMC;
- dictionary prediction with up to 65536 entries;
- deblocking, a frequency-domain gain filter, a pixel gain map, directional deringing (CDEF-like) and an adaptive loop filter (ALF);
- lossless coding by L1 (lossy) / L2 (residual) layering;
- a context-mixing (CM) probability model with a binary rANS arithmetic coder.

---

## 1 Scope

The present document specifies, for the bitstream of the scheme:

- the unit structure and the syntax and semantics of the sequence header and the frame header (clause 6);
- entropy coding (clause 7);
- the decoding processes for frames, blocks, prediction, transform, quantization, bands, shapes, in-loop filters and lossless layers (clauses 8 to 15).

Encoder behaviour is not specified. Annex B gives encoder guidelines for information.
A conforming decoder shall decode every bitstream that conforms to the present document into reconstructed pictures identical to those of the reference implementation (see the limitations in Annex D).

## 2 References

### 2.1 Normative references

None. The present document is self-contained.

### 2.2 Informative references

- [i.1] Recommendation ITU-T H.265 | ISO/IEC 23008-2 (HEVC): source of the intra angular mode numbering and of the deblocking threshold tables β and tC.
- [i.2] Recommendation ITU-T H.266 | ISO/IEC 23090-3 (VVC): long-tap deblocking, MPM and MTS concepts.
- [i.3] AOMedia AV1 Bitstream & Decoding Process Specification: Smooth/Paeth prediction, CDEF and OBMC concepts.
- [i.4] J. H. Conway, N. J. A. Sloane: "Sphere Packings, Lattices and Groups": nearest-point search for the E8 lattice.
- [i.5] J. Duda: "Asymmetric numeral systems": rANS.
- [i.6] ETSI EG 201 788 and the ETSI Drafting Rules: reference for the style of the present document.

## 3 Definitions, symbols and abbreviations

### 3.1 Definitions

- **unit:** portion of the bitstream delimited by a type and a byte length.
- **CTU:** coding tree unit of 64×64 samples (64×64 chroma samples in a chroma plane).
- **leaf:** terminal block of the quadtree; the unit of prediction and residual coding.
- **valid region:** the actual plane size w×h, i.e. excluding the padding up to a multiple of the CTU size.
- **band image:** two-dimensional decimated signal of band k produced by PQMF analysis.
- **main stream:** the single rANS code string of a frame; it carries frame-level and plane-level elements.
- **tile stream:** the rANS code string of one tile; it carries the block layer.
- **POC:** picture order count (display order).

### 3.2 Symbols

| Symbol | Meaning |
|---|---|
| Δ(q) | quantization step for quantization parameter q |
| λ | Lagrange multiplier (encoder only) |
| s, l | block side length in samples, s = 2^l |
| u, v | horizontal and vertical frequency indices of transform coefficients |
| Clip3(a,b,x) | min(max(x,a),b) |
| ≫ | arithmetic right shift (rounds towards −∞ for negative values) |

### 3.3 Abbreviations

| Abbreviation | Expansion |
|---|---|
| ALF | Adaptive Loop Filter |
| APM | Adaptive Probability Map |
| BD | Bjøntegaard Delta |
| CDEF | Constrained Directional Enhancement Filter |
| CfL | Chroma from Luma |
| CM | Context Mixing |
| DCT/DST | Discrete Cosine / Sine Transform |
| IBC | Intra Block Copy |
| MPM | Most Probable Mode |
| MTS | Multiple Transform Selection |
| OBMC | Overlapped Block Motion Compensation |
| PNS | Perceptual Noise Substitution |
| PQMF | Pseudo-Quadrature Mirror Filter bank |
| rANS | range Asymmetric Numeral Systems |
| RD | Rate–Distortion |
| RDOQ | Rate–Distortion Optimised Quantisation |
| TNS | Temporal Noise Shaping (applied here along the coefficient scan) |
| VQ | Vector Quantisation |

---

## 4 Overview

### 4.1 Decoding flow

1. Read the sequence header (SEQ) to obtain the picture format.
2. For each frame unit, extract the frame header, the main stream and the tile streams.
3. Select the reference pictures from the DPB according to the frame type.
4. Decode global motion, gains and related parameters from the main stream (inter frames only).
5. For each plane (Y/Co/Cg or Y/U/V):
   1. the shape layer (when enabled);
   2. the selection between band coding and pixel block coding, and the decoding of the selected method;
   3. in-loop filters (deblocking → frequency gain → gain map → CDEF → ALF);
   4. the lossless layer (L2, or fully lossless).
6. Store the reconstructed frame in the DPB and output frames in display order.

### 4.2 Colour space

When `color_transform` in the sequence header is 0, the three decoded planes are YCoCg-R and shall be inverse-transformed for output:

```
t  = Y − (Cg ≫ 1)
G  = Cg + t
B  = t − (Co ≫ 1)
R  = B + Co
```

When `color_transform` is 1, the planes are output unchanged.

---

## 5 Conventions

- Multi-byte fixed-length integers are big-endian.
- `u(n)` denotes an n-bit unsigned fixed-length integer. `uv` denotes a variable-length integer in LEB128 form: 7 bits per byte, least significant group first, most significant bit of each byte is a continuation flag, at most 5 bytes.
- `ae(v)` denotes a syntax element coded by the entropy coder of clause 7.

---

## 6 Bitstream structure

### 6.1 Units

| Syntax | Descriptor |
|---|---|
| unit_type | u(8) |
| unit_length | u(32) |
| unit_payload | unit_length bytes |

| unit_type | Name | Content |
|---|---|---|
| 0 | SEQ | sequence header |
| 1 | FRAME | one frame |
| 2 | TILE | reserved |
| 3 | DICT | reserved |
| 4 | SEI | reserved (decoders shall skip it) |
| 5 | EOS | end of sequence (payload length 0) |

A stream shall begin with a SEQ unit. Decoders shall skip units of unknown type. Because units are delimited, the scheme can be decoded incrementally (streaming).

### 6.2 Sequence header (SEQ)

| Syntax | Descriptor | Semantics |
|---|---|---|
| magic | u(32) | shall be 0x46564331 ('FVC1') |
| profile | u(8) | 2 |
| width | u(16) | luma width (≥ 1) |
| height | u(16) | luma height (≥ 1) |
| bit_depth | u(8) | 8 to 16 |
| chroma_format | u(8) | 0: 4:4:4, 1: 4:2:0 |
| color_transform | u(8) | 0: YCoCg-R, 1: identity |
| fps_num | uv | frame rate numerator |
| fps_den | uv | frame rate denominator |

For 4:2:0, the chroma dimensions are ⌈width/2⌉ × ⌈height/2⌉.

### 6.3 Frame payload

| Syntax | Descriptor |
|---|---|
| frame_header() | 6.4 |
| main_length | uv |
| main_stream | main_length bytes |
| num_tiles | uv (≤ 4096) |
| for (i = 0; i < num_tiles; i++) { | |
| &nbsp;&nbsp;tile_length (only if i < num_tiles−1) | uv |
| &nbsp;&nbsp;tile_stream[i] | tile_length bytes (the last tile takes the remainder of the payload) |
| } | |

### 6.4 Frame header

| Syntax | Descriptor | Semantics |
|---|---|---|
| frame_type | u(8) | 0: I, 1: P, 2: B, 3: COPY |
| poc | uv | picture order count |
| qp | u(8) | 0 to 63 |
| flags0 | u(8) | bit0 l2, bit1 lossy, bits2–4 pqmf_log2, bits5–6 pqmf_filter |
| pqmf_log2y | u(8) | log2 of the number of vertical bands (0: same as pqmf_log2) |
| log2_sizes | u(8) | bits0–3 min_log2 (≥ 2), bits4–7 max_log2 (≤ 6) |
| tools0 | u(8) | bit0 ibc, 1 tns, 2 e8, 3 cfl, 4 nf, 5 all_angular, 6 dict, 7 fir |
| flags1 | u(8) | bit0 shapes, 1 band_tools, 2 dict_reset, 3 lf, 4 lf_freq, 5 lf_map, 6 rect, 7 tmvp |
| tiles | u(8) | bits0–3 tile_cols−1, bits4–7 tile_rows−1 |
| chroma_qp | u(8) | bits0–6 cqp_off+32 (−12 to 12), bit7 aqp |
| flags2 | u(8) | bit0 alf, 1 mts, 2 band_blocks, 3 band_ns, 4 inter_ns, 5 cdef, 6 gs, 7 firb |
| for (l = 0; l < 2; l++) { num_ref[l] u(8) (≤ 4); ref_poc[l][i] uv } | | reference lists L0, L1 |

Constraints: an I frame shall have no references. A P frame shall have at least one L0 reference; a B frame shall have at least one reference in each of L0 and L1. Every referenced POC shall be present in the DPB (at most 12 pictures).

When `lossy` = 0 and `l2` = 0, the frame is coded fully losslessly (clause 15).

---

## 7 Entropy coding

### 7.1 Binary rANS

- Probabilities have 12-bit precision (P1 ∈ [1, 4095], scale 4096).
- The state x is normalised to [2^23, 2^31) with byte-wise renormalisation.
- Encoding processes the symbols in reverse order and finally writes the state as 4 bytes. The code string is the byte-reversed output.
- The decoder shall initialise x from the first 4 bytes (big-endian) and decode each binary symbol as follows:

```
s = x mod 4096
if s < P1: bit = 1; x = P1·(x ≫ 12) + s
else:      bit = 0; x = (4096 − P1)·(x ≫ 12) + s − P1
while x < 2^23: x = (x ≪ 8) | next_byte()   (0 beyond the end)
```

### 7.2 Context-mixing model (CM)

Each group of syntax elements has its own model. A binary symbol is conditioned on the triple (node, a, b).

1. **Inputs:** four hash tables of 2^14 entries each are indexed by
   - table 0: hash(node, 0, 0)
   - table 1: hash(node, a, 1)
   - table 2: hash(node, b, 2)
   - table 3: hash(node, a, b ⊕ 0xABCD)

   Each entry is a two-rate probability counter (16 bits, update rates 1/16 and 1/128), whose average gives a 12-bit probability.
2. **Mixing:** each probability is mapped to the stretch domain. Together with a bias input of 256, the five inputs are combined linearly with the weights of context (node·31 + (a & 31)) mod 1024 and mapped back by squash to p_mix. Weights are initialised to 2^16/4 and updated as w_i += (st_i · (bit·4096 − p_mix)·6) ≫ 10.
3. **APM:** p_a is obtained from a 33-bin interpolated table of stretch(p_mix) (context node mod 1024), and P1 = Clip3(1, 4095, (p_mix + 3·p_a) ≫ 2). The table adapts at rate 1/64.
4. Model tables are initialised on first use.

### 7.3 Binarization

| Name | Definition |
|---|---|
| bit | one binary symbol |
| uint(v) | the exponent e of v+1 is coded in unary on nodes 2..(2+e); the top 3 mantissa bits are coded on nodes 64+32e+i and the remaining bits are bypass-coded (equiprobable) |
| uintc(v) | same exponent part as uint; all mantissa bits are context-coded (top 5 bits on binary-tree nodes 3000+64e+tree, the rest on nodes 5000+32e+k) |
| sint(v) | zero flag on node 0, sign on node 1, then uint(|v|−1) |

---

## 8 Frame decoding process

### 8.1 References and DPB

The DPB holds up to 12 pictures. Each picture keeps its reconstruction, its luma motion field and its reference POCs. When an I frame is decoded, the dynamic part of the dictionary is reset (`dict_reset`).

### 8.2 Global parameters (inter frames, main stream)

The following are decoded in order:

1. **Global model:** `gm_type` = uint (0 translation, 1 similarity, 2 affine, 3 projective). For type ≥ 1: a, b (sint, Q16); for type 1, d = a and c = −b. For type ≥ 2: c, d. For type 3: h31, h32 (sint, Q24). The absolute value of each parameter shall not exceed 32767.
2. **Translation** t_x, t_y (sint, quarter samples, value at the picture centre).
3. For each plane p: gain g_p (64 + sint, Q6, 0 to 256), offset o_p (sint), high-band gain h_p (64 + sint, Q6, 0 to 128).

The motion vector at position (x, y) is defined relative to the picture centre (c_x, c_y) = (width/2, height/2), with u = x − c_x and v = y − c_y:

```
D  = 1 + h31·u/2^24 + h32·v/2^24          (type 3; otherwise 1)
x' = ((1 + a/2^16)·u + (b/2^16)·v + t_x/4) / D
y' = ((c/2^16)·u + (1 + d/2^16)·v + t_y/4) / D
mv = round(4·(x' − u), 4·(y' − v))        (quarter samples)
```

For type 0, mv = (t_x, t_y).

When h_p ≠ 64, plane p of the first L0 reference is replaced for prediction by the following, where L(·) is the 3×3 binomial filter (weights 1-2-1, edge replication, +8 ≫ 4):

```
ref'(x) = Clip(L(ref)(x) + ((h_p · (ref(x) − L(ref)(x)) + 32) ≫ 6))
```

The gain g_p/64 and offset o_p are also applied to the first L0 reference.

### 8.3 COPY frames

A COPY frame has no residual; each plane is the global prediction itself. For gm_type = 0, the whole plane is compensated with (t_x, t_y). Otherwise each 8×8 luma block (the corresponding size in chroma) is compensated with the mv at the block centre (x+4, y+4).

### 8.4 Plane processing order

Planes 0, 1 and 2 are processed in this order. Each plane is reconstructed in a working plane W×H padded to a multiple of the CTU size (64) and cropped to w×h for output. Reference pictures keep the cropped planes; references outside them are filled with the nearest edge sample.

---

## 9 Block layer

### 9.1 Tiles and CTUs

A plane is divided evenly in CTU units into tile_cols × tile_rows tiles. Each tile has an independent tile stream, and prediction references are restricted to the tile. CTUs within a tile are processed in raster order, and blocks within a CTU in Z (Morton) order.

### 9.2 Quadtree

A CTU is divided into root blocks of size max_log2, and each block is split into four by a split flag (bit). Blocks of size min_log2 carry no flag.
**A block lying entirely outside the valid region is not coded and carries no syntax.** Its luma motion information is set to the default (intra). Blocks straddling the valid region are coded normally, and their reconstruction outside the region is not displayed. After the block layer, every sample outside the valid region shall be replaced by a copy of the nearest sample inside the region.

Availability: position (p_x, p_y) may be referenced only if it is in the same tile and in the valid region, and precedes the current block in CTU raster order and in Morton order within the CTU.

### 9.3 Leaf syntax (outline)

A leaf carries the following syntax elements in order:

1. In inter frames, `is_inter`.
2. If inter:
   - (luma, rect enabled, s ≥ 8) partition shape 0: 2N×2N, 1: 2N×N, 2: N×2N;
   - motion per partition: merge flag and merge index (truncated unary, up to 5 candidates), or explicit motion (in B frames: direction, reference index, MVD);
   - (firb enabled, s ≥ 16) FIR motion basis flag and ψ0..3 (sint, |ψ| ≤ 16); and, for s ≤ 32, the coefficient band gain flag and a0..2 (8 + sint, 0 to 16).
3. Inside the band method: band/temporal prediction (pt = 4), band source and gain.
4. If the dictionary is enabled and s ∈ {8, 16}: dictionary flag, index and gain.
5. Otherwise (intra): IBC flag and block vector, or the intra mode (9.4); for chroma, the CfL α.
6. `cbf`. When 0 and the conditions hold, the noise substitution level (0 to 7).
7. When `cbf` = 1: MTS index (luma, s ≤ 32), quantization method (`qmode`: 0 scalar, 1 E8, 2 gain–shape), TNS (s ≥ 8 and qmode ≠ 2), coefficients (11.4), and the noise-fill level (0 to 3).

### 9.4 Intra modes

Mode numbers: 0 DC, 1 Planar, 2–34 angular (HEVC numbering), 35 CfL (chroma only), 36 Smooth, 37 Paeth. When all angular modes are enabled, a 6-candidate MPM list is used:

```
MPM = first 6 distinct entries of [left, above, Planar, DC, left±1, above±1 (if angular), vertical, horizontal, 2, 18, Smooth, Paeth]
```

When `is_mpm` is 1, the candidate index is coded in truncated unary; otherwise the remaining 31 modes are coded with a 5-bit binary tree. Reference samples come from above, left, above-right and below-left, with availability as in 9.2.

---

## 10 Prediction

### 10.1 Intra

- Angular prediction uses HEVC-style 1/32-sample interpolation, with smoothing applied to the reference samples.
- Smooth interpolates in two dimensions from the top and left edges; Paeth selects the nearest of left, above and above-left.
- CfL: chroma prediction = DC + α·(downscaled luma − mean).

### 10.2 Inter

- Motion vectors are in quarter luma samples. Luma uses the 8-tap DCT-IF; 4:2:0 chroma uses 4-tap interpolation at 1/8-sample precision.
- When ψ > 0, a 3×3 separable smoothing [a, 1−2a, a] with a = ψ/16 is applied after compensation.
- Bi-prediction averages the two predictions.
- Merge candidates: left, above, temporal (co-located motion scaled by POC distance), above-left, global (gm mv at the block centre), zero.
- MV predictor order: left, above, temporal, global (gm mv at the block position).
- Chroma motion is not transmitted; it is derived from the luma 4×4 motion field. Where luma is intra, inter motion is borrowed from a neighbour within 16 samples.
- **FIR motion basis** (firb): P′ = P + (ψ0·∂x P + ψ1·∂y P + ψ2·∂xx P + ψ3·∂yy P)/16, with differences computed inside the block using edge replication.
- **Coefficient band gains** (spec 7.6-2): the DCT coefficients of P are split into three bands (u+v < s/4, < s/2, otherwise), scaled by a_b/8 and inverse-transformed.
- **OBMC:** when the motion of a coded inter neighbour above or to the left differs from that of the current block, the prediction with the neighbour's motion is blended over ov = min(s/2, 16) samples from the boundary, with the following weights (out of 64):

| ov | Neighbour weight w_i (i = distance from the boundary) |
|---|---|
| 4 | 31, 22, 10, 1 |
| 8 | 32, 29, 25, 19, 13, 7, 3, 0 |
| 16 | 32, 31, 30, 28, 26, 24, 21, 18, 14, 11, 8, 6, 4, 2, 1, 0 |

  P ← (P·(64 − w_i) + P_nb·w_i + 32) ≫ 6

### 10.3 IBC and dictionary

- IBC: block copy from the already-coded area of the same frame. The block vector is sent as a difference from its predictor.
- Dictionary: prediction = DC prediction + (g · D_id + 8) ≫ 4, with |g| ≤ 64. The dictionary holds up to 65536 entries (4096 reserved). At the end of each frame, reconstructed blocks chosen by the encoder are added with ADD_FROM_RECON (first-in first-out replacement).

---

## 11 Transform and quantization

### 11.1 Transforms

Orthonormal DCT-II, DST-VII, DCT-VIII and identity transforms are used. The (horizontal, vertical) pair for MTS index m is given by the following table (m = 0 is DCT-II):

| m | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| horizontal | DST7 | DCT8 | DST7 | DCT8 | DST7 | DCT2 | IDTX | DCT2 |
| vertical | DST7 | DST7 | DCT8 | DCT8 | DCT2 | DST7 | IDTX | DCT8 |

m = 5 to 8 are used only for s ≤ 16.

**The inverse transform shall be computed in integer arithmetic.** Coefficients are rounded to Q8 integers (×256), and each basis matrix element is the integer round(T·2^14). The products are accumulated vertically and then horizontally, each stage followed by (acc + 2^13) ≫ 14. The output is the Q8 value divided by 256 (before rounding to samples).

### 11.2 Quantization step

```
e = qp − 4;  q = ⌊e/6⌋;  r = e − 6q
Δ(qp) = K[r] · 2^q · 2^(bit_depth − 8)
K = {1, 1.122462048309373, 1.2599210498948732, 1.4142135623730951, 1.5874010519681994, 1.7817974362806785}
```

Chroma uses Clip3(0, 63, qp + cqp_off). When aqp is enabled, a dqp (sint, −12 to 12) is sent per CTU and qp + dqp is used.

### 11.3 Coefficient scan

Coefficients are serialised with a diagonal scan (anti-diagonals from bottom-left to top-right).

### 11.4 Coefficient coding

- **qmode 0 (scalar):** `last` (uintc) is the last non-zero position in scan order. Each coefficient 0..last is coded with sint, conditioned on neighbouring coded coefficients (2D template), block size, DC/non-DC, and the cross-band and change-mask contexts. The last coefficient is coded as |v|−1 plus sign. Reconstruction is q·Δ.
- **qmode 1 (E8 lattice VQ):** coefficients are grouped in 8s (number of chunks coded with uintc). Each group is an E8 lattice point sent as a coset bit and 8 integers. Reconstruction is (lattice point / 2)·Δ.
- **qmode 2 (gain–shape):** clause 11.6.
- **TNS:** a lattice filter of order 1 to 8 is applied along the scan-ordered coefficients (PARCOR coefficients in 4 bits, |k| ≤ 7).
- **Noise filling:** pseudo-random values ±(nf/8)·Δ are placed after `last`.

### 11.5 MTS index syntax

A non-zero flag, then (for s ≤ 16) an extension flag, then 2 bits on a binary tree.

### 11.6 Gain–shape coding (gs)

- **Layout:** the non-DC coefficients are divided into octave bands of radial frequency ρ = √(u²+v²), with band b = 1 + ⌊log2 ρ⌋. Each band is cut in scan order into partitions of at most 16 coefficients.
- **Syntax:** dc (sint), number of used partitions (uintc), and per partition a mode (0 zero, 1 shape quantization, 2 noise substitution, 3 sign only, 4 copy, 5 envelope interpolation).
- **Gain:** g = 2^(qg/4). qg is sent with sint as a difference from its predictor, which is the first available of: the previous gain in the same band, the same band of the previous gain–shape block, the previous gain, and 8 (spec 5.3 partition join). The gain of mode 5 is not sent; it is extrapolated log-linearly from the previous two gains.
- **Shape quantization (mode 1):** gain flag gon (0: use the quantized values as is), quantizer type (0 nonlinear scalar + pre-transform, 1 E8, 2 random projection 16→8), the pre-transform pn for type 0 (0 identity, 1 power 0.75 (the nonlinear scalar quantizer), 2 power 0.6, 3 asinh, 4 signed log, 5 piecewise linear), then the symbols.
- **Copy (mode 4):** a permutation Π (identity, reversal, half-cyclic shift, even/odd swap) and a copy source (1st to 4th previous non-zero partition).
- **Reconstruction:** modes 1 to 4 normalise the shape energy to ĝ (except when gon = 0). Noise is uniform pseudo-random noise from SplitMix64.

---

## 12 Band (PQMF) coding

### 12.1 Filter bank

The cosine-modulated PQMF prototype is a Kaiser-windowed low-pass filter. `pqmf_filter` selects (delay m, β) from {(6, 9.0), (4, 7.0), (8, 10.0), (3, 5.0)}. The number of bands is 2^pqmf_log2 × 2^pqmf_log2y (each 2 to 16, at most 256 in total). The 2D bank is separable with circular extension at the borders.

### 12.2 Per-plane method selection

When pqmf_log2 > 0, `bands` (bit) is sent per plane in the main stream to select band coding or pixel block coding. When band_blocks is enabled and `bands` = 1, `ll_split` (bit) follows.

### 12.3 Band block coding

Each band image is normalised by its synthesis gain (band_norm) into an integer plane and coded with the block coding of clauses 9 to 11. The quantization step of band (k_x, k_y) is Δ·(1 + 0.08·(k_x + k_y)). The coded band to the left (otherwise above), mirror-corrected, is used for cross-band prediction and as context. In inter frames, the bands of the prediction picture serve as temporal prediction. When `ll_split` = 1, the LL band is re-decomposed into 2×2 bands and coded recursively with the same method (spec 3.2 band sub-tree).

---

## 13 Shape layer and dictionary

### 13.1 Shapes

Shape types are Gaussian, ellipse, rectangle and triangle. A shape has:
- a centre (quarter samples);
- log scales l1, l2 (radius 2^(l/8));
- a rotation θ (units of 2π/64);
- an amplitude;
- a linear slope (g_x, g_y);
- an edge width (soft).

Shapes are additively rendered onto luma (S += Σ shapes). The block layer codes X − S, and S is added back at the end.

Syntax: the shape count (uint), then per shape one of the following:

- If the frame is inter and the previous frame has a shape with the same index, a temporal flag. When it is 1, the shape is coded as differences in position, amplitude and slope from that shape (shape motion, spec 7.4).
- Otherwise, a flag "same type, scale, angle and soft as the previous shape", those values when different, then the position difference, amplitude and slope.

### 13.2 Dictionary

See 10.3.

---

## 14 In-loop filters

Applied per plane in the following order; all parameters are sent in the main stream.

1. **Deblocking** (when lf is enabled):
   - **Boundary strength:** a strength bs is determined for each 4-sample segment on the 8-sample grid. It is 2 for boundaries involving intra-type blocks and 1 for boundaries with residual. For luma, it is also 1 where motion differs (different direction or reference, or an MV difference ≥ 4). Otherwise it is 0.
   - **Thresholds:**
     - β(q) follows the HEVC table for q ≤ 51 and is 64 + 2(q − 51) above.
     - tC(q) follows the HEVC table for q ≤ 53 and is round(24·2^((q−53)/6)) above.
     - The offsets beta_off and tc_off (sint, |·| ≤ 12) are added to β and tC respectively.
   - **Luma:**
     - When the blocks on both sides are at least 16 samples long and flat, a long-tap filter is applied. With the shorter side length L: n = 15 if L ≥ 64, n = 7 if L ≥ 32, n = 5 if L ≥ 16.
     - Long-filter weights are f_i = (64·(2(n−i)−1) + n)/(2n). The modification limit is ⌊(tt_i·tC)/2⌋ with tt_i = max(1, (6(n−i) + n/2)/n).
     - Otherwise, the HEVC-style strong/weak filter is used.
   - **Chroma:** only p0 and q0 are modified, at boundaries with bs ≥ 1.
2. **Frequency gain filter** (lf_freq): 8×8 sine-window WOLA (hop 4). DCT coefficients are multiplied by G = max(0, 1 − μσ²/c²) with σ² = Δ²/12 and μ = mu_q/4 (mu_q is uint; 0 disables).
3. **Pixel gain map** (lf_map): γ per 32×32 cell (Q6, 0 to 128) is coded differentially. Output = stage input + γ·(filtered − stage input).
4. **CDEF** (cdef): the direction of each 8×8 block is estimated among 8 directions. A primary filter (strength 0 to 15) and secondary filters at ±45° (strength {0, 1, 2, 4}) are applied with a constraint function. Up to 4 strength pairs are sent per plane, and one pair is chosen per 64×64 block.
5. **ALF** (alf): 7×7 diamond filter, 12 classes.

---

## 15 Lossless layers (L1/L2)

- `lossy` = 0: original samples are coded directly with MED prediction (fully lossless).
- `l2` = 1: for the lossy reconstruction R, the residual e = X − R is coded with gradient and neighbouring-residual contexts so that X is reconstructed exactly.

---

## Annex A (normative): Value ranges and constraints

| Item | Range |
|---|---|
| width, height | 1 to 65535 |
| bit_depth | 8 to 16 |
| qp | 0 to 63 |
| cqp_off | −12 to 12 |
| dqp (aqp) | −12 to 12 |
| min_log2 / max_log2 | 2 to 6, min ≤ max |
| CTU | 64 |
| tiles | 1–16 × 1–16 |
| references | 0 to 4 in each of L0 and L1 |
| DPB | 12 pictures |
| pqmf_log2, pqmf_log2y | 0 to 4 |
| dictionary | up to 65536 entries |
| gm parameters | |a|,|b|,|c|,|d|,|h31|,|h32| ≤ 32767 |
| g_p / h_p | 0 to 256 / 0 to 128 (Q6) |

A decoder shall treat a stream as corrupt when it detects an out-of-range value.

## Annex B (informative): Encoder guidelines and presets

### B.1 RD optimisation

- λ = 0.57 · 2^((qp−12)/3) · 4^(bit_depth−8). Rates are estimated as −log2 p from the current probabilities of the CM models.
- **Psychovisual distortion:**
  - The error is split by a 3×3 binomial filter into low and high bands: D = ‖e_lp‖² + Σ w_c‖e_hf,c‖², with w_c = 1/(1 + ψ·σ²_c/Δ²).
  - σ²_c is the minimum, over the 4×4 cell and its four neighbours, of the source variance. This keeps edges unmasked.
  - Disabled by `--tune psnr`.
- For blocks straddling the valid region, distortion is measured only inside the region.
- For I frames (slow and above), the encoder also codes the frame without TNS/IBC/dictionary/MTS and keeps whichever version has the lower D + λR.

### B.2 Presets

| Preset | Block sizes | Intra candidates | Main additional tools | B frames | Tiles |
|---|---|---|---|---|---|
| faster | 8–32 | 4 modes | — | 3 | 2×2 |
| fast | 8–64 | 2 | RDOQ, ALF | 7 | 2×2 |
| medium | 4–64 | 3 | band tools, frequency gain, CDEF | 7 | 1×1 |
| slow | 4–64 | 6 | TNS, IBC, dictionary, MTS | 7 | 1×1 |
| placebo | 4–64 | 35 | E8, shapes, rectangular partitions, gain–shape, FIR basis | 7 | 1×1 |

QP cascade: I −5, P −2, B +1+hierarchy depth.

## Annex C (informative): Performance results

Conditions:
- Metric: weighted PSNR = (6Y + U + V)/8, BD-PSNR over 5 points.
- Encoder settings: `--preset slow --tune psnr`.
- Anchor: libaom (AV1, still-picture mode, cpu-used 4).

| Test | Result |
|---|---|
| img3 (854×362) vs AV1 | −0.712 dB (at this version) |
| Zoom + rotation clip, before/after the global model | 18,893 → 11,782 bytes (−38%), PSNR +0.05 dB |
| Focus-change clip, per-band gain | −3.6% bytes, +0.15 dB |
| Focus-change clip, FIR basis (`--firb=1`) | −5.7% bytes |
| CTU 128 (trial, not adopted) | img3 +0.027 dB, video −0.02 to −0.07 dB |

## Annex D (informative): Implementation status and known limitations

| Item | Status |
|---|---|
| Inverse transform, quantization step, OBMC weights | integer (normative) |
| PQMF, TNS, gain–shape reconstruction, frequency gain filter, generation of the CM stretch/squash tables | double-precision floating point (FP contraction and fast-math disabled); bit-exactness across implementations not yet guaranteed |
| Maximum block size | 64 (128 evaluated and not adopted) |
| Spec 5.7 PWL knots | normative default table (not transmitted) |
| Spec 11 CM | 4-input hashed contexts + mixer + APM (not PPMd-style variable order) |
| Shapes | 4 types, no shear |
| Dictionary prediction | 8×8 and 16×16 luma only |
| Gain–shape, inter shapes, band method | implemented; RD rarely selects them in tests, so they are enabled by default only at placebo |

## History

| Version | Date | Information about changes |
|---|---|---|
| V0.9.0 | 2026-10 | First version (based on the reference implementation at commit e822af4) |
