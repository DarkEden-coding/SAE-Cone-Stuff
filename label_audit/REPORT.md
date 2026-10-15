# BRT keypoint label quality audit

Audit of `datasets/brt-cone-pose/brt-cone-pose-11k` labels against image evidence (2026-10-13).
Question: do the human left/right keypoints sit on the cone silhouette edge, and does motion blur
explain the inward / low bias the trained models showed on the synthetic test set
(inward 1–1.6 % of box height on small cones, 3–5 % on big orange; low 1–2 %; +1.6 % PnP range bias)?

Sign convention: offset > 0 = label inside the silhouette. %H = percent of labelled box height.
Full numeric output: `analysis_stdout.txt`; breakdowns: `tables/*.csv`.

## Findings

**Horizontal placement — small cones are on the line.**
- Small-cone stripe-edge points (kp 0–3, n = 24,683): median −0.08 px (−0.08 %H), robust spread 1.28 px (1.36 %H).
  The detector reads −0.33 px on exact synthetic ground truth, so labels are net ≈ +0.25 px inside — far too
  small to explain the models' 1–1.6 %H inward bias.
- All gated keypoints (n = 44,902): median −0.15 %H, spread 1.71 %H, 5–95 % range −4.2 to +4.9 %H.
- No source shows an inward bias on small cones (source medians −0.42 %H amz to +0.68 %H mms).
- **Big orange is inside**, growing downward: +0.6 / +1.1 / +2.0 / +2.9 %H for pairs 0–3 (synthetic baseline
  −0.2 to −0.07 %H). This explains part of the models' 3–5 %H big-orange inward bias.
- **Foot points (kp 4/5)** have ~2× the scatter and read 0.7–1.6 %H *outside* the edge: the definition is
  ambiguous where the body flares into the base.

**Vertical placement — labels are systematically low.**
Stripe-edge labels sit ≈ 1.45 %H low (+1.9 %H at stripe top, +1.0 %H at stripe bottom), consistently across
all 18 sources (+1.05 to +1.68 %H). This matches the models' 1–2 %H low bias, so that part *is* learned from
the labels. Magnitude uncertain by about ±0.5 %H (depends on the synthetic cone model); the sign is solid.
Foot heights match the synthetic geometry within ≈ 0.6 %H.

**Motion blur — present, adds scatter, does not explain the offset.**
- Edge 10–90 % rise: median 5.75 px (5.2 %H), 5–95 % 2.75–9.25 px; 9.7 % of edges wider than 8 px, 0.2 % wider than 12 px.
- After subtracting the detector's own blur trend (measured on synthetic GT), the blurriest edges are at most
  ≈ +0.6 %H (≈ 0.5 px) more inward.
- Blur raises label scatter on small cones from 1.0 to 1.7 %H robust sd (detector alone: 0.57 to 1.25 %H).
- 18.5 % of *sharp*-edge labels are already > 1 %H inward, so the inward tail is mostly not blur.
- The rise width mixes motion blur, defocus, resampling and JPEG; images are letterboxed/resized, so motion
  blur cannot be isolated.

**Gross errors — roughly 7–9 % of cones.**
- Geometry checks flag 8.6 % of cones with H ≥ 30 px (n = 72,248), 9.3 % at H ≥ 60 px; synthetic GT triggers
  0 flags at the same thresholds.
- Hard errors (swapped left/right, ordering violations): 2.6 %. Asymmetric pairs 5.9 %, narrowing-downward 4.0 %.
- A visual check of 12 flagged cones per category found almost all visibly wrong (scrambled points, points on
  a neighbouring cone, tipped cones labelled as upright).
- Worst sources: eufs 23 %, FRT 20.5 %, fsbizkaia 17.5 %, ka 15.6 %. Best: eco 2.9 %.
- On strong clean edges, 11.7 % of small-cone stripe points are > 3 %H off the edge (synthetic 2.7 %) and
  3.5 % are > 5 %H off (synthetic 1.0 %).

## Implication for the synthetic-set results

The labels explain the models' **low** bias and part of the **big-orange inward** bias, but **not the small-cone
inward bias**. That points to either a synthetic-domain effect (how rendered edges look to the network) or
model/loss behaviour. Recommendation 1 below distinguishes the two.

## Method

- Per keypoint, a horizontal edge search along the keypoint's row inside the box (cone-colour segmentation plus
  gradient, sub-pixel fit) gives the signed offset to the silhouette; a stripe-transition detector checks vertical
  placement. Cones with H ≥ 60 px, not truncated; keypoints pass an edge-quality gate (27 % pass; ungated median
  is the same, −0.23 %H).
- **Detector validation** on `synth_data/` (exact silhouette keypoints, similar blur, noise, JPEG): robust sd
  0.89 px (0.97 %H), bias −0.21 px; known label shifts of +2 / −2 / +5 px recovered as +1.99 / −1.99 / +4.79 px.
  Left and right read ≈ 1 px apart in the same direction in real and synthetic data (a half-pixel convention
  difference that cancels in symmetric statistics). See `sheet_V_synthetic_validation.jpg`.
- Blur: per-cone 10–90 % edge rise width, corrected by the detector's synthetic blur trend.
- Gross errors: geometric plausibility checks with thresholds set from synthetic extremes.

## Limitations

- Detector accuracy on real images is inferred from synthetic validation; it may be worse in clutter.
- Only cones with H ≥ 60 px (≈ 26 % of cones; median H is 42 px) are measured for offsets; smaller cones are
  covered only by the gross-error checks.
- The inward search window is 0.3 × pair width, so the extreme inward tail is under-estimated.
- The blur correction assumes the synthetic blur resembles real blur; the gross-error thresholds are heuristic.
- Duplicate images across splits were not checked.

## Recommendations

1. Run the same detector on model predictions over real val images. ≈ 0 offset there means the small-cone
   inward bias is a synthetic-domain gap; inward on real images too means training or loss.
2. Snap keypoints to the detected edge where the gate passes and the shift is small — mainly fixes big orange,
   foot points and scatter.
3. Relabel or drop the ≈ 2.6 % hard-error cones and review asymmetric/narrowing flags, starting with eufs, FRT,
   fsbizkaia, ka, ff and tuwr.
4. Shift stripe-edge labels up ≈ 1.5 %H, or snap to the stripe transition plus a calibrated offset.
5. Redefine or exclude foot keypoints for PnP.
6. Downweight (rather than drop) blurred samples with rise > 8 px.
7. Standardise per-source conventions (e.g. FRT places the stripe bottom at 0.373 H above the box bottom vs
   0.43–0.48 H elsewhere; prom labels narrower widths).

## Files

- Scripts: `audit_lib.py`, `run_audit.py`, `geom_flags.py`, `analyze.py`, `contact_sheets.py`, `validate_synth.py`
- Data: `keypoint_measurements.csv` (166,646 rows), `cone_summary.csv` (per-cone offsets, blur, flags),
  `tables/*.csv` (incl. `detector_validation.csv`), `analysis_stdout.txt`, `work/*.csv`
- Figures: `fig_offsets.png`, `fig_blur.png`
- Contact sheets (red = label, cyan tick = detected silhouette edge, yellow tick = stripe transition):
  `sheet_V_synthetic_validation.jpg`, `sheet_A_on_the_line.jpg`, `sheet_B_inward.jpg` (only 13 of 547 cones
  qualify), `sheet_C_outward.jpg`, `sheet_D_blurred.jpg`, `sheet_D2_sharp.jpg`, `sheet_E_big_orange.jpg`,
  `sheet_F_gross_swap.jpg`, `sheet_F_gross_order.jpg`, `sheet_F_gross_asym.jpg`, `sheet_F_gross_narrowing.jpg`

Audit performed by a Sonnet 5.5 subagent; report written from its output.
