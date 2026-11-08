# EXTFP scientific protocol extracts — EXTFP_AMENDMENT_v4_POSTHOC.md

Local specification date: 2026-10-28.

Edited scientific extracts in English translation, prepared 2026-11-07. This reading copy is not byte-identical to the frozen original. Original and public file hashes are recorded in PROVENANCE.json.

## Specification timing

Post-hoc amendment. Exploratory partial areas, seed means and standard deviations, counts at fixed thresholds and positive-image counts were known before specification. The specified confidence intervals and random example selection had not been computed. The analysis interval and step rule were fixed after exploration.

## §2 Post hoc endpoint S2 — partial area (external recall 0.30–0.60)

**Status:** post-hoc sensitivity analysis. It is subject to all of the same interpretation limits as S1 (items 1–6 of v3 §2). No verdict (§2.5).

### 2.1 Target and inputs

- Pair: the single v1 confirmatory pair YOLOv8m − YOLO11l. Not applied to RT-DETR-l or to the 9 v2 runs (same reason as v3 §3.1).
- Inputs, matching, recall and FP definitions: exactly as in v3 §3.1 and §3.3 (sealed raw predictions, greedy matching at IoU 0.50, class 7, lesion micro recall, every box at or above τ on a negative image = FP). The functions are imported from the same sealed module.

### 2.2 FP function along the curve (same rule as the v3 §3.3 operating-point rule)

- Grid k = 1…999; recall(k) and F(k) = lesion recall and FP per negative image at that k. Both are non-increasing in k.
- Burden at recall r: f(r) = F(k\*(r)), where k\*(r) = **the largest k satisfying recall(k) ≥ r**. This extends the rule that v3 S1 uses at the single point R\*=0.45 to the whole interval.
- The recall interval covered by k is therefore the half-open interval (recall(k+1), recall(k)] (recall(1000) := 0). A k with recall(k+1) = recall(k) has width 0 and contributes nothing (handling of repeated recall coordinates).

### 2.3 Area (step integration, fixed)

- A = Σ_{k=1}^{999} F(k) · |(recall(k+1), recall(k)] ∩ [0.30, 0.60]|.
- Reported value = **PA = A / 0.30** (interval-mean FP per negative image; the same unit as FP per image). Whether the endpoints are open or closed does not affect the length.
- Computation: the length is computed in float64 as `max(0, min(recall(k), 0.60) − max(recall(k+1), 0.30))`. Recall is ΣwTP/ΣwGT divided in float64. No interpolation is performed.
- **Reach failure:** if recall(1) < 0.60, f(r) is undefined on part of the interval, so that curve has no PA. In the full sample recall(1) = 0.6902/0.7466, so the interval is reached.
- ΔPA = PA(YOLOv8m) − PA(YOLO11l). The sign convention is the same as in v1.

### 2.4 paired stratified bootstrap (shares the v3 §3.4 resampling as is)

- **The index matrices P and N of v3 §3.4 are used as is**: `default_rng(20261016)`, B = 10,000, P = integers(0,654,(B,654)) first, then N = integers(0,1154,(B,1154)). In the same replicate b as S1, the curves of both models are rebuilt with the same (P[b], N[b]), and PA and ΔPA are recorded. S1 and S2 are therefore computed on the same resampling.
- Reach-failure replicates (recall(1) < 0.60 for either model) are excluded and their number is reported. **If there are 100 or more, no CI is reported**; only "PA_REACH_FAIL" and the count are written.
- CI: `numpy.percentile(·, 2.5)` and `(·, 97.5)` applied to ΔPA_b and to each model's PA_b over the valid replicates. The same call as in v3 §3.4.

### 2.5 Why no verdict is assigned

The MID of 0.10 was defined for a difference in FP per image at a single operating point (v1 §7.3). Transferring the same margin to an interval average would be a new choice made after seeing the results, so S2 has no verdict table; only the point estimate and the 95% CI are reported.

**Recorded prediction:** the lower bound of the 95% CI for ΔPA is above 0, because of the direction of the inspected values.

## §3 Difference in seed means — Welch t 95% CI (replaces v3 §4.1c)

- Input: `descriptive_within_model_seed_variation` in `endpoints_v2.json` → the 3 `per_seed` values each for YOLOv8m and YOLO11l (sealed; no recomputation). YOLOv9c is not used for this difference.
- Difference d = mean(v8m) − mean(11l). SE = sqrt(s₁²/3 + s₂²/3) (s = sample SD, ddof=1). Welch–Satterthwaite degrees of freedom ν = SE⁴ / [ (s₁²/3)²/2 + (s₂²/3)²/2 ].
- 95% CI = d ± t₀.₉₇₅,ν · SE. The t quantile is `scipy.stats.t.ppf(0.975, ν)` (non-integer ν used as is). If scipy is unavailable, execution stops (substituting an approximation is prohibited).
- **No verdict.** v2 E1 (verdict R2, paired by seed number) is not changed.
- Interpretation limits: with 3 seeds per model the degrees of freedom are small and the CI is wide; the interval relies on the assumption that seed burdens are normally distributed; it does not include the image-sampling variation of each seed value itself (no seed × image 2-stage bootstrap is performed, for the same reason as v3 §4.1c).

## §4 Positive-image FP and image-level specificity (post hoc descriptive, three arms)

- Operating point: prespecified τ (0.268/0.146/0.494). Not computed at any other τ.
- **Positive-image FP per image:** for positive image i, FPpos_i = (number of boxes at or above τ) − TP_i, where TP_i comes from the v1 §6.2 matching (sealed function imported). Value = ΣFPpos_i / 654. These boxes are predictions that failed IoU 0.50 matching to a class 7 reference box, and they also include localisation errors near lesions.
- **Image-level specificity:** the proportion of negative images with 0 boxes at or above τ = 1 − (proportion of images with ≥1 FP).
- CI: the same method as v3 §4.1d — per-image vectors (FPpos_i for the 654 positive images; 0/1 indicators for the 1,154 negative images) are passed to the imported v1 `bootstrap()` (seed 20260830, B 10,000), and percentiles 2.5/97.5 are taken of the returned per-arm resampled means. Positive and negative images are passed separately (their n differ, so their index matrices differ). No verdict and no pairwise differences.

## §5 95% CIs for the 9 bars of the per-arm false-positive figure (extension of v3 §4.1d)

- Bars = 3 arms × {prespecified τ, internal-recall 0.50 τ, 0.70 τ}. Only the sealed τ values are used (`secondary_75.json` item6: 0.417/0.339/0.629, 0.171/0.146/0.478). YOLO11l's 0.70 τ is 0.146, the same as its prespecified τ, so the two bars have the same value and the same CI (left as is).
- Method: for each τ set, the v1 `load_negative_fp_counts()` and `bootstrap()` are imported and called, and percentiles 2.5/97.5 are taken. The three τ sets have the same seed and the same n, so they use the same index matrix (asserted by the code). Post hoc descriptive CIs; no verdict.
- The CI of the primary-pair difference, used as an anchor in a figure, uses the sealed value [−0.2556, −0.1681] as is and is not recomputed (v3 AC6 reproduction only).

## §6 Generalisation gap (v1.1 §7.5 item 3) not performed

The internal-to-external generalisation gap named in the protocol is not computed: the protocol defines no metric for it, and computing one would require new inference on the internal test split, which this amendment does not perform.

## §7 Selection rule for representative FP example images (fixed before results)

- Target arm: YOLOv8m, τ 0.268 (prespecified). Boxes displayed: all YOLOv8m boxes at or above τ on that image. On the third image, the reference boxes of confusable findings that overlap with IoU > 0 are also drawn.
- Pool A = negative images with ≥1 YOLOv8m box at or above τ (436 images), in roster order.
- Pool B = negative images with ≥1 box in v3 §4.1a category ⑴ (IoU > 0 with a confusable-class box), in roster order, excluding the two images drawn from pool A.
- `rng = numpy.random.default_rng(20261028)`. First, two images in the order returned by `rng.choice(len(A), size=2, replace=False)`; then one image by `rng.choice(len(B), size=1)` with the same rng. If pool B is empty, the third image is not included and that fact is stated.
- **No redraws.** Images are not replaced on grounds of image quality or appearance. The selected image keys are written to the log before the figure is drawn.
- Display processing: the timestamp band at the bottom is covered with an opaque rectangle (the masked height is stated in the figure legend). Images are only downscaled; contrast and brightness are not changed. Source: Zenodo 10.5281/zenodo.15487430, Diagnostics 2024;14:2336, under the research-publication clause of the usage agreement. The image files themselves are not deposited or redistributed.

## §9 Code acceptance criteria (frozen in the same commit as this amendment)

Implementation: the v4 items are placed together in the single v3 §5 script `analysis/extfp_posthoc_v3.py` (sharing the curve reconstruction and AC3). v4 outputs are written to separate files. v3 AC1–AC10 continue to apply unchanged.

- **V1 hashes:** in addition to v3 AC1, the v3 file sha256 = `3400864c86c9fd297689976f29dfcaf58841976452a47dcf648d965c99ff97c6` (`original_sha256` in PROVENANCE.json), and this file's sha256 = the committed value. Otherwise exit 2, `PRESPEC_MISMATCH`.
- **V2 step-integration self-check:** on the hand-built curve recall = (0.70, 0.50, 0.50, 0.40, 0.20), F = (3, 2, 1.5, 1, 0.5) (k = 1…5, recall(6) := 0), the result must match A = 0.55 and PA = 1.833333… to within 1e-12. [Hand calculation: r∈(0.50,0.60]→k=1, F=3, length 0.10; k=2 has width 0; r∈(0.40,0.50]→k=3, F=1.5, length 0.10; r∈(0.30,0.40]→k=4, F=1, length 0.10. A = 0.30+0.15+0.10 = 0.55]
- **V3 mutation tests:** ① changing k\*(r) to "the smallest k" makes V2 fail (A = 0.90) ② assigning F(k) to the interval shifted by one step, (recall(k), recall(k−1)] (left–right reversal of the step), makes V2 fail ③ if S2 uses P and N different from those of S1, `SHARED_INDEX` fails.
- **V4 full-sample S2 reach:** reach is confirmed at recall(1) = 0.690230/0.746624. The point estimate PA is output (agreement with the inspected values is not required; the difference is written to the log).
- **V5 Welch:** self-check of the seed means 0.629694/0.689197, d = −0.0595031774 (10 decimal places), `t.ppf(0.975, 2)` = 4.302653 (6 decimal places).
- **V6 positive-image FP:** for the three arms, ΣTP_i = 523/605/539 and ΣFPpos_i = 689/946/782 (exact integer match with sealed item1). Otherwise exit 3.
- **V7 specificity:** number of 0-FP negative images = 1154 − 436/504/605 = 718/650/549. Otherwise exit 3.
- **V8 per-arm false-positive figure:** external FP totals at the internal-recall τ = 349/278/412, 986/893/990 (exact integer match with sealed item6); means at the prespecified τ = sealed `primary_endpoint.json` to within 1e-12; assert that the index matrices of the three τ sets are identical.
- **V9 FP examples:** two selections from the same input give the same keys; pool A size 436; selected keys recorded in the log before the figure is generated.
- **V10 outputs:** a CSV with one row per S2 replicate (b, ΔPA, the two PAs, reached or not), a v4 JSON summary, a log and a sha256 sidecar. The S2 result is accepted only as the single line printed by the script (`S2_CI=<low>,<high>` or `S2_CI=PA_REACH_FAIL`).

## §10 What is not done

- The §2 interval (0.30–0.60), integration rule, seed and B are not changed after seeing the results. No other intervals or interpolations are additionally reported.
- No verdict is assigned to S2 or to the Welch CI. The v1 P1, v2 E1/E2 and v3 S1 verdicts are not changed.
- No retraining or new inference. The generalisation gap (v1.1 §7.5 item 3), the IoU 0.3 sensitivity analysis, variance-component analysis and the seed × image 2-stage bootstrap are not performed.
- FP examples are not redrawn, and no examples from other arms are added.
- External images, annotations and per-image predictions are not deposited or redistributed.
