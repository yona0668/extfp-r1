# EXTFP scientific protocol extracts — EXTFP_AMENDMENT_v3_POSTHOC_DRAFT.md

Local specification date: 2026-10-19.

Edited scientific extracts in English translation, prepared 2026-11-07. This reading copy is not byte-identical to the frozen original. Original and public file hashes are recorded in PROVENANCE.json.

## Specification timing

Post-hoc specification. External curves, matched-recall point estimates, exploratory partial areas and training-seed summaries had been inspected before specification. The reversal at matched recall was already known. These analyses are not confirmatory. The box-overlap decomposition executes an earlier specified definition.

## §2 Circularity and optimistic bias — interpretation limits (attached to every result)

1. **τ is chosen with external labels.** This falls under the prohibitions in v1.1 §6 and §9 (no threshold
   selection with external data, `EXTFP_PRESPEC.md`) and in v2 §V5 and §V8. This amendment makes an exception
   to that prohibition **for this post hoc curve analysis only**; the prohibition continues to apply to the
   existing endpoints and τ. The result of the exception is called only a
   "**sensitivity-standardized post-hoc comparison**".
2. **This is not deployment performance.** In actual deployment, τ must be set without the external ground
   truth. An operating point whose recall is matched with the external ground truth is an oracle calibration
   that cannot be reached in deployment; this value is not used as a deployment FP burden or as a recommended
   operating point.
3. **Optimistic bias.** τ is set on the 654 positive images of the same corpus, and FP is measured on the
   1,154 negative images of the same corpus. The images do not overlap, but the distribution is the same, so
   the comparison is made with recall matched more exactly than it would be in reality.
4. **The choice of R\* is itself outcome-dependent.** R\* was set after viewing the external recalls of the two
   confirmatory operating points (0.4154 and 0.4805). The bootstrap does not reflect the uncertainty of the
   choice of R\*.
5. **Patient clustering is not reflected.** The analysis roster (`external_cohort_roster.jsonl`) has no
   patient-linkage field, so only image-level resampling is possible (whether the metadata of the original
   dataset release contain patient information is unverified). If one patient contributed several images, the CI
   comes out narrower.
6. **One point does not represent the curve.** The single result at R\*=0.45 is not used to claim which curve is
   better overall. In prior exploration, τ and FP moved together with R; this inspection is stated in
   Specification timing.

## §3 Post-hoc sensitivity endpoint S1 — FP difference at external recall 0.45

**Status:** post-hoc sensitivity analysis. **Separate** from the existing primary endpoint (v1 §7.1–7.4). Its
verdict does not change v1 P1 or v2 E1/E2.

### 3.1 Target

- Pair: the single v1 confirmatory pair **YOLOv8m − YOLO11l** (sealed v1 runs, seed 20260515). The RT-DETR-l
  pairs are confounded pairs and are not computed (v1 §7.4). Not applied to the nine v2 runs (§6 "What is not
  done").
- Input: sealed raw predictions `external_predictions_{yolov8m,yolo11l}.jsonl` (sha256 `19f6cc08…`, `2d34994e…`;
  must match `raw_external_predictions.sha256`). Retraining and new inference: 0.
- Image order: row order of `external_cohort_roster.jsonl`. The 654 positive images (`is_positive=true`) and the
  1,154 negative images are each numbered with 0-based indices in that order.

### 3.2 R\* = 0.45 (post hoc representative point)

- Definition: the midpoint 0.4479745… of the external recalls 0.4154 and 0.4805 of the two confirmatory
  operating points, rounded to two decimal places.
- Not a confirmatory primary endpoint. For other R (0.415, 0.481, 0.50), the point estimates inspected in prior
  exploration are reported only as **descriptive values**, with no CI or verdict attached.

### 3.3 Operating-point rule (fixed)

- Grid: τ_k = k/1000, k = 1…999 (v1 §5.3). Comparisons are made on the integer k.
- Matching: exactly as in v1 §6.2 (IoU 0.50, greedy 1:1 in descending score order within each image,
  class-agnostic, external ground truth = class 7). The implementation **imports** the functions of the sealed
  script `analysis/extfp_secondary_75.py` (0 copies or reimplementations).
- Recall: lesion micro. TP_i(k) and GT_i of positive image i are added as many times as the image was drawn into
  the sample: recall(k) = Σ w_i·TP_i(k) / Σ w_i·GT_i (in the full sample, w_i = 1 and Σ GT_i = 1,259).
- **Selection: the largest k among those satisfying recall(k) ≥ 0.45.** To avoid floating point, this is
  decided with the integer expression `20·ΣwTP ≥ 9·ΣwGT`.
- **Ties:** in per-image greedy matching, the matching of boxes at or above τ is a prefix of the full matching,
  so TP(k) is non-increasing in k and "the largest k" is unique. No separate tie rule is needed (the code
  asserts uniqueness).
- **Target not reached:** if the condition is not met even at k=1, that replicate is recorded as a "reach
  failure" (no Δ). The full sample reaches the target, because its recall at k=1 is 0.6902/0.7466.
- FP/negative image = (Σ w_j of the number of boxes at or above τ on negative images at the selected k) / 1,154.
  Exactly the v1 §7.1 definition (every box on a negative image is an FP; no exemption for confusable findings).
- Δ = FP/image(YOLOv8m) − FP/image(YOLO11l). The sign convention is the same as in v1 (Δ<0 = YOLOv8m lower).

### 3.4 paired stratified bootstrap (fixed)

- `rng = numpy.random.default_rng(20261016)`. B = 10,000.
- Fixed drawing order: first the positive index matrix `P = rng.integers(0, 654, size=(B, 654))`, then the
  negative index matrix `N = rng.integers(0, 1154, size=(B, 1154))`. Stratum sizes are the same as in the
  original sample.
- **The same (P[b], N[b]) is applied to both models simultaneously** (paired). Indices are not drawn separately
  for each model.
- In each replicate, **τ is reselected for each of the two models by the §3.3 rule** (based on recall in the
  resampled positive images). At that τ, FP/image on the resampled negative images is computed, and Δ_b is
  recorded.
- Reach-failure replicates (either model not reaching the target at k=1) are removed from the Δ distribution,
  and their number is reported. **If reach failures are 1% of B (100 replicates) or more, the verdict is
  automatically D0 (indeterminate — reach failure).** If there are 0 valid replicates, no CI is computed, and
  `null` with the reason `NO_VALID_REPLICATE` is recorded (verdict D0).
- CI: `numpy.percentile(·, 2.5)` and `numpy.percentile(·, 97.5)` (default linear interpolation) on Δ_b of the
  valid replicates — the same call as in v1 `analysis/extfp_pairwise_7274.py`. BCa is not used. The verdict is
  made on float64 values before rounding.
- What this bootstrap reflects: sampling variation of positive and negative images and the resulting τ
  reselection. What it does not reflect: patient clustering, the choice of R\*, training-seed variation (§2-4
  and §2-5, §4.3).

### 3.5 Verdict table (95% CI [L, U], MID = 0.10, boundaries inherited from v1 §7.3.2)

v1's `judge()` (`analysis/extfp_pairwise_7274.py`) is imported to obtain the three-way classification, and only
"substantial difference" is split in two by sign. The branches are applied in order, and the first branch that
matches is used.

| Branch | Condition (v1 boundaries: the endpoint closer to 0 has \|value\| ≥ 0.10; equivalence is a closed interval) | Name |
|---|---|---|
| 0 | reach failures ≥ 100 (including 0 valid replicates) | **D0 indeterminate (reach failure)** |
| A | judge = substantial difference and U < 0 (i.e., U ≤ −0.10) | **Substantial difference in the original direction** (YOLOv8m lower) |
| B | judge = substantial difference and L > 0 (i.e., L ≥ +0.10) | **Substantial difference in the reverse direction** (YOLO11l lower) |
| C | judge = substantial equivalence (−0.10 ≤ L and U ≤ +0.10) | **Exploratory equivalence** |
| D | none of A, B and C (e.g., [0.02, 0.15], a CI that crosses the boundary of the equivalence range) | **Indeterminate** |

The code asserts that the verdict function is surjective (total) and mutually exclusive (§5 AC8). Because this
verdict is post-hoc, no branch is used as "confirmatory". **Recorded prediction: B** (because the point Δ +0.168
had already been seen).

## §4 Retained and excluded items

### 4.1 Retained — prespecified analyses performed late (not post hoc, but disclosed as "performed after the results were known")

- **(a) §8 box-level FP decomposition.** Exactly as worded in v1.1 §8: boxes on negative images at or above τ
  (0.268/0.146/0.494) are divided into ⑴ those with IoU > 0 with any confusable-class box in the same image
  (Endodontic treatment, Bone resorbtion, Root fragment, Furcation lesion, Apical surgery, Root resorption — the
  6 classes listed in §8) and ⑵ the rest, and counts and per-image values are reported. They are not removed
  from the denominator (1,154). No CI, test or verdict. All three arms. The image-level FP of the 282 clean
  negative images (already sealed) is cited as is.
  - Details the wording does not specify are fixed now: IoU in pixel coordinates xyxy; ⑴ if the intersection
    area > 0. The confusable-class set is imported as `CONFUSABLE_CLASSES` from the sealed cohort code
    (`analysis/extfp_external_cohort.py`, `legend.txt` mapping, already used to count the 872 hard negative
    images). These details are disclosed as **implementation choices made in the amendment**.
- **(b) FROC curve overlay.** The sealed CSV (§7.5-2) as is. The figure marks a horizontal line at R\*=0.45 and
  the two confirmatory operating points. No new computation.
- **(c) Per-seed descriptive statistics.** The per-seed burdens, range and SD in the sealed v2 output `aggregate_results/endpoints_v2.json` are cited as
  is. The seed-averaged difference −0.0595 is used only as a **descriptive value**, and the three per-seed Δ
  (−0.2366, −0.0087, +0.0667) are listed side by side. The three seeds are only enumerated; no distribution is
  estimated from them. Pairing architectures by seed number is a choice that requires justification (the same
  integer seed does not produce the same random path in different architectures). No seed×image two-stage
  bootstrap is performed (three seeds cannot yield a CI for training variation).

### 4.1-2 Post hoc descriptive outputs (not prespecified)

- **(d) Per-arm 95% CIs for figure display.** The per-arm descriptive endpoints of v1.1
  §7.1 have no CI specified, and the 95% CI of §7.3.1 covers only pairwise differences, so these CIs are
  **post hoc descriptive CIs** (they only reuse the existing resampling procedure and are not a prespecified
  item). The sealed outputs contain no per-arm FP/image CI (0 CI entries in `primary_endpoint.json`,
  verified). They are obtained exactly by the v1 §7.3.1 procedure — percentile 2.5/97.5 on the per-arm
  resampled means returned by the imported `bootstrap()` (seed 20260830, B 10,000). For description only; no
  verdict. The Δ CI is only reproduced (AC6), not newly created.

### 4.2 Post-hoc description — second-center subgroup (v1.1 §7.5 item 4)

- `test_alte_cabinete` 180 images = 132 negative + 48 positive. Only metrics already defined in v1.1 are used:
  FP/image on the 132 negative images (sealed τ; values already inspected during exploration) + lesion micro
  recall on the 48 positive images (sealed τ, IoU 0.50 matching). Point estimates only; no CI or verdict.
  Labelled "post-hoc exploratory".

### 4.3 Excluded (reasons recorded — scientific summary of the original)

- Because the partial area differed with the integration method and the interval and curves had already been
  inspected, v3 did not adopt it as a formal post hoc endpoint. Later, v4 §2 stated that prior inspection and
  separately specified a post hoc analysis using step integration and a fixed interval. The v3 decision is not
  retroactively revised.
- The generalisation gap from the internal test to the external set was not computed in this amendment, because
  the required internal predictions would have to be newly generated. The subsequent v4 §6 states both that the
  metric is unspecified and that new inference would be required.
- Assembling all seven exploratory items of v2 §V6 was outside the scope of v3. The per-run operating points
  already reported through the primary-endpoint procedure, and the saturation counts at those operating points,
  do not substitute for a saturation tally over the full score grid or for an exploratory list presenting
  operating points, parameters and inference times together.

## §5 Code acceptance criteria (frozen before the code, in the same commit as this amendment)

Planned script path: `analysis/extfp_posthoc_v3.py` (writing or running it before the amendment is frozen is
prohibited).

- **AC1** At start, unless `EXTFP_PRESPEC.md` CRLF sha256 = `c04a5225…bddb` and this amendment file's sha256 =
  the committed value, exit 2 and print `PRESPEC_MISMATCH`.
- **AC2** If the sha256 values of the two raw prediction files differ from `raw_external_predictions.sha256`,
  exit 2, `INPUT_MISMATCH`.
- **AC3** The curve rebuilt from the full sample matches `secondary_75_froc_{yolov8m,yolo11l}.csv` as integers in
  `tp`, `fn` and `total_fp_1154` on all 999 rows. If even one differs, exit 3, `FROC_REPRO_FAIL`.
- **AC4** Reproduce FP 649/893 (0.5624/0.7738) and TP 523/605 (recall 0.4154/0.4805) at τ 0.268/0.146.
- **AC5** The §3.3 rule on the full sample reproduces τ 0.192/0.182, recall 0.451152/0.451946 and FP
  0.782496/0.614385 to 6 digits.
- **AC6** Reproduce Δ0–1 −0.2114 [−0.2556, −0.1681] to 4 digits with the imported v1 `bootstrap()` (seed
  20260830).
- **AC7** The matching, curve and verdict functions are imported from `extfp_secondary_75` and
  `extfp_pairwise_7274`. If a local definition with the same name exists, the assert fails. These modules do not
  run analysis or write files on import, but `extfp_secondary_75.py` runs `sys.path.insert` (search-path change —
  allowed, recorded in the log). Imports are done with `PYTHONDONTWRITEBYTECODE=1` so that no bytecode is written
  to the repository.
- **AC8** Self-check of the verdict function: (L,U) = (−0.30,−0.10)→A, (0.10,0.30)→B, (−0.10,0.10)→C,
  (0.02,0.08)→C, (0.02,0.15)→D, (−0.05,0.15)→D, reach failures 100→D0, valid replicates 0→D0 and CI null. All
  five names are reached (surjective), and exactly one per input.
- **AC9** Three mutation tests, each of which must fail with a named assert: ① using a different index matrix for
  each model → `PAIRED_INDEX` fails ② "the largest k" changed to "the smallest k" → AC5 fails ③ drawing from the
  1,808 images without strata → `STRATUM_SIZE` (654 positive and 1,154 negative per replicate) fails.
- **AC10** Outputs: a CSV with one row per replicate (b, k_v8m, k_11l, the two recalls, the two FPs, Δ, reached or
  not), a JSON summary, a log and a sha256 sidecar. The verdict is accepted only as the single line printed by
  the script (`S1_VERDICT=<A|B|C|D|D0>`).

## §6 What is not done

- R\*, the verdict table, the seed, B and the operating-point rule are not changed after seeing the results. CIs
  are not attached to other R values to look for a "better" point.
- Reselecting the v1 or v2 τ, recomputing or reinterpreting existing endpoints, and editing or regenerating
  sealed files are prohibited.
- Retraining and new inference are prohibited. S1 is not applied to the nine v2 runs.
- S1 is not used as deployment performance, a recommended operating point or a conclusion about which model is
  better.
- The partial area and the generalisation gap (v1.1 §7.5 item 3) are not added later as "supplementary analyses".
