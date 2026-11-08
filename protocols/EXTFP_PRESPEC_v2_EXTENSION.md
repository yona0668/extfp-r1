# EXTFP scientific protocol extracts — EXTFP_PRESPEC_v2_EXTENSION.md

Local specification date: 2026-09-01.

Edited scientific extracts in English translation, prepared 2026-11-07. This reading copy is not byte-identical to the frozen original. Original and public file hashes are recorded in PROVENANCE.json.

## Specification timing

The replication was specified after all original results were known, including the original direction of difference. All nine replication runs were new. A feasibility probe had been observed; the new replication endpoints had not been computed.

## §V1 Question and estimand

What v1 answered: *"Two detectors with capacities within 2.1% of each other differed substantially in external
false-alarm burden in a single training run at seed 20260515."* This verdict is
**conditional on one training run and one seed per model**, and retraining variability was not measured. v2
measures those two gaps.

> **E1 (stability — confirmatory)**: Does the burden gap of the prespecified pair 0–1 replicate **when training is
> repeated (different seeds)**, with the same direction and a substantial magnitude?
> **E2 (spread — confirmatory)**: Among **3 capacity-matched (≤2.1%) detectors trained under the same protocol**
> (successive generations: YOLOv8m 2023 → YOLOv9c 2024 → YOLO11l 2024), does the between-model width
> (max − min) of external false-alarm burden exceed the MID?

- **estimand**: on the 1,154 negative images of the Mureșanu corpus, ⑴ E1: for the pair trained at seed r,
  Δ_r = FP/image(arm 0, seed r) − FP/image(arm 1, seed r), r ∈ {0, 1, 2}; ⑵ E2: max − min of the
  **model-level burden** (each a 3-seed mean) of the 3 capacity-matched models. Both values are **conditional on
  the selected τ and the trained checkpoints** (the same conditional interpretation as v1).
- **Non-questions — all prohibitions of v1.1 §1 and §7.4 are inherited**: the **causal effect** of generation,
  architecture or attention **is not asked** (one model per generation; implementation and training dynamics
  change together with each model). Population generalisation is not claimed (single external corpus). Agreement
  in direction across three or more pairs is not described as a "trend", "monotonic" or a "generation effect".
  Contrasts involving RT-DETR-l remain confounded pairs.
- **The maximum claim E2 can support** (fixed in advance): that external false-alarm burden differed by [D] among
  3 capacity-matched detectors trained under the same protocol — that is, at most the practical conclusion that
  **burden is a model-specific property that cannot be inferred from capacity or development-set performance**.
  "Which generation is superior" is not claimed, not even as a follow-on statement.

### k-fold cross-validation

k-fold cross-validation is not used: training (development corpus) and evaluation (the entire external corpus of
1,808 images) are separate, and the seed replication of E1 varies only training stochasticity while holding the
data, split and τ rule fixed.

**Resampling of the training data (fold training, bootstrap retraining) is not done either** — this is recorded
with its reasons in §V8.

### Why capacity matching is kept, and why "CNN vs transformer" is not done

**Capacity matching is not a causal-inference device; it is what supports the clinical claim.** Even with
matching, the architecture effect is not identified anyway (generation, loss, assignment, implementation and
pretraining move together, with n=1 per family). What matching does is to keep the compared models alike on their specification sheets (capacity within
2.1%), which reduces the difference in parameter count as an explanation of a divergence in external false
alarms; it does not exclude capacity-related effects (v1.1 §4.1).
Without matching, every new arm would become a confounded pair (exploratory), like RT-DETR, and could not enter
the verdict, and an observed gap would remain open to the alternative explanation of a difference in size; unmatched arms
would add only exploratory comparisons. **The
confirmatory core of E1 and E2 holds only on the matched set M.**

**The "CNN vs transformer" framework is not adopted** (v1.1 §4): ⑴ in this zoo, the only pure-convolution models
are YOLOv8 (C2f) and YOLOv9 (GELAN), and YOLO11 is a hybrid containing C2PSA, so **the dichotomy does not fit the
actual models**; ⑵ there is only one transformer detector family, RT-DETR (n=1), and that one model differs at
the same time in capacity (+30%), NMS-free design and one-to-one assignment, so these are **inseparable**; ⑶
filling both camps would require going outside Ultralytics, at which point the single implementation and single
frozen protocol (G2) would break. **The honest design for genuinely asking about the effect of attention is module
ablation** (a variant of YOLO11l with only C2PSA replaced by C2f, versus the original); because it requires a
non-canonical architecture, it is outside the scope of this study and is left to a **follow-up study**.

**Prohibited in advance — camp narrative.** M consists of two convolutional models (arms 0 and 4) and one
partial-attention model (arm 1). Even if the E2 result happens to split along that boundary (e.g., only arm 1 is
higher/lower), it is **not described as "because it is an attention family"** — it is an anecdote with n=1 per
cell.

---

## §V2 Arm and run matrix

### New arm 4 — YOLOv9c

| Arm | Model | Capacity reference | Relative to existing pair | Status |
|--:|---|--:|---|---|
| 0 | YOLOv8m | 25.90M | — | 1 sealed v1 run (outside the matrix) + **3 new runs** |
| 1 | YOLO11l | 25.37M | — | 1 sealed v1 run (outside the matrix) + **3 new runs** |
| 2 | *(vacant — YOLO12l, excluded by measurement in v1.1)* | 26.45M | — | **remains vacant** (record preserved) |
| 3 | RT-DETR-l | 32.97M | +27~30%, NMS-free | **not extended** — the 1 sealed v1 run is only cited, as exploratory |
| **4** | **YOLOv9c** | **25.59M** (architecture instantiation; to be confirmed with the value measured from the .pt at download) | **−1.2% (vs 0) · +0.87% (vs 1)** | **3 new runs** |

- **Capacity-matched set M = {arm 0, 1, 4}.** The largest pairwise gap is still the **2.1%** of 0–1 (arm 4 falls
  between them). E2 is defined only on M. Arm 3 is **excluded** from M because of the double confound of capacity
  and post-processing family (the same logic as v1.1 §4.1 and §7.2).
- **Why YOLOv9c.** ⑴ It is **the only other runnable generation that both falls within the capacity band (≤2.1%)
  and belongs to the same post-processing family as arms 0 and 1 (Detect head, standard NMS)** — YOLO12l (26.45M)
  matches in capacity but cannot be trained on this card (v1.1 §4); **YOLOv10l
  (25.89M) also matches in capacity but belongs to the NMS-free (one-to-one head) family**, which carries the same
  post-processing confound as arm 3, so it is not placed in M — placing it in M would mix post-processing families
  within the E2 spread and break the estimand "between-model width within the same post-processing pipeline";
  YOLOv9e (58.21M), YOLOv5l (53.23M) and YOLOv8l (43.69M) differ in capacity by close to a factor of two. These
  figures are parameter counts checked before the design. ⑵ It is supported by Ultralytics 8.4.42, so the
  "single implementation, single protocol" property of §5 is maintained. ⑶ It belongs to the GELAN family
  (RepNCSPELAN4, ADown, SPPELAN; no attention module) and has a Detect head, so the disabling of internal NMS plus
  the common NMS of §5.3 applies to it identically to arms 0 and 1.
- **Arm 2 remains vacant.** It is not filled in v2 either — the record of the hardware exclusion is preserved by
  its number.
- **Pretrained weights for arm 4**: `yolov9c.pt`, an official Ultralytics release asset (download URL, sha256 and
  parameter count recorded in the weights provenance record immediately upon download). COCO-80
  pretrained, under the same conditions as arms 0, 1 and 3. If the assets do not include `yolov9c.pt` or integrity
  verification fails, **arm 4 is excluded and the fallback of §V4 applies** (the weights are not obtained from
  unofficial sources).

### Run matrix — 9 new runs + 2 sealed v1 runs (anchors outside the matrix)

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| arm 0 YOLOv8m | new | new | new |
| arm 1 YOLO11l | new | new | new |
| arm 4 YOLOv9c | new | new | new |

- **Seed set {0, 1, 2}** — the set fixed at design. The three models share the same seed set (crossed design).
  Δ_r of E1 pairs runs of the same seed.
- **The 2 sealed v1 runs (arms 0 and 1, seed 20260515) are outside the matrix.** They are used only as the
  direction and magnitude anchor for E1 and for descriptive reporting, and **are not included in the model means
  of E2** — as a result, all 9 input runs of E2 are unknown at the time of the freeze (Specification timing).
  Sealed runs are not regenerated (§V8, G10).
- Run naming and output location: **the existing main root (`runs/extfp/`) is used as is**, with the seed embedded
  in the run id (`<arm>-s<seed>-<UTC-timestamp>-<pid>`; directories follow the existing pattern
  `{arm}_rung{rung}__{run_id}`). No separate `runs/extfp_v2/` root is created — this preserves the main/probe
  two-root structure of the output-policy gate (check_output_policy). The provenance JSONL uses the same
  append-only format as v1.

---

## §V3 Training — all provisions of v1.1 §5 apply; the seed is the only registered deviation

- **Carried over (0 changes)**: `imgsz 1280` · `epochs 100` · `patience 30` · `batch 4` (no automatic reduction;
  all §5.2 fail-closed guards) · `workers 4` · Ultralytics **8.4.42** · training data = 2,661 originals +
  photometric-perturbation copies (1,331 iterations per epoch) · horizontal flip only · `cache false` ·
  exclusive use of a single RTX 4090 card · environment `envs/yolo11` · no per-arm hyperparameter tuning ·
  all warm-up/accumulate provisions (§5.1).
- **Registered deviation — seed.** The `seed 20260515` of v1.1 §5 is replaced by a **per-run registered seed**
  (§V2 table). The registered set is **{0, 1, 2}** (not extended or replaced after seeing results — §V8). In the
  manifest, the fixed `common.seed` is replaced by a **registration scheme, `seeds.registered: [20260515, 0, 1, 2]`**,
  and the trainer checks ⑴ before training, whether the requested seed is registered, and ⑵ after training,
  that the trainer-state seed == the requested seed (G9). A `--seed` argument and an arm 4 entry are added to
  `scripts/train_extfp_arm.py`, and generation of seed-embedded run ids is added to the launcher, and **the
  transitive manifest is revised with the sha256 of the modified scripts** (§V10). When `--seed` is not given,
  the default remains 20260515, so as not to break v1 reproducibility.
- **Transitive manifest v2**: the sha256 of the list formed by adding ⑴ this document, ⑵ the modified training
  and guard scripts and ⑶ the `yolov9c.pt` sha256 to the list of v1.1 §5.0 is committed before main training
  begins. Gate G2 checks it for every run.
- **Unverified premise for YOLOv9c — stated.** Whether v9c can be trained with batch 4 on this card has not yet
  been measured (its GFLOPs are larger than those of v8m and 11l). Probe P2 of §V4 decides this, and the §5.2
  guards decide it again during main training. **Even if the probe or a guard fails, batch and imgsz are not
  changed** — only the fallback (§V4) applies.

---

## §V4 Feasibility probe P2 — arm 4, run before the freeze

Same procedure as P1 (RT-DETR-l). The order **commit this document (DRAFT) → run P2 → record the probe
observations → freeze** is followed. The P2 criteria are fixed now:

- 1 epoch, **seed 0**, otherwise the §V3 protocol unchanged (under `runs/extfp_probe/`, with the run id attached).
- **Pass conditions (all)**: `trained_batch == 4` · `oom_retries == 0` · 1,331/1,331 iterations ·
  peak reserved VRAM < 98% of 24,564 MiB · `spilled == false`.
- The probe's internal val metrics are information seen before the freeze — **they are not used for any design
  choice.** The probe weights are preserved but are not fed into main training, τ or any endpoint (G8).

**Fallback — on P2 failure or failure to obtain the weights (fixed in advance).** Arm 4 is excluded and v2 is
frozen as a **seeds-only reduced version**: the run matrix is arms 0 and 1 × 2 new seeds (4 runs), only E1
remains confirmatory and **E2 is not performed** (the spread of two models carries the same information as E1,
so it is not set up as a separate estimand). The reason for the exclusion is recorded in this document together
with the measured values, and the arm 4 number is preserved as vacant. In the reduced version, only the E1
statements of §V7 are used as conclusion statements.

---

## §V5 Inference, post-processing and operating point — all provisions of v1.1 §5.3 and §6 apply

- **Carried over (0 changes)**: raw storage at `conf ≥ 0.001` · internal NMS disabled (all of arms 0, 1 and 4;
  arm 4 also has a Detect head, so the same applies) · common class-agnostic NMS at IoU 0.7 · the order
  merge→NMS→sort→`max_det 300` · saturation reporting rule · score grid 0.001–0.999/0.001 · τ = per-run maximum
  box-level micro F1 on internal val (379 images) (matching IoU 0.50, greedy 1:1, class-agnostic, larger τ on
  ties) · no external re-selection · post-processing symmetry check (G6).
- **τ is per run.** Each of the 7 new runs selects its own τ on internal val — stability (E1) thus becomes a
  statement about the **whole pipeline** of "training → τ selection → deployment" (the same unit as deployment
  in practice).
- **New runs also store the raw predictions on the 379 internal val images at `conf ≥ 0.001`** — so that the
  τ-selection bootstrap of §V6-7 can run without new inference (same format and same post-processing as the
  stored external predictions).
- **Fixed-recall sensitivity rule — fully specified this time.** The selection rule that was a post hoc
  operationalisation in v1 is frozen with the same wording: *among grid τ satisfying internal val recall ≥ target
  (0.50, 0.70), the **largest τ***. Because recall is non-increasing in τ, this is the boundary point that
  satisfies the target from above. Its definition is the same as for the v1 values, so it is comparable across
  runs. Its status is **exploratory**, as in v1.

---

## §V6 Endpoints and verdicts

Common bootstrap settings (v1.1 §7.3.1 applies; only the seed is new): image-level **joint** resampling (one
resample is applied simultaneously to all relevant runs) · B = 10,000 · **seed 20260903** · percentile 95% CI ·
BCa not used.

### E1 — retraining stability of the prespecified pair (confirmatory)

For each registered seed r ∈ {0, 1, 2}:

- Δ_r = FP/negative image(arm 0, seed r) − FP/negative image(arm 1, seed r), 1,154 images paired, CI with the
  settings above.
- **Per-run tripartite verdict V_r** — the rule of v1.1 §7.3.2 is applied as is (MID 0.10):
  - **S_same**: substantial difference, in the same direction as v1 (Δ_r < 0, arm 0 lower)
  - **S_opp**: substantial difference, in the opposite direction
  - **EQ**: CI ⊂ [−0.10, +0.10]
  - **IND**: otherwise
- **Composite branches (fixed in advance)**:
  - **R1 (stable replication)**: **all three** V_r are S_same
  - **R3 (direction reversal)**: any V_r is S_opp
  - **R2 (partial/unstable)**: otherwise (EQ or IND mixed in, and no S_opp)

### E2 — spread of the three capacity-matched models (confirmatory)

- Model-level burden = **arithmetic mean** of FP/negative image over that model's 3 seed runs. D = max − min
  (3 models, M = {0, 1, 4}).
- CI: joint resampling (9 runs simultaneously) → model means and D recomputed for each resample → percentile 95%.
- **Tripartite verdict (fixed in advance)** — because D ≥ 0, the sign condition drops out:
  - **G1 (substantial spread)**: CI lower bound ≥ 0.10
  - **G2 (practical equivalence)**: CI upper bound < 0.10
  - **G3 (indeterminate)**: otherwise
- **Caution — this CI is also conditional**: because only images are resampled, it is conditional on "these 9
  checkpoints and τ", and seed variation is not contained in the CI (with only 3 seeds, this is a listing of
  observations, not a distribution estimate). The descriptive items below show that listing.
- **Predetermination — stated.** All 9 input runs of E2 are unknown at the time of the freeze, but the design was
  made with knowledge of the sealed v1 gap (−0.2114, seed 20260515), so **as long as that gap largely replicates
  at the new seeds, D ≥ MID is plausible**. The information value of E2 lies not in whether the boundary is
  crossed but in **the size of the width, its CI and the position of arm 4**.

### Descriptive (no verdict) — variation across seeds

- The individual values, range (max − min) and sample standard deviation of the 3-seed burdens of each model
  (0, 1, 4) are reported in a table.
- The 3-seed values of Δ(0−1) are listed, with the sealed v1 value (−0.2114; seed 20260515, outside the matrix)
  shown alongside (the original text reads "including the sealed −0.2114").
- **No test or verdict of any kind is attached** (n = 3). Their use is limited to presenting "between-model D
  versus within-model seed range" **side by side as numbers** — no ratio threshold is created (no arbitrary
  constants are added).

### Confirmatory family and multiplicity

- The confirmatory family = **the two questions {E1, E2}.** Because the verdicts are based on **magnitude (MID)**
  rather than significance, no formal multiple-comparison correction is applied, following the logic of v1.1
  §7.3. However, **as a sensitivity analysis, verdicts based on 97.5% percentile CIs are reported alongside** (a
  conservative check on claiming both questions simultaneously). The primary verdict is based on 95% — this
  priority is also fixed now.
- The three V_r of E1 (one per registered seed; the original text reads "two") are repeated verdicts asking "does it replicate in each case", and are not folded into a
  pooled statistic.

### Exploratory (prespecified, no verdict)

1. Pairwise contrasts for arm 4: Δ(4−0) and Δ(4−1) on a seed-mean basis, with CIs — **tabulated without a
   verdict** (the confirmatory pair remains 0–1 regardless of the results, inheriting v1.1 §7.2).
2. Per-run fixed-recall (0.50, 0.70) burden (§V5 rule).
3. Per-run FROC curves — the 3 seeds overlaid for each model (seed band).
4. Per-run lesion-level F1 and recall (654 positive images, same calculation as v1.1 §7.5-1).
5. Per-run τ, parameters and inference time — reported beside every operating point (§14 inherited).
6. Number of `max_det`-saturated images (per run and per operating point).
7. **τ-selection bootstrap (new — the variability of τ selection was not measured in v1).**
   For each run: resample the 379 internal val images at the image level (B = 10,000, **seed 20260904**, rerunning
   the entire τ-selection rule of §6 — maximum box-level micro F1, larger τ on ties — for every resample) → report
   per run the distribution of τ (2.5th, 50th and 97.5th percentiles) and the 2.5–97.5% interval of the
   distribution of the **external burden** at each resampled τ (looked up from the stored score grid, 0 new
   inference runs). **No verdict (exploratory)** — its use is limited to showing numerically "how conditional τ
   is on the val draw". New runs store internal val predictions at `conf ≥ 0.001` (§V5). For the 3 sealed runs,
   stored val predictions are used as is if they exist; otherwise new inference is run on the 379 val images with
   the sealed checkpoints, **read-only** — this is a new output, not a regeneration of sealed outputs (no conflict
   with G10), and this entire analysis is prespecified now, before any new v2 result exists.

---

## §V7 Conclusion statements — fixed before the results

**The v1 P1 statement is unchanged.** It is not revised under any v2 result.

### E1 axis (stability)

| Branch | Statement |
|---|---|
| **R1** | "This gap **replicated in retraining at all 3 registered seeds, in the same direction and with a magnitude exceeding the MID** — it is not chance in a single training run but a stable property of the two models under this protocol." |
| **R2** | "In retraining at new seeds, **the magnitude of the gap did not replicate consistently by the MID criterion** (there was no reversal of direction). Comparisons of external false alarms between detectors based on a single training run should be cautious about magnitude claims, and the v1 verdict remains an observation conditional on the seed 20260515 training run." |
| **R3** | "In retraining at new seeds, **the direction of the gap reversed in some cases.** A single-training comparison cannot determine even which model has the lower external false-alarm burden — without training repetition, detector benchmarks cannot support a ranking between models." |

R2 and R3 are reported in the same way as R1; they are not discarded results.

*Reading-copy note (2026-11-07):* "reversal of direction" in these statements means an S_opp verdict (§V6): a substantial
difference in the opposite direction. An opposite-sign difference whose verdict is EQ or IND is not a reversal
under this rule, and the R2 parenthetical is read in that sense.

### E2 axis (spread)

| Branch | Statement |
|---|---|
| **G1** | "Among **3 capacity-matched (≤2.1%) detectors** trained under the same protocol, external false-alarm burden was [min–max], and the between-model width [D] (95% CI …) exceeded the prespecified minimal important difference." |
| **G2** | "The external false-alarm burden of the 3 capacity-matched detectors lay **within a practically equivalent range** by the MID criterion — under this protocol and on this corpus, the burden does not change substantially whichever of these three models is chosen. This is an observation of three models and is not generalised to other models." |
| **G3** | "This sample was not sufficient to judge the between-model width by the MID criterion." (not to be restated as equivalence) |

**Combining the E1 and E2 verdicts (fixed in advance).** A combined conclusion is drawn only under R1 + G1 and
R2 + G1; under R1 + G1, the between-model D and the within-model seed ranges are reported side by side as numbers.
If E1 is R3 (with any E2 verdict), the R3 statement is the top-level conclusion, the spread is reported
descriptively only, and the practical conclusion is distrust of benchmarks without training repetition. Under
R1 + G3, the stability and indeterminate statements are given side by side, and the practical conclusion stays at
the v1 level (per-model measurement needed) without being extended to the spread. Under any other combination,
each axis's fixed statement is given side by side as is, and no combined claim is made.

### Prohibitions (all branches — inherited from v1.1 §7.4, plus v2 additions)

- No conversion to workload (prevalence unavailable, v1.1 §7.3).
- Changing the wording of the P1, R or G statements after seeing the results, changing the MID, seeds or run list,
  and restating indeterminate as equivalence — all prohibited.

---

## §V8 What is not done (v1.1 §9 inherited + v2 additions)

- All of v1.1 §9 is inherited (no training or τ selection on external data, no label correction, no per-arm
  tuning, no reader recruitment, no new data and no new IRB, no use of probe metrics for design, no workaround
  when a guard fails, etc.).
- **No regeneration of sealed runs.** The v1 checkpoints, predictions, τ and metrics are not recreated (G10).
- **No seeds are added after seeing results.** The run matrix is closed at the 9 runs of §V2 (6 runs under the
  fallback). "Just one more seed" does not happen under any result. Even if the results of some runs come to be
  inspected midway, the run list and decision rules remain unchanged, and **no v2 verdict is made before all
  registered runs are complete** (any inspection is recorded in the provenance).
- **The composition of E2 (M, mean, max−min) is not changed after seeing the results** (e.g., "let's switch to the
  median", "let's add arm 3 as well").
- **No k-fold cross-validation and no resampling of the training data (fold training, bootstrap retraining).**
  The reasons are given in §V1: evaluation is fixed to the external set; a model trained on a
  reduced training set (4/5) is not the model that would be deployed, so the estimand changes; and it conflicts
  with the frozen-split and τ discipline. Sensitivity to training-data composition is a component of variation
  that this extension does not measure and is treated as a limitation (seed replication measures only training
  stochasticity with the data fixed).
- **No reinterpretation of the v1 verdict.** Even if R2 or R3 occurs, the v1 statement is not revised
  retroactively (the conditional scope of the v1 verdict is kept distinct from the v2 retraining-stability
  result).

---

## §V9 Gates — v1.1 §10 applies + 2 new gates

G1 (duplicate sweep: the corpus is unchanged, so the v1 result is reconfirmed as valid) · G2 (checked against
transitive manifest v2, all 7 new runs) · G3 (internal origin of τ) · G4 (environment) · G5 (sha256 of this
document = sidecar v2.0 entry) · G6 (post-processing frozen and symmetric, including arm 4) · G7 (5 feasibility
checks, all new runs) · G8 (probe isolation, including P2) — all apply.

| New | Condition |
|---|---|
| **G9 seed provenance** | For each new run, **trainer-state seed == §V2 registered seed** (not args.yaml, for the same reason as in v1.1 §5.2). For the 2 sealed runs, seed == 20260515 is reconfirmed |
| **G10 sealed artifacts unchanged** | The sha256 of the sealed v1 artifacts (prediction JSONL, τ, metrics, checkpoints) matches the sealed record, and paths and command lines confirm that the v2 analysis consumed them **read-only** |

Gates are run in order; a failure is reported and then work stops — the same as in v1.1.

---

## §V10 Freeze procedure — scientific summary of the original

The execution order was: fixing the design, seed rules and implementation → recording the source and hash of the
official weights → the P2 feasibility probe and recording of its observations → fixing the hashes of the final
specification and the implementation manifest → main training.
At the freeze, new main training and new external inference had not started; the probe had already been run and
inspected.
Subsequent changes do not overwrite the existing specification and are recorded as separate
clarifications/amendments.
This paragraph is a summary of the original procedure; it is not a new freeze or an independent timestamp proof.
