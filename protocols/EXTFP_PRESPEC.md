# EXTFP scientific protocol extracts — EXTFP_PRESPEC.md

Local specification date: 2026-08-30.

Edited scientific extracts in English translation, prepared 2026-11-07. This reading copy is not byte-identical to the frozen original. Original and public file hashes are recorded in PROVENANCE.json.

## Specification timing

Version 1.1 was specified after a feasibility probe and one training run, but before external inference, threshold selection and the primary endpoint. The development study's detectors had already been evaluated for lesion detection, internally and on the 654 lesion-positive external images, and this study's training and probe measurements were known; no false-positive endpoint on the 1,154 lesion-negative images had been computed. The primary comparison has one confirmatory pair.

## §1 Question and estimand — scientific summary of the original

Using the public annotations as the reference, we measure the number of false positives per image for each detector on lesion-negative images, and the differences between detectors. The observed differences among the three detectors are compared; causal effects of attention or architecture are not estimated.
No re-reading by the authors is performed, and no claim is made about clinical truth or about generalisation to other populations.
If the distributed annotations missed lesions, the false-alarm burden may be biased upward; the size of this bias was not measured.

## §2 Source of the ground truth — the public reference standard is used as is (decision, 2026-08-29)

The 14-class annotations of the Mureșanu distribution are **adopted as is as the ground truth. We do not re-read them.**

- **class 7 (Apical periodontitis) box present = positive image** (654 images)
- **no class 7 box = negative image** (1,154 images)

### The basis and its limits are recorded together

Supporting evidence (full count): the 1,154 negative images are densely annotated for the remaining 13 classes, with **14,375 boxes in total, a median of 11 per image and 0 images with an empty label**.
That is, the absence of class 7 is an absence within that annotation, not a lack of annotation.

**However, "no annotation → no disease" is a statement the distributor never made; it is an inference we make.**
The distributor's ICC 0.91 has a single line of descriptive text as its source, and what it is a value for has not been confirmed — it is agreement among what was marked, not a measure of whether everything was found. Therefore:

1. **The primary endpoint is defined as "number of false positives per image against the public reference standard", not as "specificity".** This is a declaration that another party's reference standard is used as is, so it is a genuine external validation, and our own reading does not enter it.
2. **Uncertainty in annotation recall acts equally on all arms and therefore does not bias the primary comparison.**
   Only the absolute values are affected, and specificity is reported only as a **secondary metric** with this assumption stated.
   *Reading-copy note (2026-11-07):* this was a planning assumption, not a demonstrated property; the revised manuscript states (Limitations) that unannotated lesions counted as false positives could also penalise the detector that finds more of them.

### Alternative not adopted — fixing the negative set by author re-reading

Considered and **rejected.** ⑴ Correcting the labels with our own reading would change what is measured from "performance against the public reference standard" to "performance against our reading", and it would no longer be external validation. ⑵ A single reader, unblinded, with no κ, is a **weaker** reference standard than annotations calibrated by three annotators. ⑶ This study is a zero-reader design. The visual review of a 60-image sample is **not used** as an evaluation reference standard.

## §3 Data (frozen)

| Role | Source | Size |
|---|---|--:|
| Training | Reuse of the development study's **frozen split after duplicate removal** (Hanoi corpus, Mendeley `10.17632/kx52tk2ddj.3`, CC BY 4.0) | train **2,661** / val **379** |
| Internal test | test portion of the same split | **764** (all lesion-positive) |
| **External evaluation** | Mureșanu / Cluj-Napoca (Zenodo 15487430, terms of use: see the original source, ethics 117/04.06.2024) | **1,808** = positive 654 + **negative 1,154** |

- The external corpus is **for evaluation only**. It is not used anywhere in training, validation or threshold selection.
- The train/valid/test split names of the distribution are **theirs and unrelated to ours.** The 1,808 images are used as a single evaluation set, but `test_alte_cabinete` (180 images from another dental clinic) is reported separately as a **prespecified subgroup**.
- No redistribution of images. If images are shown in figures, the bottom strip is cropped.
- **Gate G1 — near-duplicate sweep**: training 2,661 images ↔ external 1,808 images are swept with dHash+pHash Hamming ≤ 12, and hits are confirmed with Pearson *r* ≥ 0.99. **If even a single hit occurs, the work does not start and this is reported.**

## §4 Arms — three prespecified detectors (frozen, revised in v1.1)

**The "CNN versus attention" dichotomy is not used, because it is factually incorrect.** This was confirmed by reading the Ultralytics architecture definitions directly: `yolo11.yaml` contains **C2PSA (partial self-attention)** and therefore cannot be a CNN arm, and `yolov10l.yaml` also has PSA. The only pure-convolution definition is `yolov8.yaml` (C2f + SPPF).

**In v1.1 the "attention ladder" construct itself was abandoned.** The ladder rested on the premise that C2PSA, A2C2f and AIFI are **ordinal steps of attention capacity**, but they are different modules and different detection families, so that premise does not hold. A monotonicity claim **could never have been made in the first place**.

| arm | Model | Attention mechanism | Head | Parameters | Status |
|--:|---|---|---|--:|---|
| **0** | **YOLOv8m** | none (C2f + SPPF) | Detect | **25.90M** | **training completed** (2026-08-30, Specification timing) |
| **1** | **YOLO11l** | **C2PSA** — local partial self-attention | Detect | **25.37M** | **newly trained** (retraining) |
| **3** | **RT-DETR-l** | **AIFI** transformer encoder + decoder | RTDETRDecoder (**NMS-free**) | 32.97M | **newly trained** |

- **Arm numbers 0, 1 and 3 are kept.** The slot of arm 2 (YOLO12l) is not refilled, so that the numbering records
  that the design had this slot and left it empty.
- **YOLO12l (arm 2) is excluded.** Reason: under the §5 protocol (imgsz 1280 · batch 4) it **cannot be trained**
  on the available hardware. This is a feasibility decision made before any external inference.
- **Arm 1, YOLO11l, is retrained.** The premise of v1.0 §4 that the development study's YOLO11l weights would be
  reused **no longer holds**: those weights are not in the repository (only an archive from before the re-split
  and the COCO initial weights remain), so no design can avoid retraining.
- **Each arm is one model.** No family- or generation-level generalisation ("attention models are …", "this
  generation is …") is made. Claims are limited to "in these three models …".
- The source, hash and parameter count of the pretrained weights are fixed in a separate weights provenance record
  (not included). All three files are Ultralytics release **v8.4.0**, COCO-80 pretrained.

### 4.1 Structure of the capacity contrasts — what is comparable and what is not (new in v1.1)

| Pair | Parameters | Capacity difference | Standing |
|---|---|--:|---|
| **0–1** (YOLOv8m ↔ YOLO11l) | 25.90M ↔ 25.37M | **0.53M = 2.1% of the smaller** | **primary comparison** — relatively close in capacity |
| **0–3** (YOLOv8m ↔ RT-DETR-l) | 25.90M ↔ 32.97M | 7.07M = **+27.3%** | **confounded secondary** |
| **1–3** (YOLO11l ↔ RT-DETR-l) | 25.37M ↔ 32.97M | 7.60M = **+30.0%** | **confounded secondary** |

- **Only 0–1 is a primary comparison with relatively close capacity.** Even so, the two are not regarded as "equal in capacity" — there is a 2.1% difference, and the two models differ in generation and implementation. What is measured is **the observed difference between two detectors, not the causal effect of attention**.
- **0–3 and 1–3 are confounded secondary comparisons that carry RT-DETR-l's parameter difference of about +30%.** In differences arising here, the architecture effect and the capacity effect are **not separated.** Moreover, RT-DETR-l belongs to an **NMS-free family**, so its post-processing properties also differ (§5.3 equalises them, but they do not become completely identical; the original text cites §5.1).
- **§7.4 implements this division of roles as a verdict rule.** Results of the confounded pairs are **not combined** with the primary evidence.

## §5 Training protocol — training identical to the development study, inference different (frozen, expanded in v1.1)

**Training**: `imgsz 1280` · `epochs 100` · `patience 30` · **`batch 4`** · `seed 20260515` · `workers 4` · Ultralytics **8.4.42** · training data = originals + one photometrically perturbed copy · **horizontal flip only (vertical flip prohibited, `fliplr 0.5` / `flipud 0.0`)** · `cache false` · environment `envs/yolo11` (`env_yolo11.txt`).

### 5.0 Batch, hardware and transitive protocol (new in v1.1)

§5 of v1.0 **never stated `batch`.** `batch=4` existed only in `PROTOCOL` of `scripts/train_extfp_arm.py` and in the development study's training arguments, so **the sha256 of this document alone did not fix the actual protocol.** This section closes that gap.

- **`batch = 4` (for every arm, single GPU).** This value is now **in this document.** It is not set differently per arm. Automatic reduction during execution (including Ultralytics' OOM retry) is **not permitted** — §5.2.
- **The hardware is frozen**: **a single NVIDIA RTX 4090 24,564 MiB card**; one arm occupies one card exclusively, and runs are not executed concurrently. `CUDA_DEVICE_ORDER=PCI_BUS_ID` + `CUDA_VISIBLE_DEVICES` are set before torch is imported. **DDP and multi-GPU are not used** (not executable on Windows/torch 2.5.1, and with `nbs=64` there is no effective-batch benefit either).
- **transitive protocol manifest.** What is frozen is not this document alone but the following list of sha256 values. The manifest is created and committed before main training starts, and gate G2 checks against it.

  | Item | File |
  |---|---|
  | Prespecification | `EXTFP_PRESPEC.md` (this document) |
  | Training scripts | `scripts/train_extfp_arm.py` · `scripts/run_extfp_training.py` · `scripts/extfp_guards.py` |
  | Data config | `configs/periapical_yolo.yaml` |
  | Split manifest | Hash of the file list of the development study's frozen split |
  | Weights | sha256 of `yolov8m.pt` · `yolo11l.pt` · `rtdetr-l.pt` (weights provenance record) |
  | Environment | `env_yolo11.txt` (python · torch · ultralytics · CUDA) |

  For every run, the trainer writes `source_sha256` (the hash of the source actually executed), the git SHA, the dirty status, the GPU UUID, the PID and argv to an append-only provenance JSONL. **`args.yaml` is not used as evidence** — it records only the *requested* batch (§5.2).

### 5.1 Effective batch and warm-up (new in v1.1)

Ultralytics sets `accumulate = max(round(nbs / batch), 1)` with `nbs = 64` as the reference.
Therefore at batch 4 the **optimizer effective batch is 64**, with about 83 optimizer steps per epoch.

**During warm-up, however, accumulate is interpolated from 1 → `nbs/batch`.** Because `warmup_epochs = 3.0`, warm-up lasts `3 × 1,331 = 3,993` iterations.

> **The `accumulate = 6` and effective batch 24 recorded by the P1 probe are artefacts of a 1-epoch probe.**
> The probe ended at 1,331 iterations and left values from **the middle of warm-up**. They are not a property of the arm.
>
> **Rule**: **if an effective-batch comparison between arms is needed, values after the end of warm-up
> (iteration > 3,993) are used.** The expected values at that point are `accumulate = 16` and effective batch 64,
> and gate G2 checks against these values. Values during warm-up are not used for arm comparisons or for judging
> protocol identity.

### 5.2 Execution guards — fail-closed, no automatic reduction (new in v1.1)

Main training of each arm is valid only if it satisfies **all** of the following. If any one deviates, **STOP
on the spot** and report.

| Check | Pass condition | Rationale |
|---|---|---|
| Batch | **`trained_batch == 4`** (read from the trainer state) | `args.yaml` records the requested value and therefore cannot serve as evidence |
| OOM retries | **`oom_retries == 0`** | On OOM, Ultralytics halves the batch and leaves only a one-line warning |
| iteration | Iterations per epoch in the log match **`ceil(5322/4) = 1,331`**, and neither `2,661` nor `666` appears alongside | If the batch changes midway, the denominator changes |
| Memory | Peak reserved VRAM **< 98% of card capacity**, `spilled == false` | Spill-over into host memory raises no warning |
| Log binding | The log carries **the run id marker of this run** (checked twice, before and after training) | Prevents a passing log from another run being presented as evidence |

> **Meaning of STOP (prespecified)**: if a check above fails, **the batch is not reduced, the protocol is not
> changed, and work does not move on to another arm.** The failure and its provenance are recorded, work stops,
> and the investigators decide how to proceed. If the conclusion is that a protocol change is needed, that is a **v1.2 amendment**,
> made in writing before external inference.

**Unverified premise — stated explicitly (v1.1).** **Whether YOLO11l can be trained at batch 4 on this hardware had not
been measured.** The guards above decide, and the STOP rule above applies on failure.

### 5.3 Inference and post-processing (frozen, double NMS corrected in v1.1)

**Inference differs from the development study — this is an intended deviation.** The development study saved its outputs truncated at `--conf 0.05`; that makes the range below 0.05 **unrecoverable**, so the full-range τ curves promised by §6 and §7.5 could not be drawn, and the possibility that an arm's F1-optimal τ lies below 0.05 could not be excluded in advance either.

| Item | Fixed value | Reason |
|---|---|---|
| Storage threshold | **`conf ≥ 0.001`** | For full-range τ curves and τ re-selection to be possible, low-threshold candidates must remain |
| `max_det` | **300 / image** | Storing at a low threshold inflates the number of outputs, so the cap is fixed |
| **Disabling internal NMS** | **For the YOLO family (arms 0 and 1), the predictor's internal NMS is effectively disabled and the raw candidates are taken** — `iou = 1.0` (no suppression), `agnostic = True`, and `max_det` set generously inside (≥ 3,000) so that no truncation occurs internally | **Correction of a v1.0 defect.** The Ultralytics detect predictor already applies class-specific NMS and `max_det` **before** building `Results`. Left as is, arms 0 and 1 would receive **double NMS** (internal class-specific + external class-agnostic) while arm 3 would receive single processing, breaking "identical post-processing" |
| **Common NMS** | **class-agnostic, IoU 0.7**, applied **identically afterwards** to the **raw candidates** of the three arms | The endpoint is a count of false positives, yet **RT-DETR-l is NMS-free** (`RTDETRDecoder`). Without equalising post-processing, architectural differences become entangled with the duplicate-suppression policy |
| Merge order | ① class-agnostic merging → ② common NMS (IoU 0.7) → ③ sort by score, descending → ④ truncate at `max_det 300` | The order changes the counts, so it is fixed as a string |
| score grid | **from 0.001 to 0.999 in steps of 0.001** | τ selection and the curves are computed on the same grid |
| τ selection range | the whole grid above (the lower bound is not set at 0.05) | The possibility that an arm's optimum lies below 0.05 is not excluded |

**`max_det` saturation rule (new in v1.1).** `max_det = 300` applies **only to step ④ of the pipeline above, not to the
`conf ≥ 0.001` raw storage stage.** If truncation at ④ actually occurred on any image (= the image had more than 300
candidates), the number of such images is **reported per arm in the logs and tables.** The decision rule is as follows.

- If **0 images are saturated at the operating point τ**, there is no problem — the primary endpoint counts only boxes at or above τ, so it is unaffected.
- If **even one image is saturated at the operating point τ**, that arm's FP/image is a **downward-truncated (censored)
  value**. This is not hidden: it is reported as "≥", **together with the number of saturated images**. The cap is not raised
  and the run is not repeated (that would make post-processing differ between arms).
- Saturation in the low-threshold range (the FROC tail where τ is very low) is expected, and that part of the curve is drawn
  **with a truncation mark**.

Raw predictions (JSONL) are preserved in their `conf ≥ 0.001` state, and every aggregate must be re-derivable from this file.

**No hyperparameter other than the architecture is tuned per arm.** It is not tuned even if an attention arm is disadvantaged — the moment it is tuned, the comparison becomes "A as tuned by us vs untuned B". The full resolved hyperparameters are preserved per arm in a separate training configuration record (not included).

## §6 Operating point — the threshold is not set by looking at external data (frozen, fully specified in v1.1)

**τ is selected per arm by maximising F1 on the internal val set (379 images), and that value is held fixed for use on external data.**
τ is not re-selected with external data. As a result, the threshold-selection optimism problem that the development study had to address **does not arise** for the primary endpoint.

§6 of v1.0 consisted of that one sentence only. **What F1 is, how the three classes are pooled, what the matching IoU is,
and which τ is chosen in a tie were all left blank.** v1.1 fills them in.

### 6.1 class mapping — the three classes are pooled into one

- The internal training and validation labels have **three classes** (`class_3` · `class_4` · `class_5` = PAI 3/4/5), and the corresponding label in the external
  Mureșanu corpus is a **single class** (class 7, Apical periodontitis).
- **Both τ selection and the primary endpoint are computed after pooling into the single class `lesion` (class-agnostic).**
  The class labels of predicted boxes are **ignored**, and only the scores are used. Reference boxes also ignore class.
- This pooling follows **the same rule** as the class-agnostic merging and NMS of §5.3, and it is identical internally and externally.
- Performance by PAI grade is outside the scope of the study and is not reported, even as a secondary metric.

### 6.2 Definition of F1 — box level, class-agnostic, micro

- **Matching IoU threshold = 0.50** (a single value; no average over multiple thresholds and no mAP).
- Matching is **greedy one-to-one in descending score order** — starting from the highest-scoring prediction, each is paired with the not-yet-matched reference box
  of largest IoU; if IoU ≥ 0.50 it is a TP, otherwise an FP. Each reference box is matched at most once.
  Reference boxes left over are FN.
- **micro F1** — computed from TP/FP/FN **summed over all** 379 internal val images (not the mean of per-image F1).
  `F1 = 2TP / (2TP + FP + FN)`.
- The τ grid is the same as in §5.3 (0.001–0.999, steps of 0.001).
- **Tie-break**: if several τ attain the maximum F1, the **largest τ** is chosen. Rationale: because this study's
  primary endpoint is the number of false positives, choosing the **more conservative operating point (the one that detects less)** in a tie
  does not create bias in the direction of the endpoint. If still tied, the **smallest index** (grid order).

### 6.3 What this τ is selected on — stated explicitly

> **The 379 internal val images are all lesion-positive (0 empty labels).**

**In a study whose primary endpoint is false positives, τ is selected on a set that contains not a single negative image.**
This is not a defect but **an unavoidable property of the design**, and it is handled openly, as follows.

1. **Even so, τ is not re-selected on external data.** Selecting it externally would set the endpoint by looking at the endpoint itself
   and make the primary endpoint meaningless. Avoiding the optimism problem is preferred.
2. **The implication has a fixed direction**: maximising F1 on a positive-only set tends to lean towards recall, and such a τ produces
   **more false positives** on negative images. That is, **the primary endpoint may be biased
   upward.** This is **the same direction** as the label-omission bias of §2.
3. **This property acts identically on all three arms** — all of them choose τ on the same set by the same rule.
   It is therefore considered **not to push the pairwise comparisons (§7.2) systematically to one side**. However, this too is an
   assumption (of the same kind as the corresponding item in §2).
4. **Sensitivity analysis (secondary, prespecified)**: FP/image with τ set not by maximum F1 but by **fixing recall at 0.50 and 0.70**
   is reported as a supplement. The primary verdict is not changed.

As a secondary item, full-range τ curves (FROC form: false positives per image vs lesion recall) are reported overlaid across arms.

## §7 Endpoints and conclusion rules (frozen, §7.2, §7.3 and §7.4 rewritten in v1.1)

### 7.1 Primary — descriptive: external false-alarm burden

Reported per arm on the **1,154** negative images (= images without a class 7 box). **All values are relative to the public reference standard.**

- **False positives per image (FP/image)** — the mean is given together with the **median and IQR**. Because a few images can dominate the total, the result is not described by the mean alone.
- **Proportion of images with at least one false positive** (= 1 − image-level specificity). Reported with the assumption of §2 stated.
- Table of the FP count distribution (0, 1, 2, 3+).
- Definition: on a negative image, **every lesion box output at or above τ is a false positive**. The hard-negative distinction of §8 is
  not applied in the primary endpoint.
- If `max_det` saturation occurred, it is reported as "≥" under the truncation rule of §5.3.

**The primary descriptive results remain however the comparisons of §7.2–§7.4 turn out.** They are the centre of this study.

### 7.2 Comparison — the role of each pair and its strength of evidence are fixed first (rewritten in v1.1)

- Image-level **paired** comparison (all three arms are applied to the same images).
- 95% CI of the difference from **10,000 bootstrap** image-level resamples (construction in §7.3).

**The three pairs do not have the same standing. This is fixed in advance.**

| Pair | Standing | Strength of evidence | Correction |
|---|---|---|---|
| **0–1** YOLOv8m ↔ YOLO11l | **primary comparison** (confirmatory) | The parameter counts of the two models differ by 2.1% and are **relatively close** (§4.1). The only pair this study adjudicates | Confirmatory family size **1** → no multiple-comparison correction needed (§7.3) |
| **0–3** YOLOv8m ↔ RT-DETR-l | **confounded secondary** (exploratory) | +27.3% capacity difference · NMS-free family. The architecture effect and the capacity effect are **not separated** | Not corrected. **Not adjudicated** |
| **1–3** YOLO11l ↔ RT-DETR-l | **confounded secondary** (exploratory) | +30.0% capacity difference · same as above | Same as above |

**Prespecified prohibitions (enforced by §7.4):**

- **Results of the confounded pairs are not combined with the primary evidence.** "A difference appeared in two of the three pairs" or
  "the directions all agree" is **not used as grounds.** Each pair has a different confound, so combining them does not hold.
- **Even if the directions of the three pairs line up, this is not taken as a monotonic change, a ladder or a trend across generations.**
  v1.0 had such rules (§7.4 A/B/C/D), and **v1.1 deleted them.**
- **No causal effect of attention or architecture is claimed.** Not for any result, and not even for the primary pair.
  0–1 is a pair of two detectors that differ wholesale in generation and implementation, not a contrast in which only the attention module was changed.
- **The standing of a pair is not changed after seeing the results.** The primary pair is 0–1 now and remains 0–1 to the end.

**Reporting obligation**: the differences and CIs of all three pairs are given in a table. Standing (confirmatory/exploratory) and capacity difference (%) are
stated **in the same table**, so that a reader can tell from the table alone what each pair carries.

### 7.3 Verdict by magnitude — MID, adjusted CI and tripartite verdict (CI definition and verdict logic corrected in v1.1)

With n=1,154 paired images the CIs are narrow, so **even clinically meaningless differences become significant**. Verdicts are therefore made by
magnitude, not by significance.

> **MID = 0.10 FP per lesion-negative image.** This value is defined **on the negative-conditional scale** —
> "one false alarm per 10 images without a lesion".

**No conversion to a daily clinic burden is made.** Up to v0.3 this was written as "reading 100 images a day gives 10 per day", but
**that is an arithmetic error.** The actual number of additional false alarms is `100 × (proportion of lesion-negative images) × 0.10`, and this corpus,
being a **negative-conditional design**, does not provide clinical prevalence. In a clinic with prevalence p, the additional marks per 100 images are
`10 × p` — without knowing p the conversion cannot be made, and we do not have p. Values are reported **only on the measurement scale**, and the conversion is left to the reader.
*Reading-copy note (2026-11-07):* with p as prevalence, the expression consistent with the preceding sentence is `10 × (1 − p)`; the original text reads `10 × p`. No conversion is made, so no reported value depends on this expression.

**The MID has no external standard;** its only basis is that it was fixed in advance.
The claim "below this, the workflow would not change" is unsupported and is therefore **not made.**
The MID is not changed after seeing the results.

#### 7.3.1 Definition of the "adjusted CI" (new in v1.1)

v1.0 built the tripartite verdict on an **"adjusted CI"** without stating what it was. Holm is a p-value
procedure, so no corresponding interval follows from it automatically. It is fixed as follows.

| Item | Fixed value |
|---|---|
| Resampling unit | **Image** (the 1,154 negative images). In each resample the values of **the three arms are drawn together** (joint/paired resampling) — the same image set is applied to the three arms simultaneously |
| Repetitions | **B = 10,000** |
| seed | **20260830** (fixed; a rerun must give identical intervals) |
| Interval type | **percentile** bootstrap CI (BCa is not used — there is no basis for assuming that the acceleration and bias corrections are stable for this endpoint, and the simpler the rule, the more binding the prespecification) |
| **primary pair (0–1)** | The confirmatory family size is **1**, so no multiple-comparison correction is needed. **"Adjusted CI" = the 95% percentile CI as is**, and this is the input to the verdict of §7.3.2 |
| **secondary pairs (0–3 · 1–3)** | Uncorrected 95% percentile CIs are reported as **exploratory**. **They do not enter the verdict** |
| Holm | Because the confirmatory family is a single pair, **there is nothing to apply it to.** The three-pair Holm provision of v1.0 was deleted in v1.1 |
| Secondary metrics | The same bootstrap settings are used for the metrics of §7.5, but all of them are **exploratory** and no verdict is made on them |

#### 7.3.2 Tripartite verdict (condition for "difference" corrected in v1.1)

Applied only to the primary pair (0–1). Δ = the difference in FP/image between the two arms of the pair.

| Verdict | Condition |
|---|---|
| **Substantial difference** | The adjusted CI excludes 0 and **the entire CI lies outside the equivalence range** — that is, the two endpoints of the CI have the same sign and **the absolute value of the endpoint nearer 0 is ≥ MID** |
| **Substantial equivalence** | **The entire adjusted CI lies within [−0.10, +0.10]** |
| **Verdict withheld (indeterminate)** | Neither of the above — the CI straddles an MID boundary and excludes neither |

**The v1.0 condition ("CI excludes 0 **and** point estimate |Δ| ≥ MID") was defective and is corrected.** That rule would
pass a result such as point estimate 0.11, CI [0.01, 0.20] as "substantial difference" — yet that CI **does not exclude effects
below the MID**, such as 0.02. That magnitude cannot be claimed from a point estimate holds for equivalence and for
difference alike.

**A "withheld" verdict arising from an insufficient sample is an honest result, and it is not reported as "no difference".**

### 7.4 The conclusion statement is fixed before the results (rewritten in v1.1)

Verdicts are made **only on the primary pair 0–1**. **0–3 and 1–3 contribute to no verdict.**
**In every case, the descriptive results of §7.1 remain.**

| Result | Condition (0–1) | Conclusion to be stated |
|---|---|---|
| **P1** Difference | the adjusted CI meets "substantial difference" of §7.3.2 | **"Even between two detectors whose capacities are within 2.1% of each other, the false-alarm burden on external negative images differed substantially (≥ MID)."** The direction, i.e. which one was lower, is stated, but **the cause is not attributed to architecture or attention** — these are two models in which generation, training dynamics and implementation changed together |
| **P2** Substantial equivalence | adjusted CI ⊂ [−MID, +MID] | **"Between these two detectors the external false-alarm burden was substantially equivalent — changing the backbone within this range did not change the burden."** However, this is **an observation on a single pair** and is not generalised to other architecture pairs |
| **P3** Withheld | otherwise | **"This sample was not sufficient to judge the size of the difference between the two detectors against the MID."** It is not rewritten as "no difference" |

**Rules for describing the confounded pairs (0–3 · 1–3) — separate, and not combined with the verdict above:**

- Differences and CIs are reported **descriptively only**. No claim of significance is made, and the magnitude relative to the MID is mentioned
  **for reference only**.
- Whatever direction or magnitude results, **because RT-DETR-l has about 30% more parameters and belongs to an NMS-free family, the effect
  of architecture cannot be separated from the effect of capacity in these differences.**
- Even if the three pairs all point in the same direction, this is **not taken as a consistent trend, monotonicity or a generation effect.**

**Not to be done**: changing the wording of P1–P3 after seeing the results, adjusting the MID, adding or excluding arms,
changing the standing (primary/secondary) of a pair, rewriting withheld as equivalence, or reinforcing the primary conclusion with the
confounded pairs.

### 7.5 Secondary metrics (prespecified, fixed order)

All are **exploratory** and do not enter the verdict of §7.4.

1. Lesion-level F1 / recall on the 654 positive images
2. Full-range τ curves — false positives per image vs lesion recall (FROC form), arms overlaid
3. **Generalisation gap** from internal test 764 → external 654 (per arm)
4. `test_alte_cabinete` 180-image subgroup
5. **Parameter count and inference time** — reported beside every operating point (§14)
6. τ sensitivity of §6.3 (recall fixed at 0.50 and 0.70)
7. Number of images with `max_det` saturation from §5.3 (per arm and per operating point)

## §8 Hard negatives — decided before seeing the results (frozen)

Of the 1,154 negative images, **872 (75.6%)** carry findings that can be confused with a periapical radiolucency (Endodontic
treatment 660 · Bone resorbtion 358 · Root fragment 216 · Furcation lesion 85 ·
Apical surgery 38 · Root resorption 4).

- **The primary endpoint does not distinguish them.** Every detection is a false positive. This is a conservative and simple rule,
  and the between-arm comparison does not need this distinction.
- **Prespecified decomposition (secondary)**: false positives are reported split into ⑴ those with IoU > 0 against a confusable-class box and ⑵ all others.
  **Nothing is removed from the denominator.** This decomposition is descriptive and changes no primary conclusion.
- The number of false positives on the 282 clean negative images alone is reported as a **subgroup**.

## §9 What is not done

- Training, fine-tuning or threshold selection with external data — all prohibited
- Label re-reading or label correction (§2)
- Per-arm hyperparameter tuning (§5)
- Adding or changing endpoints, rules or arms after seeing the results
- Reader recruitment, new data acquisition, new IRB
- **Using internal val metrics from the feasibility probe or the training logs for design choices** — they are not used for arm selection, τ, the number of epochs,
  augmentation, hyperparameters, early-stopping rules, conclusion rules or the MID (Specification timing)
- **Reducing the batch or changing the protocol to continue after a guard failure** — STOP and report (§5.2)
- **Reinforcing the primary conclusion with the confounded pairs**, or treating aligned directions of the three pairs as monotonicity (§7.4)

## §10 Execution gates (v1.1: G2 rewritten · G6 extended · G7 and G8 added)

| Gate | Condition |
|---|---|
| G1 | Training set ↔ external set near-duplicate sweep hits **0** (§3) |
| **G2** | **Confirmation from the trainer state and provenance that all three arms were trained with the same protocol.** Items checked: `trained_batch == 4` · `oom_retries == 0` · iterations per epoch `1,331` · `accumulate == 16` and effective batch `64` **after the end of warm-up** (§5.1) · `source_sha256`, weight sha256 and environment match the transitive protocol manifest of §5.0. **`args.yaml` is not used as evidence** — it records only the requested batch |
| G3 | Confirmation from command lines and logs that no external data were used for τ selection. **τ must have been derived only by the rule of §6 (class-agnostic · IoU 0.50 · micro F1 · larger τ in a tie)** |
| G4 | Environment `check_environment.py --stage detector` PASS |
| G5 | The sha256 of this document matches the **v1.1 entry** of the sidecar (`EXTFP_PRESPEC.sha256`). The sidecar's **v1.0 entry** matches the preserved file `EXTFP_PRESPEC_v1.0.md` |
| **G6** | **Confirmation that the §5.3 post-processing is frozen** — raw predictions preserved at `conf ≥ 0.001`; **internal NMS of arms 0 and 1 disabled** (prediction call arguments `iou=1.0` · `agnostic=True` · internal `max_det ≥ 3000` confirmed from command lines and logs); the same class-agnostic NMS (IoU 0.7) applied to all three arms in the order of §5.3; and the number of images with `max_det 300` truncation recorded per arm. **Post-processing symmetry check**: confirm that the post-processing code paths of the three arms went through the same function |
| **G7** | **Feasibility gate (new).** Main training of each arm **passed all** five checks of §5.2 — batch 4 · OOM retries 0 · iterations 1,331 · peak reserved VRAM < 98% of the card (`spilled == false`) · log–run id binding. **If any one fails, STOP**; no workaround by batch reduction or protocol change |
| **G8** | **Probe isolation (new).** Confirmation from paths and command lines that the outputs of `runs/extfp_probe/` (weights, predictions, metrics) were **not used as input anywhere** in main training, τ selection, external inference or endpoint computation. The probe weights are preserved, not deleted, but are not used |

**The gates must be passed in order, and a failure means report and stop.** No gate is passed over with "fix it later and
continue".

## §14 Parameter asymmetry — remains only at rung 3 (rewritten in v1.1 for the three arms)

The parameters of the three arms are **25.90 / 25.37 / 32.97M**. 0–1 differ by **0.53M (2.1% of the smaller)** and are
relatively close, while the two pairs involving RT-DETR-l carry differences of **+27.3% and +30.0%** (§4.1).

Interpretation is limited accordingly. Only the 0–1 pair is adjudicated. Wherever RT-DETR-l differs from the other two, an effect of architecture cannot be separated from an effect of capacity or of post-processing, so those two contrasts are exploratory and are not combined with the primary one. No unofficial intermediate scale is constructed to match RT-DETR-l, because it would be canonical to no family. Even for the primary pair no causal effect of attention or of architecture is claimed, because the two models also differ in generation, training dynamics and implementation. Parameter count and inference time are reported beside every operating point.

**The pair-standing table of §7.2 and the verdict rules of §7.4 implement this** — only 0–1 is adjudicated, and the pairs involving rung 3
are reported as exploratory, with the confound attached.
