# EXTFP scientific protocol extracts — EXTFP_PRESPEC_v1.0.md

Local specification date: 2026-08-29.

Edited scientific extracts in English translation, prepared 2026-11-07. This reading copy is not byte-identical to the frozen original. Original and public file hashes are recorded in PROVENANCE.json.

## Specification timing

Original specification. The development study's detectors had already been evaluated for lesion detection, internally and on the 654 lesion-positive external images. No false-positive endpoint on the 1,154 lesion-negative images had been computed. This initial version was superseded by v1.1 before external inference.

## §1 Question and estimand — scientific summary of the original

Using the public annotations as the reference, we measure the number of false positives per image for each detector on lesion-negative images, and the differences between detectors. The initial comparison question was the difference in burden between the rungs of the attention ladder. This configuration was abandoned in v1.1 before external inference.
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

## §4 Arms — the attention ladder (frozen)

**The "CNN versus attention" dichotomy is not used, because it is factually incorrect.** This was confirmed by reading the Ultralytics architecture definitions directly: `yolo11.yaml` contains **C2PSA (partial self-attention)** and therefore cannot be a CNN arm, and `yolov10l.yaml` also has PSA. The only pure-convolution definition is `yolov8.yaml` (C2f + SPPF).
The arms are therefore constructed as a **ladder of degree of attention**.

| Rung | Model | Attention mechanism | Parameters | Status |
|--:|---|---|--:|---|
| **0** | **YOLOv8m** | none (C2f + SPPF) | **25.90M** | newly trained |
| **1** | **YOLO11l** | **C2PSA** — local partial self-attention | **25.37M** | development-study weights **reused** |
| **2** | **YOLO12l** | **A2C2f** — area-attention-centred backbone | **26.45M** | newly trained |
| **3** | **RT-DETR-l** | **AIFI** transformer encoder + decoder | 32.97M | newly trained |

## §5 Training protocol — training identical to the development study, inference different (frozen)

**Training**: `imgsz 1280` · `epochs 100` · `patience 30` · `seed 20260515` · Ultralytics **8.4.42** · training data = originals + one photometrically perturbed copy · **horizontal flip only (vertical flip prohibited)** · environment `envs/yolo11` (`env_yolo11.txt`).

**Inference differs from the development study — this is an intended deviation.** The development study saved its outputs truncated at `--conf 0.05`; that makes the range below 0.05 **unrecoverable**, so the full-range τ curves promised by §6 and §7.5 could not be drawn, and the possibility that an arm's F1-optimal τ lies below 0.05 could not be excluded in advance either. Inference and post-processing for this study are **frozen separately in §5.1.** The reused YOLO11l weights are also **re-run for inference** here (the weights are unchanged; the predictions are newly produced).

### 5.1 Inference and post-processing (frozen)

| Item | Fixed value | Reason |
|---|---|---|
| Storage threshold | **`conf ≥ 0.001`** | For full-range τ curves and τ re-selection to be possible, low-threshold candidates must remain |
| `max_det` | **300 / image** | Storing at a low threshold inflates the number of outputs, so the cap is fixed. Whether it is reached is reported from the logs |
| **Common NMS** | **class-agnostic, IoU 0.7**, applied **afterwards** to the raw outputs of all arms | The endpoint is a count of false positives, yet **RT-DETR-l is NMS-free** (`RTDETRDecoder`). Without equalising post-processing, the attention effect becomes entangled with the duplicate-suppression policy. It is harmless for outputs that already contain no duplicates |
| Merge order | ① class-agnostic merging → ② NMS → ③ sort by score, descending → ④ truncate at `max_det` | The order changes the counts, so it is fixed as a string |
| score grid | **from 0.001 to 0.999 in steps of 0.001** | τ selection and the curves are computed on the same grid |
| τ selection range | the whole grid above (the lower bound is not set at 0.05) | The possibility that an arm's optimum lies below 0.05 is not excluded |

Raw predictions (JSONL) are preserved in their `conf ≥ 0.001` state, and every aggregate must be re-derivable from this file.

**No hyperparameter other than the architecture is tuned per arm.** It is not tuned even if an attention arm is disadvantaged — the moment it is tuned, the comparison becomes "A as tuned by us vs untuned B". The full resolved hyperparameters are preserved per arm in a separate training configuration record (not included).

## §6 Operating point — the threshold is not set by looking at external data (frozen)

**τ is selected per arm by maximising F1 on the internal val set (379 images), and that value is held fixed for use on external data.**
τ is not re-selected with external data. As a result, the threshold-selection optimism problem that the development study had to address **does not arise** for the primary endpoint.

As a secondary item, full-range τ curves (FROC form: false positives per image vs lesion recall) are reported overlaid across arms.

## §7 Endpoints and conclusion rules (frozen)

### 7.1 Primary — descriptive: external false-alarm burden

Reported per arm on the **1,154** negative images (= images without a class 7 box). **All values are relative to the public reference standard.**

- **False positives per image (FP/image)** — the mean is given together with the **median and IQR**. Because a few images can dominate the total, the result is not described by the mean alone.
- **Proportion of images with at least one false positive** (= 1 − image-level specificity). Reported with the assumption of §2 stated.
- Table of the FP count distribution (0, 1, 2, 3+).
- Definition: on a negative image, **every lesion box output at or above τ is a false positive**. The hard-negative distinction of §8 is not applied in the primary endpoint.

### 7.2 Primary — comparative: differences between ladder rungs

- Image-level **paired** comparison (all four arms are applied to the same images).
- 95% CI of the difference from **10,000 bootstrap** image-level resamples.
- **The confirmatory contrast family = only the 3 capacity-matched pairs: `0–1`, `1–2`, `0–2`.** **Holm correction** is applied to these three only.
  Because the three models are 25.37–26.45M (±2%), these are the only contrasts in which an attention effect can be read separately from capacity.
- **Contrasts involving rung 3 (`2–3`, `0–3`) are exploratory and are not included in the Holm family.** They are reported with CIs, but
  **no conclusion about attention is drawn from those contrasts alone** (§14). Because the reason for excluding them from the confirmatory family is
  the same as the capacity confound of §14, the test family and the interpretation priority become **a single rule**.
- Up to v0.3, the Holm family (`0–1, 1–2, 2–3, 0–3`) and the interpretation priority (`0–1, 1–2, 0–2`) differed —
  `0–2`, which was called central, was not in the test list. This item corrects that.

### 7.3 Verdict by magnitude — MID and the tripartite verdict

With n=1,154 paired images the CIs are narrow, so **even clinically meaningless differences become significant**. Verdicts are therefore made by magnitude, not by significance.

> **MID = 0.10 FP per lesion-negative image.** This value is defined **on the negative-conditional scale** —
> "one false alarm per 10 images without a lesion".

**No conversion to a daily clinic burden is made.** Up to v0.3 this was written as "reading 100 images a day gives 10 per day", but
**that is an arithmetic error.** The actual number of additional false alarms is `100 × (proportion of lesion-negative images) × 0.10`, and this corpus,
being a **negative-conditional design**, does not provide clinical prevalence. In a clinic with prevalence p, the additional marks per 100 images are
`10 × p` — without knowing p the conversion cannot be made, and we do not have p. Values are reported **only on the measurement scale**, and the conversion is left to the reader.
*Reading-copy note (2026-11-07):* with p as prevalence, the expression consistent with the preceding sentence is `10 × (1 − p)`; the original text reads `10 × p`. No conversion is made, so no reported value depends on this expression.

**The MID has no external standard;** its only basis is that it was fixed in advance.
The claim "below this, the workflow would not change" is unsupported and is therefore **not made.**

**Tripartite verdict (fixed in advance)** — based on the adjusted CI:

| Verdict | Condition |
|---|---|
| **Substantial difference** | adjusted CI excludes 0 **and** point estimate \|Δ\| ≥ MID |
| **Substantial equivalence** | **the entire adjusted CI lies within [−0.10, +0.10]** |
| **Verdict withheld (indeterminate)** | neither of the above — the CI straddles an MID boundary and excludes neither |

Up to v0.3 this was written as "if the point estimate \|Δ\| < 0.10, there is no substantial difference", but **claiming equivalence from a point estimate
is wrong.** Equivalence is supported only when the whole interval lies within the equivalence range. **A "withheld" verdict arising from an insufficient
sample is an honest result, and it is not reported as "no difference".**

The MID is not changed after seeing the results.

### 7.4 The conclusion statement is fixed before the results

Depending on the direction of FP/image along the ladder, one of the four below is used. **In every case, the descriptive results of §7.1 remain.**

Verdicts are made **only on the 3 confirmatory pairs (`0–1`, `1–2`, `0–2`)** (§7.2). Rung 3 contributes nowhere.

| Result | Condition | Conclusion |
|---|---|---|
| **A** Attention reduces | point estimates ordered in the 0→2 direction **and** at least 1 pair "substantial difference" | Attention reduces false alarms under domain shift. **Because capacity is matched within ±2%, a capacity explanation is excluded** |
| **B** Substantial equivalence | **all 3 pairs** "substantial equivalence" (adjusted CI ⊂ [−0.10, +0.10]) | **External false-alarm burden is not resolved by replacing the backbone** — it is a property of domain shift, not of architecture |
| **C** Attention increases | direction opposite to A and at least 1 pair "substantial difference" | Attention is more specialised to the source domain and is disadvantaged under shift |
| **D** Directional inconsistency | point-estimate directions are not ordered | No claim about families is made; results are reported **as per-model values only** |
| **E** Indeterminate | none of the four above applies (withheld pairs remain) | **It is reported that the sample is insufficient to answer this question.** It is not rewritten as "no difference" |

- **Why E exists**: A–D in v0.3 could not accommodate the case where "the point estimates are ordered but the CI cannot exclude the opposite direction, and some
  difference exceeds the MID". Calling such a result **D (directional inconsistency) is a category error**, so it is
  separated out.
- **Not to be done**: changing the wording of A–E after seeing the results, adjusting the MID, adding or excluding arms,
  or rewriting withheld as equivalence.

### 7.5 Secondary metrics (prespecified, fixed order)

1. Lesion-level F1 / recall on the 654 positive images
2. Full-range τ curves — false positives per image vs lesion recall (FROC form), arms overlaid
3. **Generalisation gap** from internal test 764 → external 654 (per arm)
4. `test_alte_cabinete` 180-image subgroup
5. **Parameter count and inference time** — reported beside every operating point (§14)

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

## §10 Execution gates

| Gate | Condition |
|---|---|
| G1 | Training set ↔ external set near-duplicate sweep hits **0** (§3) |
| G2 | Confirmation, by comparison of resolved configs, that **all four arms** were trained with the same protocol string (rung 1 YOLO11l compared against the development study's config) |
| G3 | Confirmation from command lines and logs that no external data were used for τ selection |
| G4 | Environment `check_environment.py --stage detector` PASS |
| G5 | The sha256 of this document matches the sidecar |
| **G6** | **Confirmation that the §5.1 post-processing is frozen** — raw predictions preserved at `conf ≥ 0.001`, the same class-agnostic NMS (IoU 0.7) applied to all four arms in the same order, and the number of `max_det 300` hits recorded in the log. **Explicit confirmation that it was also applied to the NMS-free arm (RT-DETR-l)** |

## §14 Parameter asymmetry — remains only at rung 3 (2026-08-29)

As a result of reconstructing the arms as a ladder, **the asymmetry is confined to rung 3 alone**: rungs 0, 1 and 2 are 25.90 / 25.37 / 26.45M,
a spread of ±2%, and only RT-DETR-l is 32.97M (+30%).

Interpretation is limited accordingly. Three of the four detectors are matched in capacity to within 2%, so the contrasts that carry the architectural argument compare models of the same size. No unofficial intermediate scale is constructed to match RT-DETR-l, because it would be canonical to no family. Wherever RT-DETR-l differs from the other three, an effect of the transformer architecture cannot be separated from an effect of capacity, and no conclusion about attention is drawn from that arm alone. Parameter count and inference time are reported beside every operating point.

**The interpretation priority of §7.2 implements this** — 0–1, 1–2 and 0–2 carry the conclusion, and pairs involving rung 3
are reported with the confound attached.
