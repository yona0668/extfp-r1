"""EXTFP public analysis copy.

EDIT analysis/public_inputs.py or set EXTFP_DATA_ROOT and EXTFP_INPUT_MAP.
Provide your local inputs and their independently recorded SHA-256 values.
See README.md for input schemas and the analysis order.
"""
from __future__ import annotations

import hashlib
import json
import platform
import public_inputs
import sys
from pathlib import Path

import numpy as np

REPO = public_inputs.data_root()
sys.path.insert(0, str(Path(__file__).resolve().parent))

from extfp_tau_selection import (  # noqa: E402  -- the frozen, already-sealed rules
    GRID,
    MATCH_IOU,
    greedy_match,
)
from extfp_external_cohort import LESION_CLASS, read_label  # noqa: E402

OUT = REPO / "runs/extfp/external"
RESULT_REFERENCE = "configured-result-reference"
TAU_REFERENCE = "configured-threshold-reference"
PAIRWISE_REFERENCE = "configured-result-reference"

PRESPEC_SHA_CRLF = "c04a522514c4214b16724005d871a52695adb0557207615e031e3e4cf801bddb"

RECALL_TARGETS = (0.50, 0.70)  # 6.3-4, the only two, fixed in the frozen text

# 14: parameter counts. The prespec's own figures are the stock-config ones and
# are NOT restated or revised here; what this records is the measured count of
# the three trained checkpoints, read out of the sealed training logs.
ARM_ORDER = ["yolov8m", "yolo11l", "rtdetr-l"]
PARAMS_MEASURED = {          # unfused / fused, from runs/extfp/logs/*.log
    "yolov8m": (25858057, 25841497),
    "yolo11l": (25312793, 25281625),
    "rtdetr-l": (32812241, 31989905),
}
PARAMS_PRESPEC_S14_M = {"yolov8m": 25.90, "yolo11l": 25.37, "rtdetr-l": 32.97}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_reference(reference_id: str, path: str) -> str:
    return public_inputs.read_snapshot(path)


def sealed_tau() -> dict:
    blob = read_reference(TAU_REFERENCE, "runs/extfp/tau_selection/tau_selection.json")
    return {a["arm"]: a["tau"] for a in json.loads(blob)["arms"]}


def load_gt_class7(label_path: Path, width: int, height: int) -> np.ndarray:
    """External GT for the folded lesion class.

    6.1 folds classes for MATCHING purposes, but on the external corpus only
    class 7 (Apical periodontitis) is the lesion -- that is the same definition
    3 and 7.1 already use to call an image positive or negative. The other
    classes are findings, not lesions, and are not ground truth here.
    """
    rows = [r for r in read_label(label_path) if r[0] == LESION_CLASS]
    if not rows:
        return np.zeros((0, 4), dtype=np.float64)
    out = []
    for row in rows:
        cx, cy, w, h = row[1], row[2], row[3], row[4]
        out.append([(cx - w / 2) * width, (cy - h / 2) * height,
                    (cx + w / 2) * width, (cy + h / 2) * height])
    return np.asarray(out, dtype=np.float64)


def counts_at_grid(scores: np.ndarray, is_tp: np.ndarray, total_gt: int):
    """TP / FP / FN over the whole score grid, micro (6.2). Vectorised."""
    order = np.argsort(-scores, kind="stable")
    s, t = scores[order], is_tp[order]
    cum_tp = np.cumsum(t)
    n_at = np.searchsorted(-s, -GRID, side="right")
    idx = np.clip(n_at - 1, 0, None)
    tp = np.where(n_at > 0, cum_tp[idx], 0).astype(np.int64)
    allp = n_at.astype(np.int64)
    return tp, allp - tp, total_gt - tp


def n_at_grid(scores: np.ndarray) -> np.ndarray:
    s = np.sort(scores)[::-1]
    return np.searchsorted(-s, -GRID, side="right").astype(np.int64)


def grid_index(tau: float) -> int:
    i = int(np.argmin(np.abs(GRID - tau)))
    if abs(GRID[i] - tau) > 1e-9:
        raise SystemExit(f"tau {tau} is not on the frozen grid")
    return i


def main() -> int:
    # ------------------------------------------------------------------ guards
    prespec = REPO / "specifications/EXTFP_PRESPEC.md"
    if sha256_file(prespec) != PRESPEC_SHA_CRLF:
        raise SystemExit("PRESPEC HAS CHANGED -- refusing to run 7.5")

    sealed = {}
    for line in (OUT / "raw_external_predictions.sha256").read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        digest, rest = line.split(None, 1)
        sealed[rest.split()[0]] = digest
    for name, digest in sealed.items():
        got = sha256_file(OUT / name)
        if got != digest:
            raise SystemExit(f"SEAL BROKEN: {name} {got} != {digest}")

    for reference_id, rel in ((RESULT_REFERENCE, "runs/extfp/external/primary_endpoint.json"),
                        (PAIRWISE_REFERENCE, "runs/extfp/external/pairwise_7274.json")):
        if read_reference(reference_id, rel) != (REPO / rel).read_text(encoding="utf-8"):
            raise SystemExit(f"{rel} differs from {reference_id} -- refusing to run")

    taus = sealed_tau()
    primary = json.loads((OUT / "primary_endpoint.json").read_text(encoding="utf-8"))
    primary_by_arm = {a["arm"]: a for a in primary["arms"]}
    inference = json.loads((OUT / "external_inference.json").read_text(encoding="utf-8"))
    timing = {a["arm"]: a for a in inference["arms"]}

    roster = {}
    for line in (OUT / "external_cohort_roster.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        roster[f"{r['split']}/{r['image']}"] = r

    val_record = json.loads(read_reference(TAU_REFERENCE, "runs/extfp/tau_selection/tau_selection.json"))

    arms_out = []
    for arm in ARM_ORDER:
        tau = taus[arm]
        ti = grid_index(tau)

        pos_scores, pos_is_tp, neg_scores = [], [], []
        total_gt = n_pos = n_neg = 0
        sat_pos_all = sat_neg_all = 0
        sat_neg_full_at_grid = np.zeros(len(GRID), dtype=np.int64)
        gt_mismatch = []

        jsonl = OUT / f"external_predictions_{arm}.jsonl"
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            meta = roster[d["key"]]
            boxes = np.asarray(d["boxes_xyxy"], dtype=np.float64).reshape(-1, 4)
            scores = np.asarray(d["scores"], dtype=np.float64)

            if d["is_positive"]:
                n_pos += 1
                gt = load_gt_class7(REPO / meta["label_path"], d["width"], d["height"])
                if len(gt) != d["n_lesion_boxes_gt"]:
                    gt_mismatch.append(d["key"])
                total_gt += len(gt)
                # boxes arrive already score-descending from 5.3 step (3)
                pos_scores.append(scores)
                pos_is_tp.append(greedy_match(boxes, scores, gt))
                sat_pos_all += int(d["max_det_saturated"])
            else:
                n_neg += 1
                neg_scores.append(scores)
                sat_neg_all += int(d["max_det_saturated"])
                if d["max_det_saturated"]:
                    # 5.3: truncation only bites at a given tau if the stored
                    # list is STILL full at that tau
                    sat_neg_full_at_grid += (n_at_grid(scores) == len(scores)).astype(np.int64)

        if gt_mismatch:
            raise SystemExit(f"[{arm}] GT box count disagrees with the sealed roster: {gt_mismatch[:5]}")
        if (n_pos, n_neg) != (654, 1154):
            raise SystemExit(f"[{arm}] cohort is {n_pos}+{n_neg}, expected 654+1154")

        ps = np.concatenate(pos_scores) if pos_scores else np.zeros(0)
        pt = np.concatenate(pos_is_tp) if pos_is_tp else np.zeros(0, bool)
        ns = np.concatenate(neg_scores) if neg_scores else np.zeros(0)

        tp, fp_pos, fn = counts_at_grid(ps, pt, total_gt)
        denom = 2 * tp + fp_pos + fn
        f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
        recall = tp / total_gt
        precision = np.where(tp + fp_pos > 0, tp / np.maximum(tp + fp_pos, 1), 0.0)
        fp_neg_total = n_at_grid(ns)                      # 7.1 definition, every box counts
        fp_per_image = fp_neg_total / n_neg

        # ---- guard: the curve must reproduce the SEALED primary endpoint at tau
        sealed_mean = primary_by_arm[arm]["fp_per_image_mean"]
        sealed_total = primary_by_arm[arm]["total_false_positives"]
        if int(fp_neg_total[ti]) != sealed_total or abs(fp_per_image[ti] - sealed_mean) > 1e-12:
            raise SystemExit(f"[{arm}] 7.5 curve does not reproduce the sealed 7.1 endpoint "
                             f"({fp_neg_total[ti]} vs {sealed_total}) -- refusing to report")

        # ---- item 6: tau re-fixed at internal-val recall 0.50 / 0.70
        val_arm = next(a for a in val_record["arms"] if a["arm"] == arm)
        v_scores, v_tp, v_gt = [], [], 0
        for line in (REPO / val_arm["predictions_jsonl"]).read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            v_scores.append(np.asarray(d["scores"], dtype=np.float64))
            v_tp.append(np.asarray(d["is_tp_iou50"], dtype=bool))
            v_gt += int(d["n_gt"])
        vs = np.concatenate(v_scores) if v_scores else np.zeros(0)
        vt = np.concatenate(v_tp) if v_tp else np.zeros(0, bool)
        v_tp_at, _, _ = counts_at_grid(vs, vt, v_gt)
        v_recall = v_tp_at / v_gt

        sensitivity = []
        for target in RECALL_TARGETS:
            hit = np.nonzero(v_recall >= target)[0]
            if len(hit) == 0:
                sensitivity.append({"target_internal_recall": target, "tau": None,
                                    "reason": "internal val recall never reaches the target"})
                continue
            # 6.2 tie-break, inherited unchanged: the LARGEST tau among those
            # that reach the target (the more conservative operating point)
            k = int(hit.max())
            sensitivity.append({
                "target_internal_recall": target,
                "tau": float(round(GRID[k], 3)),
                "internal_val_recall_at_tau": float(v_recall[k]),
                "external_fp_per_image": float(fp_per_image[k]),
                "external_total_fp": int(fp_neg_total[k]),
                "external_lesion_recall_654": float(recall[k]),
                "external_lesion_f1_654": float(f1[k]),
                "negative_images_saturated_at_tau": int(sat_neg_full_at_grid[k]),
                "censored": bool(sat_neg_full_at_grid[k] > 0),
            })

        # ---- write the FROC curve (item 2), full grid, one CSV per arm
        csv = OUT / f"secondary_75_froc_{arm}.csv"
        with csv.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write("tau,lesion_recall_654,lesion_precision_654,lesion_f1_654,"
                     "tp,fp_positive_images,fn,fp_per_negative_image,total_fp_1154,"
                     "negative_images_saturated_at_tau\n")
            for i in range(len(GRID)):
                fh.write(f"{GRID[i]:.3f},{recall[i]:.6f},{precision[i]:.6f},{f1[i]:.6f},"
                         f"{tp[i]},{fp_pos[i]},{fn[i]},{fp_per_image[i]:.6f},"
                         f"{fp_neg_total[i]},{sat_neg_full_at_grid[i]}\n")

        params_unfused, params_fused = PARAMS_MEASURED[arm]
        arms_out.append({
            "arm": arm,
            "operating_point_tau": tau,
            "tau_source": f"sealed at {TAU_REFERENCE}, internal val, never re-chosen",
            # ---- item 1
            "item1_positive_654": {
                "exploratory": "not used for 7.4 P1",
                "n_positive_images": n_pos,
                "n_gt_lesion_boxes": int(total_gt),
                "at_operating_point_tau": {
                    "tau": tau,
                    "tp": int(tp[ti]), "fp": int(fp_pos[ti]), "fn": int(fn[ti]),
                    "precision": float(precision[ti]),
                    "recall": float(recall[ti]),
                    "f1": float(f1[ti]),
                },
                "definition": ("6.2 verbatim: matching IoU 0.50, greedy 1:1 score-descending, "
                               "micro over all 654 images, class-agnostic (6.1)"),
            },
            # ---- item 2
            "item2_froc_curve": {
                "exploratory": "not used for 7.4 P1",
                "csv": str(csv.relative_to(REPO)).replace("\\", "/"),
                "grid": "0.001..0.999 step 0.001 (5.3)",
                "x": "false positives per negative image (1,154 images, 7.1 definition)",
                "y": "lesion recall on the 654 positive images (6.2 matching)",
                "n_grid_points": int(len(GRID)),
                "fp_per_image_at_grid_min_tau": float(fp_per_image[0]),
                "recall_at_grid_min_tau": float(recall[0]),
                "max_recall_reached": float(recall.max()),
                "grid_points_with_any_saturated_negative_image": int((sat_neg_full_at_grid > 0).sum()),
            },
            # ---- item 5
            "item5_capacity_and_time": {
                "exploratory": "not used for 7.4 P1",
                "parameters_trained_checkpoint_unfused": params_unfused,
                "parameters_trained_checkpoint_fused": params_fused,
                "parameters_prespec_14_millions": PARAMS_PRESPEC_S14_M[arm],
                "note_14": ("14's figures are the stock-config counts and stand unrevised; "
                            "the capacity percentages in 7.2 / 4.1 are NOT recomputed here"),
                "external_inference_seconds_1808_images": timing[arm]["seconds"],
                "external_inference_ms_per_image": timing[arm]["ms_per_image"],
                "timing_source": f"measured during the sealed run at {RESULT_REFERENCE}; not re-run",
            },
            # ---- item 6
            "item6_tau_sensitivity": {
                "exploratory": "not used for 7.4 P1",
                "rule": ("6.3-4: tau re-fixed at internal-val recall 0.50 and 0.70 instead of "
                         "max F1; still selected on the INTERNAL val split (9 forbids external "
                         "threshold selection); tie-break = largest tau reaching the target (6.2)"),
                "internal_val_gt_boxes": int(v_gt),
                "points": sensitivity,
            },
            # ---- item 7
            "item7_max_det_saturation": {
                "exploratory": "not used for 7.4 P1",
                "max_det": 300,
                "rule": "5.3: saturation is only material if the stored list is still full at tau",
                "positive_images_saturated_at_storage": sat_pos_all,
                "negative_images_saturated_at_storage": sat_neg_all,
                "negative_images_saturated_at_operating_point_tau": int(sat_neg_full_at_grid[ti]),
                "negative_images_saturated_at_sensitivity_points": {
                    str(p["target_internal_recall"]): p.get("negative_images_saturated_at_tau")
                    for p in sensitivity},
                "max_over_whole_grid": int(sat_neg_full_at_grid.max()),
                "censored_anywhere": bool(sat_neg_full_at_grid.max() > 0),
            },
        })

    summary = {
        "STATUS": "EXPLORATORY / DESCRIPTIVE ONLY -- NOT USED FOR 7.4 P1",
        "prespec": "protocols/EXTFP_PRESPEC.md 7.5 (frozen v1.1)",
        "prespec_sha256_crlf": PRESPEC_SHA_CRLF,
        "executed_items": [1, 2, 5, 6, 7],
        "withheld_items": {
            "3": "generalisation gap 764 -> 654: metric and formula not specified in the frozen text",
            "4": "test_alte_cabinete 180-image subgroup: metric not specified in the frozen text",
        },
        "not_done": ["no new hypothesis test", "no new MID", "no superiority judgement",
                     "no new threshold", "no new subgroup", "no re-inference",
                     "7.1-7.4 unchanged"],
        "inputs_verified": {
            "raw_predictions_sha256": sealed,
            "primary_endpoint_json": f"byte-identical to {RESULT_REFERENCE}",
            "pairwise_7274_json": f"byte-identical to {PAIRWISE_REFERENCE}",
            "tau": taus,
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "match_iou": MATCH_IOU,
        "arms": arms_out,
    }
    (OUT / "secondary_75.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------ report
    print("7.5 items 1/2/5/6/7 -- EXPLORATORY; not used for 7.4 P1")
    print("")
    print("item 1 -- lesion-level detection on the 654 positive images, at the sealed tau")
    print(f"{'arm':10s} {'tau':>6s} {'TP':>5s} {'FP':>5s} {'FN':>5s} "
          f"{'prec':>7s} {'recall':>7s} {'F1':>7s}")
    for a in arms_out:
        p = a["item1_positive_654"]["at_operating_point_tau"]
        print(f"{a['arm']:10s} {p['tau']:6.3f} {p['tp']:5d} {p['fp']:5d} {p['fn']:5d} "
              f"{p['precision']:7.4f} {p['recall']:7.4f} {p['f1']:7.4f}")

    print("")
    print("item 5 -- capacity and inference time, beside every operating point (14)")
    print(f"{'arm':10s} {'params(M)':>10s} {'fused(M)':>9s} {'ms/image':>9s} {'sec/1808':>9s}")
    for a in arms_out:
        c = a["item5_capacity_and_time"]
        print(f"{a['arm']:10s} {c['parameters_trained_checkpoint_unfused']/1e6:10.3f} "
              f"{c['parameters_trained_checkpoint_fused']/1e6:9.3f} "
              f"{c['external_inference_ms_per_image']:9.2f} "
              f"{c['external_inference_seconds_1808_images']:9.1f}")

    print("")
    print("item 6 -- tau sensitivity: tau re-fixed at internal-val recall 0.50 / 0.70")
    print(f"{'arm':10s} {'point':>7s} {'tau':>6s} {'val R':>7s} {'ext FP/img':>11s} "
          f"{'ext R':>7s} {'ext F1':>7s} {'params(M)':>10s} {'ms/img':>7s}")
    for a in arms_out:
        c = a["item5_capacity_and_time"]
        base = a["item1_positive_654"]["at_operating_point_tau"]
        base_fp = primary_by_arm[a["arm"]]["fp_per_image_mean"]
        print(f"{a['arm']:10s} {'F1max':>7s} {base['tau']:6.3f} "
              f"{'-':>7s} {base_fp:11.4f} "
              f"{base['recall']:7.4f} {base['f1']:7.4f} "
              f"{c['parameters_trained_checkpoint_unfused']/1e6:10.3f} "
              f"{c['external_inference_ms_per_image']:7.2f}")
        for p in a["item6_tau_sensitivity"]["points"]:
            if p["tau"] is None:
                print(f"{'':10s} {p['target_internal_recall']:7.2f} {'n/a':>6s}  {p['reason']}")
                continue
            print(f"{'':10s} {p['target_internal_recall']:7.2f} {p['tau']:6.3f} "
                  f"{p['internal_val_recall_at_tau']:7.4f} {p['external_fp_per_image']:11.4f} "
                  f"{p['external_lesion_recall_654']:7.4f} {p['external_lesion_f1_654']:7.4f} "
                  f"{c['parameters_trained_checkpoint_unfused']/1e6:10.3f} "
                  f"{c['external_inference_ms_per_image']:7.2f}")

    print("")
    print("item 7 -- max_det 300 saturation (5.3)")
    print(f"{'arm':10s} {'stored pos':>10s} {'stored neg':>10s} {'at tau':>7s} "
          f"{'grid max':>9s}  censored")
    for a in arms_out:
        s = a["item7_max_det_saturation"]
        print(f"{a['arm']:10s} {s['positive_images_saturated_at_storage']:10d} "
              f"{s['negative_images_saturated_at_storage']:10d} "
              f"{s['negative_images_saturated_at_operating_point_tau']:7d} "
              f"{s['max_over_whole_grid']:9d}  {'YES' if s['censored_anywhere'] else 'no'}")

    print("")
    print("item 2 -- FROC curves written per arm:")
    for a in arms_out:
        print(f"  {a['item2_froc_curve']['csv']}")
    print("")
    print("items 3 and 4: WITHHELD, not computed.")
    print(f"wrote {OUT / 'secondary_75.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
