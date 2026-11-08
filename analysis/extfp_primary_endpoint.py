"""EXTFP public analysis copy.

EDIT analysis/public_inputs.py or set EXTFP_DATA_ROOT and EXTFP_INPUT_MAP.
Provide your local inputs and their independently recorded SHA-256 values.
See README.md for input schemas and the analysis order.
"""
from __future__ import annotations

import json
import public_inputs
import sys
from pathlib import Path

import numpy as np

REPO = public_inputs.data_root()
sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT = REPO / "runs/extfp/external"
TAU_REFERENCE = "configured-threshold-reference"


def sealed_tau() -> dict:
    blob = public_inputs.read_snapshot('runs/extfp/tau_selection/tau_selection.json')
    return {a["arm"]: a["tau"] for a in json.loads(blob)["arms"]}


def quartiles(x: np.ndarray) -> tuple[float, float, float]:
    """Median and IQR. Linear interpolation, the numpy default."""
    if len(x) == 0:
        return 0.0, 0.0, 0.0
    return (float(np.median(x)),
            float(np.percentile(x, 25)),
            float(np.percentile(x, 75)))


def main() -> int:
    # ---- refuse to run unless G6 passed -----------------------------------
    g6_log = OUT / "gate_g6.log"
    if not g6_log.exists() or "G6 PASS" not in g6_log.read_text(encoding="utf-8"):
        raise SystemExit("G6 has not passed -- refusing to compute the primary endpoint (7 / 10)")

    rec = json.loads((OUT / "external_inference.json").read_text(encoding="utf-8"))
    taus = sealed_tau()
    roster = {}
    for line in (OUT / "external_cohort_roster.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        roster[f"{r['split']}/{r['image']}"] = r

    results = []
    for a in rec["arms"]:
        arm = a["arm"]
        tau = taus[arm]
        neg_fp: list[int] = []          # FP count per NEGATIVE image
        neg_keys: list[str] = []
        neg_saturated_at_tau = 0
        pos_boxes_at_tau = 0
        subgroup_fp: list[int] = []
        hard_fp: list[int] = []
        clean_fp: list[int] = []
        n_pos = n_neg = 0

        for line in (REPO / a["predictions_jsonl"]).read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            key = d["key"]
            meta = roster[key]
            scores = np.asarray(d["scores"], dtype=np.float64)
            n_at_tau = int((scores >= tau).sum())

            if d["is_positive"]:
                n_pos += 1
                pos_boxes_at_tau += n_at_tau
                continue

            # 7.1: on a negative image every box at or above tau is a FP
            n_neg += 1
            neg_fp.append(n_at_tau)
            neg_keys.append(key)
            # 5.3: saturation only matters if it bit AT the operating point,
            # i.e. the stored list was truncated AND is still full at tau
            if d["max_det_saturated"] and n_at_tau == len(scores):
                neg_saturated_at_tau += 1
            if meta["is_subgroup"]:
                subgroup_fp.append(n_at_tau)
            if meta["n_confusable_boxes"] > 0:
                hard_fp.append(n_at_tau)
            else:
                clean_fp.append(n_at_tau)

        fp = np.asarray(neg_fp, dtype=np.int64)
        med, q1, q3 = quartiles(fp)
        dist = {
            "0": int((fp == 0).sum()),
            "1": int((fp == 1).sum()),
            "2": int((fp == 2).sum()),
            "3+": int((fp >= 3).sum()),
        }
        worst = sorted(zip(neg_fp, neg_keys), reverse=True)[:5]

        results.append({
            "arm": arm,
            "tau": tau,
            "n_negative_images": n_neg,
            "n_positive_images": n_pos,
            # ---- 7.1 primary
            "total_false_positives": int(fp.sum()),
            "fp_per_image_mean": float(fp.mean()) if len(fp) else 0.0,
            "fp_per_image_median": med,
            "fp_per_image_q1": q1,
            "fp_per_image_q3": q3,
            "fp_per_image_iqr": q3 - q1,
            "fp_per_image_max": int(fp.max()) if len(fp) else 0,
            "images_with_any_fp": int((fp > 0).sum()),
            "prop_images_with_any_fp": float((fp > 0).mean()) if len(fp) else 0.0,
            "fp_count_distribution": dist,
            # 5.3 censoring rule
            "negative_images_saturated_at_tau": neg_saturated_at_tau,
            "censored": neg_saturated_at_tau > 0,
            # ---- secondary, descriptive only
            "secondary": {
                "hard_negative_images": len(hard_fp),
                "hard_fp_per_image_mean": float(np.mean(hard_fp)) if hard_fp else 0.0,
                "clean_negative_images": len(clean_fp),
                "clean_fp_per_image_mean": float(np.mean(clean_fp)) if clean_fp else 0.0,
                "subgroup_test_alte_cabinete_negative_images": len(subgroup_fp),
                "subgroup_fp_per_image_mean": float(np.mean(subgroup_fp)) if subgroup_fp else 0.0,
                "positive_images_boxes_at_tau": pos_boxes_at_tau,
            },
            "worst_images": [{"key": k, "fp": int(n)} for n, k in worst],
        })

    summary = {
        "prespec": "protocols/EXTFP_PRESPEC.md 7.1 (frozen v1.1)",
        "tau_reference": TAU_REFERENCE,
        "tau_applied": taus,
        "gate_g6": "PASS",
        "definition": ("on a negative image every box with score >= tau is a false positive; "
                       "8 hard/clean is NOT applied to the primary (7.1)"),
        "cohort": rec["cohort"],
        "arms": results,
    }
    (OUT / "primary_endpoint.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # ---- report
    print(f"{'arm':10s} {'tau':>6s} {'FP/img':>8s} {'median':>7s} {'IQR':>11s} "
          f"{'>=1 FP':>8s} {'total FP':>9s} {'0/1/2/3+':>20s}  cens")
    for r in results:
        d = r["fp_count_distribution"]
        print(f"{r['arm']:10s} {r['tau']:6.3f} {r['fp_per_image_mean']:8.3f} "
              f"{r['fp_per_image_median']:7.1f} "
              f"{r['fp_per_image_q1']:.1f}-{r['fp_per_image_q3']:<6.1f} "
              f"{r['prop_images_with_any_fp']*100:7.1f}% {r['total_false_positives']:9d} "
              f"{d['0']:5d}/{d['1']:4d}/{d['2']:4d}/{d['3+']:5d}  "
              f"{'YES' if r['censored'] else 'no'}")
    print(f"\nnegative images per arm: {results[0]['n_negative_images']}")
    print(f"wrote {OUT / 'primary_endpoint.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
