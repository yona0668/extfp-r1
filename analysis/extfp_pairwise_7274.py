"""EXTFP public analysis copy.

EDIT analysis/public_inputs.py or set EXTFP_DATA_ROOT and EXTFP_INPUT_MAP.
Provide your local inputs and their independently recorded SHA-256 values.
See README.md for input schemas and the analysis order.
"""
from __future__ import annotations

import hashlib
import json
import public_inputs
import sys
from pathlib import Path

import numpy as np

REPO = public_inputs.data_root()
OUT = REPO / "runs/extfp/external"

RESULT_REFERENCE = "configured-result-reference"
TAU_REFERENCE = "configured-threshold-reference"

MID = 0.10                         # 7.3, fixed, never adjusted
B = 10_000                         # 7.3.1
SEED = 20260830                    # 7.3.1
CI_LOW, CI_HIGH = 2.5, 97.5        # 7.3.1 percentile bootstrap

# 7.2 / 4.1 -- standing fixed in advance, never changed by a result
PAIRS = [
    ("yolov8m", "yolo11l", "primary (confirmatory)", 2.1),
    ("yolov8m", "rtdetr-l", "confounded secondary (exploratory)", 27.3),
    ("yolo11l", "rtdetr-l", "confounded secondary (exploratory)", 30.0),
]
PRIMARY_PAIR = ("yolov8m", "yolo11l")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sealed_tau() -> dict[str, float]:
    blob = public_inputs.read_snapshot('runs/extfp/tau_selection/tau_selection.json')
    return {a["arm"]: a["tau"] for a in json.loads(blob)["arms"]}


def load_negative_fp_counts(taus: dict[str, float]) -> tuple[list[str], dict[str, np.ndarray]]:
    """7.1 definition: on a negative image, every box with score >= tau is a FP.

    Returns the fixed image order and, per arm, the per-image FP count vector.
    """
    rec = json.loads((OUT / "external_inference.json").read_text(encoding="utf-8"))
    counts: dict[str, dict[str, int]] = {}
    for a in rec["arms"]:
        arm = a["arm"]
        tau = taus[arm]
        jl = REPO / a["predictions_jsonl"]
        # the file must be the one the sealed record hashes
        if sha256_file(jl) != a["predictions_sha256"]:
            raise SystemExit(f"[{arm}] predictions sha256 drift -- refusing to proceed")
        per_image: dict[str, int] = {}
        for line in jl.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            if d["is_positive"]:
                continue                      # 7.1 is on negatives only
            s = np.asarray(d["scores"], dtype=np.float64)
            per_image[d["key"]] = int((s >= tau).sum())
        counts[arm] = per_image

    arms = sorted(counts)
    keys = sorted(counts[arms[0]])
    for arm in arms[1:]:
        if sorted(counts[arm]) != keys:
            raise SystemExit("negative image sets differ between arms -- not paired")
    if len(keys) != 1154:
        raise SystemExit(f"negative cohort is {len(keys)}, expected 1154")
    return keys, {arm: np.array([counts[arm][k] for k in keys], dtype=np.float64)
                  for arm in arms}


def verify_against_sealed(fp: dict[str, np.ndarray]) -> list[tuple[str, bool, str]]:
    """The rebuilt counts must reproduce the sealed 7.1 endpoint exactly."""
    sealed = json.loads((OUT / "primary_endpoint.json").read_text(encoding="utf-8"))
    checks = []
    for a in sealed["arms"]:
        arm = a["arm"]
        v = fp[arm]
        checks.append((f"{arm}: total FP == sealed 7.1",
                       int(v.sum()) == a["total_false_positives"],
                       f"{int(v.sum())} vs {a['total_false_positives']}"))
        checks.append((f"{arm}: mean FP/image == sealed 7.1",
                       abs(float(v.mean()) - a["fp_per_image_mean"]) < 1e-12,
                       f"{v.mean():.6f} vs {a['fp_per_image_mean']:.6f}"))
        checks.append((f"{arm}: images with >=1 FP == sealed 7.1",
                       int((v > 0).sum()) == a["images_with_any_fp"],
                       f"{int((v > 0).sum())} vs {a['images_with_any_fp']}"))
    return checks


def bootstrap(fp: dict[str, np.ndarray]) -> tuple[dict, np.ndarray]:
    """7.3.1: joint/paired image-level resampling, B=10,000, seed=20260830.

    One resample draws a set of images and applies it to ALL THREE arms at once,
    so every pair difference comes from the same resampled image set.
    """
    arms = sorted(fp)
    n = len(fp[arms[0]])
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n, size=(B, n))          # the shared image draws
    boot_means = {arm: fp[arm][idx].mean(axis=1) for arm in arms}
    return boot_means, idx


def judge(ci_low: float, ci_high: float) -> tuple[str, str]:
    """7.3.2, applied to the primary pair only."""
    excludes_zero = (ci_low > 0) or (ci_high < 0)
    same_sign = (ci_low > 0 and ci_high > 0) or (ci_low < 0 and ci_high < 0)
    nearer_zero = min(abs(ci_low), abs(ci_high))
    whole_ci_inside = (ci_low >= -MID) and (ci_high <= MID)

    if excludes_zero and same_sign and nearer_zero >= MID:
        return "substantial difference", (
            f"CI excludes 0 and lies entirely outside the equivalence range: "
            f"endpoints share a sign and the endpoint nearer 0 is "
            f"|{nearer_zero:.4f}| >= MID {MID}")
    if whole_ci_inside:
        return "substantial equivalence", (
            f"the whole CI [{ci_low:.4f}, {ci_high:.4f}] lies inside "
            f"[-{MID}, +{MID}]")
    return "indeterminate", (
        f"CI [{ci_low:.4f}, {ci_high:.4f}] straddles a MID boundary: it rules out "
        f"neither a difference >= MID nor an effect < MID")


VERDICT_TO_CONCLUSION = {
    "substantial difference": ("P1", (
        "Even between two detectors whose capacities are within 2.1% of each other, the false-alarm burden on "
        "external negative images differed substantially (>= MID).")),
    "substantial equivalence": ("P2", (
        "Between these two detectors the external false-alarm burden was substantially equivalent — changing "
        "the backbone within this range did not change the burden.")),
    "indeterminate": ("P3", (
        "This sample was not sufficient to judge the size of the difference between the two detectors "
        "against the MID.")),
}


def main() -> int:
    taus = sealed_tau()
    keys, fp = load_negative_fp_counts(taus)

    print("=== inputs fixed from the sealed run ===")
    checks = verify_against_sealed(fp)
    width = max(len(n) for n, _, _ in checks)
    if any(not ok for _, ok, _ in checks):
        for n, ok, d in checks:
            print(f"  [{'PASS' if ok else 'FAIL'}] {n.ljust(width)}  {d}")
        raise SystemExit("rebuilt FP counts do not match the sealed 7.1 endpoint")
    for n, ok, d in checks:
        print(f"  [PASS] {n.ljust(width)}  {d}")
    print(f"  negative cohort: {len(keys)} images; tau {taus}\n")

    boot_means, _ = bootstrap(fp)

    # ---- 7.2: all three pairs, one table, standing and capacity in it
    rows = []
    for a, b, standing, cap in PAIRS:
        delta = float(fp[a].mean() - fp[b].mean())
        boot_delta = boot_means[a] - boot_means[b]
        lo = float(np.percentile(boot_delta, CI_LOW))
        hi = float(np.percentile(boot_delta, CI_HIGH))
        rows.append({
            "pair": f"{a} - {b}",
            "arm_a": a, "arm_b": b,
            "standing": standing,
            "capacity_difference_pct": cap,
            "delta_fp_per_image": delta,
            "ci95_percentile": [lo, hi],
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
            "judged": (a, b) == PRIMARY_PAIR,
        })

    print("=== 7.2 pairwise comparison (all three pairs reported; only 0-1 is judged) ===")
    print(f"{'pair':26s} {'standing':36s} {'cap%':>6s} {'delta':>8s} "
          f"{'95% percentile CI':>22s}  judged")
    for r in rows:
        lo, hi = r["ci95_percentile"]
        print(f"{r['pair']:26s} {r['standing']:36s} {r['capacity_difference_pct']:6.1f} "
              f"{r['delta_fp_per_image']:8.4f} [{lo:9.4f}, {hi:9.4f}]  "
              f"{'YES' if r['judged'] else 'no (exploratory)'}")

    primary = next(r for r in rows if r["judged"])
    lo, hi = primary["ci95_percentile"]

    print("\n=== 7.3 MID and the adjusted CI ===")
    print(f"  MID = {MID} FP per lesion-negative image (frozen; not adjusted)")
    print(f"  confirmatory family size = 1 -> no multiplicity correction; "
          f"adjusted CI == the 95% percentile CI")
    print(f"  bootstrap: image-level joint resampling, B={B}, seed={SEED}, percentile")
    print(f"  primary pair {primary['pair']}: delta = {primary['delta_fp_per_image']:.4f}, "
          f"adjusted CI = [{lo:.4f}, {hi:.4f}]")
    print(f"  |endpoint nearer 0| = {min(abs(lo), abs(hi)):.4f}  vs MID {MID}")
    print(f"  CI entirely inside [-{MID}, +{MID}]? "
          f"{'yes' if (lo >= -MID and hi <= MID) else 'no'}")

    verdict, reason = judge(lo, hi)
    code, sentence = VERDICT_TO_CONCLUSION[verdict]
    print("\n=== 7.3.2 tri-partite decision (primary pair only) ===")
    print(f"  verdict: {verdict.upper()}")
    print(f"  reason : {reason}")
    direction = (f"{primary['arm_a']} lower" if primary["delta_fp_per_image"] < 0
                 else f"{primary['arm_b']} lower")
    print("\n=== 7.4 prespecified conclusion ===")
    print(f"  {code}: {sentence}")
    print(f"  direction: {direction} "
          f"({primary['arm_a']} {fp[primary['arm_a']].mean():.4f} vs "
          f"{primary['arm_b']} {fp[primary['arm_b']].mean():.4f} FP/image)")
    print("  cause is NOT attributed to architecture or attention (7.4).")
    print("  0-3 and 1-3 contributed nothing to this verdict (7.2, 7.4).")

    summary = {
        "prespec": "protocols/EXTFP_PRESPEC.md 7.2/7.3/7.4 (frozen v1.1)",
        "result_reference": RESULT_REFERENCE,
        "tau_reference": TAU_REFERENCE,
        "tau_applied": taus,
        "negative_cohort_images": len(keys),
        "bootstrap": {"unit": "image", "joint_paired_across_arms": True,
                      "B": B, "seed": SEED, "interval": "percentile",
                      "percentiles": [CI_LOW, CI_HIGH]},
        "MID": MID,
        "multiplicity": ("confirmatory family size 1 (pair 0-1 only); no correction; "
                         "Holm has no applicable target (7.3.1)"),
        "s7_2_pairs": rows,
        "s7_3_primary": {
            "pair": primary["pair"],
            "delta_fp_per_image": primary["delta_fp_per_image"],
            "adjusted_ci95": [lo, hi],
            "endpoint_nearer_zero_abs": min(abs(lo), abs(hi)),
            "ci_excludes_zero": primary["ci_excludes_zero"],
            "ci_inside_equivalence_range": bool(lo >= -MID and hi <= MID),
        },
        "s7_3_2_verdict": verdict,
        "s7_3_2_reason": reason,
        "s7_4_conclusion_code": code,
        "s7_4_conclusion": sentence,
        "s7_4_direction": direction,
        "confounded_pairs_note": (
            "RT-DETR-l has about 30% more parameters and is NMS-free, so this difference cannot "
            "separate the effect of architecture from the effect of capacity."),
        "not_done": ["causal attribution to architecture or attention",
                     "summing confounded pairs into the primary evidence",
                     "monotonic / ladder / generation-trend narration",
                     "any change to MID, tau, arms, pair standing or metric"],
    }
    (OUT / "pairwise_7274.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT / 'pairwise_7274.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
