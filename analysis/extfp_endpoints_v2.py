"""EXTFP public analysis copy.

EDIT analysis/public_inputs.py or set EXTFP_DATA_ROOT and EXTFP_INPUT_MAP.
Provide your local inputs and their independently recorded SHA-256 values.
See README.md for input schemas and the analysis order.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import public_inputs
import sys
from pathlib import Path

import numpy as np

REPO = public_inputs.data_root()
sys.path.insert(0, str(Path(__file__).resolve().parent))

from extfp_tau_selection import common_postprocess, sha256_file  # noqa: E402

OUT = REPO / "runs/extfp/external_v2"
TAU_REFERENCE = "configured-threshold-reference"
MID = 0.10
B = 10_000
BOOT_SEED = 20260903

MODELS = ("yolov8m", "yolov9c", "yolo11l")   # M = {arm 0, arm 4, arm 1}
SEEDS = (0, 1, 2)
RUN_KEYS = [f"{m}-s{s}" for s in SEEDS for m in MODELS]   # seed-major, frozen order

# v1 sealed anchor (out of matrix; direction reference for E1 only)
V1_ANCHOR_DELTA = -0.2114


def sealed_tau() -> dict:
    blob = public_inputs.read_snapshot('runs/extfp/tau_selection_v2/tau_selection_v2.json')
    return {a["arm"]: a["tau"] for a in json.loads(blob)["arms"]}


def parse_sidecar() -> dict:
    """raw_external_predictions.sha256 -> {filename: sha256}."""
    out = {}
    for ln in (OUT / "raw_external_predictions.sha256").read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split()
        out[parts[1]] = parts[0]
    return out


def preflight(rec: dict, taus: dict) -> list:
    checks: list = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), str(detail)))

    check("nine inference runs recorded", rec["inference_runs"] == 9, rec["inference_runs"])
    arms = {a["arm"] for a in rec["arms"]}
    check("run set == nine canonical", arms == set(RUN_KEYS), sorted(arms))
    check("tau NOT applied at inference",
          rec["postprocess"]["tau_applied_at_inference"] is False)
    live_pp = hashlib.sha256(inspect.getsource(common_postprocess).encode()).hexdigest()
    check("shared common_postprocess == live frozen module (G6-style)",
          rec["postprocess"]["common_postprocess_source_sha256"] == live_pp, live_pp[:16] + "...")
    check("cohort identical to v1 sealed identity",
          rec["cohort"]["identical_to_v1_sealed_identity"] is True)
    check("cohort counts 1808/654/1154",
          rec["cohort"]["counts"]["total"] == 1808
          and rec["cohort"]["counts"]["positive"] == 654
          and rec["cohort"]["counts"]["negative"] == 1154)

    sidecar = parse_sidecar()
    check("sidecar pins nine files", len(sidecar) == 9, len(sidecar))
    for a in rec["arms"]:
        key = a["arm"]
        g = a["completion_guard"]
        check(f"{key}: completion guard clean",
              g["written_count"] == 1808 and g["duplicate_keys"] == 0
              and g["missing_keys"] == 0 and g["keys_match_frozen_manifest"] is True,
              f"written={g['written_count']}")
        p = REPO / a["predictions_jsonl"]
        disk = sha256_file(p)
        check(f"{key}: predictions sha256 == record == sealed sidecar",
              disk == a["predictions_sha256"] == sidecar.get(p.name), disk[:16] + "...")
        check(f"{key}: tau == sealed record {TAU_REFERENCE}",
              a["tau_carried_not_applied"] == taus[key], taus[key])
        check(f"{key}: saturation recorded",
              isinstance(a["max_det_saturated_images"], int), a["max_det_saturated_images"])

    width = max(len(n) for n, _, _ in checks)
    failed = [n for n, ok, _ in checks if not ok]
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name.ljust(width)}  {detail}")
    if failed:
        raise SystemExit(f"\nENDPOINT PREFLIGHT FAILED: {failed}\nNothing was computed.")
    print(f"\nEndpoint preflight PASS ({len(checks)}/{len(checks)})\n")
    return checks


def trifurcate(lo: float, hi: float) -> str:
    """v1.1 7.3.2 on one CI. Returns 'diff' / 'eq' / 'ind'."""
    same_sign = (lo > 0 and hi > 0) or (lo < 0 and hi < 0)
    substantive = same_sign and min(abs(lo), abs(hi)) >= MID
    if substantive:
        return "diff"
    if -MID <= lo and hi <= MID:
        return "eq"
    return "ind"


def e1_verdict(lo: float, hi: float) -> str:
    t = trifurcate(lo, hi)
    if t == "diff":
        return "S_same" if hi < 0 else "S_opp"   # v1 direction: Delta < 0 (arm 0 lower)
    return "EQ" if t == "eq" else "IND"


def main() -> int:
    rec = json.loads((OUT / "external_inference_v2.json").read_text(encoding="utf-8"))
    taus = sealed_tau()
    print("=== endpoint preflight (completion / saturation / seal) ===")
    preflight_checks = preflight(rec, taus)

    by_key = {a["arm"]: a for a in rec["arms"]}

    # ---- load per-run FP counts on the negative cohort, fixed image order ----
    neg_keys_ref: list = []
    fp_rows = []
    censored = {}
    for k in RUN_KEYS:
        a = by_key[k]
        tau = taus[k]
        neg = {}
        sat_at_tau = 0
        for line in (REPO / a["predictions_jsonl"]).read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            if d["is_positive"]:
                continue
            scores = np.asarray(d["scores"], dtype=np.float64)
            n_at_tau = int((scores >= tau).sum())
            neg[d["key"]] = n_at_tau
            if d["max_det_saturated"] and n_at_tau == len(scores):
                sat_at_tau += 1
        if not neg_keys_ref:
            neg_keys_ref = sorted(neg)
            if len(neg_keys_ref) != 1154:
                raise SystemExit(f"negative cohort is {len(neg_keys_ref)}, expected 1154")
        elif sorted(neg) != neg_keys_ref:
            raise SystemExit(f"[{k}] negative image set differs across runs")
        fp_rows.append(np.asarray([neg[key] for key in neg_keys_ref], dtype=np.int64))
        censored[k] = sat_at_tau
    FP = np.stack(fp_rows)                     # (9, 1154) in RUN_KEYS order
    idx_of = {k: i for i, k in enumerate(RUN_KEYS)}

    # ---- observed statistics ------------------------------------------------
    burden = {k: float(FP[idx_of[k]].mean()) for k in RUN_KEYS}
    model_mean = {m: float(np.mean([burden[f"{m}-s{s}"] for s in SEEDS])) for m in MODELS}
    d_obs = max(model_mean.values()) - min(model_mean.values())
    delta_obs = {s: burden[f"yolov8m-s{s}"] - burden[f"yolo11l-s{s}"] for s in SEEDS}

    # ---- ONE joint bootstrap over all nine runs (SV6) -----------------------
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, FP.shape[1], size=(B, FP.shape[1]))
    boot = np.empty((9, B), dtype=np.float64)
    for i in range(9):
        boot[i] = FP[i][idx].mean(axis=1)

    def ci(dist, level):
        alpha = (100.0 - level) / 2.0
        lo, hi = np.percentile(dist, [alpha, 100.0 - alpha])
        return float(lo), float(hi)

    # E1
    e1 = []
    for s in SEEDS:
        dist = boot[idx_of[f"yolov8m-s{s}"]] - boot[idx_of[f"yolo11l-s{s}"]]
        lo95, hi95 = ci(dist, 95.0)
        lo975, hi975 = ci(dist, 97.5)
        e1.append({
            "seed": s,
            "delta_obs": delta_obs[s],
            "ci95": [lo95, hi95],
            "verdict_95": e1_verdict(lo95, hi95),
            "ci975_sensitivity": [lo975, hi975],
            "verdict_975_sensitivity": e1_verdict(lo975, hi975),
        })
    verdicts95 = [x["verdict_95"] for x in e1]
    if all(v == "S_same" for v in verdicts95):
        r_branch = "R1"
    elif any(v == "S_opp" for v in verdicts95):
        r_branch = "R3"
    else:
        r_branch = "R2"
    verdicts975 = [x["verdict_975_sensitivity"] for x in e1]
    if all(v == "S_same" for v in verdicts975):
        r_branch_975 = "R1"
    elif any(v == "S_opp" for v in verdicts975):
        r_branch_975 = "R3"
    else:
        r_branch_975 = "R2"

    # E2
    mboot = np.stack([np.mean([boot[idx_of[f"{m}-s{s}"]] for s in SEEDS], axis=0)
                      for m in MODELS])                      # (3, B)
    d_dist = mboot.max(axis=0) - mboot.min(axis=0)
    d_lo95, d_hi95 = ci(d_dist, 95.0)
    d_lo975, d_hi975 = ci(d_dist, 97.5)

    def g_branch(lo, hi):
        if lo >= MID:
            return "G1"
        if hi < MID:
            return "G2"
        return "G3"

    e2 = {
        "M": list(MODELS),
        "model_mean_burden": model_mean,
        "model_argmax": max(model_mean, key=model_mean.get),
        "model_argmin": min(model_mean, key=model_mean.get),
        "D_obs": d_obs,
        "ci95": [d_lo95, d_hi95],
        "verdict_95": g_branch(d_lo95, d_hi95),
        "ci975_sensitivity": [d_lo975, d_hi975],
        "verdict_975_sensitivity": g_branch(d_lo975, d_hi975),
    }

    # descriptive, no judgment (SV6): within-model seed variation
    desc = {}
    for m in MODELS:
        vals = [burden[f"{m}-s{s}"] for s in SEEDS]
        desc[m] = {
            "per_seed": {str(s): burden[f"{m}-s{s}"] for s in SEEDS},
            "range_max_minus_min": float(max(vals) - min(vals)),
            "sd_sample": float(np.std(vals, ddof=1)),
        }

    summary = {
        "prespec": ("EXTFP_PRESPEC_v2_EXTENSION.md SV6 (frozen v2.0) + v2.1 clarification; "
                    "bootstrap/trifurcation per v1.1 7.3.1/7.3.2"),
        "tau_reference": TAU_REFERENCE,
        "tau_applied": taus,
        "mid": MID,
        "bootstrap": {
            "B": B,
            "seed": BOOT_SEED,
            "unit": "image (1,154 negatives), joint across all nine runs",
            "ci": "percentile; 95% decides, 97.5% sensitivity reported (SV6)",
            "rng": "numpy default_rng",
        },
        "n_negative_images": int(FP.shape[1]),
        "burden_per_run": burden,
        "censoring_5_3": {k: {"negative_images_saturated_at_tau": censored[k],
                              "censored": censored[k] > 0} for k in RUN_KEYS},
        "E1": {
            "definition": "Delta_r = burden(yolov8m, r) - burden(yolo11l, r), r in {0,1,2}; "
                          "v1 anchor direction Delta < 0 (arm 0 lower), anchor value -0.2114 "
                          "(seed 20260515, out of matrix, cited not recomputed)",
            "per_seed": e1,
            "branch_95": r_branch,
            "branch_975_sensitivity": r_branch_975,
        },
        "E2": e2,
        "descriptive_within_model_seed_variation": desc,
        "delta_0_1_listing": {"v1_anchor_seed20260515": V1_ANCHOR_DELTA,
                              **{f"seed{s}": delta_obs[s] for s in SEEDS}},
        "n_preflight_checks": len(preflight_checks),
    }
    (OUT / "endpoints_v2.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # ---- report -------------------------------------------------------------
    print("run          tau    FP/img   censored")
    for k in RUN_KEYS:
        print(f"{k:12s} {taus[k]:5.3f}  {burden[k]:7.4f}   "
              f"{'YES' if censored[k] else 'no'}")
    print("\nE1 (MID 0.10, 95% percentile CI, joint bootstrap B=10,000 seed 20260903):")
    for x in e1:
        print(f"  seed {x['seed']}: Delta = {x['delta_obs']:+.4f}  "
              f"CI95 [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}]  -> {x['verdict_95']}"
              f"   (97.5% [{x['ci975_sensitivity'][0]:+.4f}, {x['ci975_sensitivity'][1]:+.4f}]"
              f" -> {x['verdict_975_sensitivity']})")
    print(f"  E1 branch: {r_branch} (95%)   sensitivity 97.5%: {r_branch_975}")
    print("\nE2 (M = yolov8m/yolov9c/yolo11l, seed-mean burden):")
    for m in MODELS:
        print(f"  {m:8s}: mean {model_mean[m]:.4f}  seeds "
              + " ".join(f"{burden[f'{m}-s{s}']:.4f}" for s in SEEDS)
              + f"  range {desc[m]['range_max_minus_min']:.4f} sd {desc[m]['sd_sample']:.4f}")
    print(f"  D = {d_obs:.4f}  CI95 [{d_lo95:.4f}, {d_hi95:.4f}]  -> {e2['verdict_95']}"
          f"   (97.5% [{d_lo975:.4f}, {d_hi975:.4f}] -> {e2['verdict_975_sensitivity']})")
    print(f"\nwrote {OUT / 'endpoints_v2.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
