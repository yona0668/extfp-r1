"""EXTFP public analysis copy.

EDIT analysis/public_inputs.py or set EXTFP_DATA_ROOT and EXTFP_INPUT_MAP.
Provide your local inputs and their independently recorded SHA-256 values.
See README.md for input schemas and the analysis order.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # AC7: no bytecode in the repository

import csv
import hashlib
import json
import platform
import public_inputs
from pathlib import Path

import numpy as np

REPO = public_inputs.data_root()
_PATH_BEFORE = list(sys.path)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import extfp_external_cohort as cohort_mod  # noqa: E402
import extfp_pairwise_7274 as pw  # noqa: E402
import extfp_secondary_75 as s75  # noqa: E402  (runs sys.path.insert at import)
import extfp_tau_selection as tsel  # noqa: E402
from extfp_external_cohort import CONFUSABLE_CLASSES, LESION_CLASS, read_label  # noqa: E402
from extfp_pairwise_7274 import bootstrap, judge, load_negative_fp_counts  # noqa: E402
from extfp_secondary_75 import (  # noqa: E402
    PRESPEC_SHA_CRLF, counts_at_grid, load_gt_class7, n_at_grid, sealed_tau, sha256_file)
from extfp_tau_selection import GRID, greedy_match  # noqa: E402

IMPORTED = {
    "CONFUSABLE_CLASSES": cohort_mod, "LESION_CLASS": cohort_mod, "read_label": cohort_mod,
    "bootstrap": pw, "judge": pw, "load_negative_fp_counts": pw,
    "PRESPEC_SHA_CRLF": s75, "counts_at_grid": s75, "load_gt_class7": s75,
    "n_at_grid": s75, "sealed_tau": s75, "sha256_file": s75,
    "GRID": tsel, "greedy_match": tsel,
}

EXT = REPO / "runs/extfp/external"
OUT = REPO / "runs/extfp/posthoc_v3"
V3_PATH = REPO / "specifications/EXTFP_AMENDMENT_v3_POSTHOC_DRAFT.md"
V4_PATH = REPO / "specifications/EXTFP_AMENDMENT_v4_POSTHOC.md"
V3_SHA = "3400864c86c9fd297689976f29dfcaf58841976452a47dcf648d965c99ff97c6"      # CRLF working copy
V4_SHA_LF = "28f5ecf00810363e7aae9264216591c57b5ddf28f015be69e5ba34aa2d7e01f3"   # as committed (LF)

PAIR = ("yolov8m", "yolo11l")
ARMS = ["yolov8m", "yolo11l", "rtdetr-l"]
N_POS, N_NEG = 654, 1154
B = 10_000
SEED_S1 = 20261016
SEED_FIG = 20261028
R_LO, R_HI = 0.30, 0.60
REACH_FAIL_LIMIT = 100
TAU_R50 = {"yolov8m": 0.417, "yolo11l": 0.339, "rtdetr-l": 0.629}
TAU_R70 = {"yolov8m": 0.171, "yolo11l": 0.146, "rtdetr-l": 0.478}

LOG: list[str] = []


def log(msg: str = "") -> None:
    print(msg, flush=True)
    LOG.append(msg)


def stop(code: int, tag: str, detail: str) -> None:
    log(f"{tag}: {detail}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "posthoc_v3.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    raise SystemExit(code)


# ---------------------------------------------------------------- pure rules

def select_k(sum_tp: np.ndarray, sum_gt: np.ndarray) -> np.ndarray:
    """v3 3.3: largest grid index with 20*sum_tp >= 9*sum_gt; -1 = not reached.

    sum_tp (..., 999) integers, sum_gt (...,) integers. TP is non-increasing in
    k, so the satisfying set is a prefix and the largest index is unique.
    """
    ok = 20 * sum_tp >= 9 * sum_gt[..., None]
    assert not np.any(ok[..., 1:] & ~ok[..., :-1]), "UNIQUE_K: satisfying set is not a prefix"
    return ok.sum(axis=-1) - 1


def partial_area(recall: np.ndarray, fp: np.ndarray):
    """v4 2.2-2.3: step rule, k covers (recall(k+1), recall(k)], recall(1000) := 0."""
    nxt = np.concatenate([recall[..., 1:], np.zeros(recall.shape[:-1] + (1,))], axis=-1)
    length = np.clip(np.minimum(recall, R_HI) - np.maximum(nxt, R_LO), 0.0, None)
    a = (fp * length).sum(axis=-1)
    return a, a / (R_HI - R_LO), recall[..., 0] >= R_HI


def s1_verdict(lo, hi, n_fail: int, n_valid: int) -> str:
    """v3 3.5 table, first matching branch; exclusivity asserted."""
    if n_fail >= REACH_FAIL_LIMIT or n_valid == 0:
        return "D0"
    j, _ = judge(lo, hi)
    flags = {"A": j == "substantial difference" and hi < 0,
             "B": j == "substantial difference" and lo > 0,
             "C": j == "substantial equivalence"}
    flags["D"] = not any(flags.values())
    hit = [k for k, v in flags.items() if v]
    assert len(hit) == 1, f"VERDICT_EXCLUSIVE: {hit}"
    return hit[0]


def check_paired(idx_a: np.ndarray, idx_b: np.ndarray) -> None:
    assert np.array_equal(idx_a, idx_b), "PAIRED_INDEX: models resampled with different indices"


def check_strata(n_pos_rows: np.ndarray, n_neg_rows: np.ndarray) -> None:
    assert np.all(n_pos_rows == N_POS) and np.all(n_neg_rows == N_NEG), \
        "STRATUM_SIZE: a replicate does not hold 654 positives and 1,154 negatives"


def check_shared(digest_s1: str, digest_s2: str) -> None:
    assert digest_s1 == digest_s2, "SHARED_INDEX: S2 does not reuse the S1 P/N matrices"


def index_digest(p: np.ndarray, n: np.ndarray) -> str:
    return hashlib.sha256(p.tobytes() + n.tobytes()).hexdigest()


def draw_pn(seed: int):
    rng = np.random.default_rng(seed)
    p = rng.integers(0, N_POS, size=(B, N_POS))
    n = rng.integers(0, N_NEG, size=(B, N_NEG))
    return p, n


def expect_fail(name: str, fn) -> None:
    try:
        fn()
    except AssertionError as e:
        log(f"  [PASS] mutant {name} rejected: {str(e)[:90]}")
        return
    raise AssertionError(f"MUTATION_SURVIVED: {name}")


# ---------------------------------------------------------------- data

def load_arm(arm: str, roster: list[dict]) -> dict:
    """Per-image grid matrices in roster order (v3 3.1)."""
    recs = {}
    for line in (EXT / f"external_predictions_{arm}.jsonl").read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        recs[d["key"]] = d
    tp_rows, gt, pos_scores, pos_tp, neg_rows, neg_keys, pos_keys = [], [], [], [], [], [], []
    neg_boxes = {}
    for r in roster:
        key = f"{r['split']}/{r['image']}"
        d = recs[key]
        boxes = np.asarray(d["boxes_xyxy"], dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(d["scores"], dtype=np.float64)
        assert np.all(np.diff(scores) <= 0), f"SCORE_ORDER: {key}"
        if r["is_positive"]:
            g = load_gt_class7(REPO / r["label_path"], d["width"], d["height"])
            is_tp = greedy_match(boxes, scores, g)
            cum = np.concatenate([[0], np.cumsum(is_tp)])
            tp_rows.append(cum[n_at_grid(scores)])
            gt.append(len(g))
            pos_scores.append(scores)
            pos_tp.append(is_tp)
            pos_keys.append(key)
        else:
            neg_rows.append(n_at_grid(scores))
            neg_keys.append(key)
            neg_boxes[key] = (boxes, scores, d["width"], d["height"])
    assert (len(tp_rows), len(neg_rows)) == (N_POS, N_NEG), "COHORT_SIZE"
    return {
        "tp": np.asarray(tp_rows, dtype=np.int64),            # (654, 999)
        "nbox_pos": np.asarray([n_at_grid(s) for s in pos_scores], dtype=np.int64),
        "gt": np.asarray(gt, dtype=np.int64),                  # (654,)
        "fp": np.asarray(neg_rows, dtype=np.int64),            # (1154, 999)
        "pos_scores": np.concatenate(pos_scores), "pos_is_tp": np.concatenate(pos_tp),
        "pos_keys": pos_keys, "neg_keys": neg_keys, "neg_boxes": neg_boxes,
    }


def confusable_boxes(label_path: Path, w: int, h: int) -> np.ndarray:
    rows = [r for r in read_label(label_path) if r[0] in CONFUSABLE_CLASSES]
    return np.asarray([[(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]
                       for _, cx, cy, bw, bh in rows], dtype=np.float64).reshape(-1, 4)


def overlaps_any(boxes: np.ndarray, others: np.ndarray) -> np.ndarray:
    """v3 4.1a: intersection area > 0 in pixel xyxy."""
    if len(boxes) == 0 or len(others) == 0:
        return np.zeros(len(boxes), dtype=bool)
    iw = np.minimum(boxes[:, None, 2], others[None, :, 2]) - np.maximum(boxes[:, None, 0], others[None, :, 0])
    ih = np.minimum(boxes[:, None, 3], others[None, :, 3]) - np.maximum(boxes[:, None, 1], others[None, :, 1])
    return ((iw > 0) & (ih > 0)).any(axis=1)


def pct(v: np.ndarray) -> list[float]:
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


# ---------------------------------------------------------------- main

def main() -> int:
    log("EXTFP post-hoc v3+v4 -- POST HOC; does not change v1 P1, v2 E1/E2 or sealed files")
    log(f"python {platform.python_version()}  numpy {np.__version__}")
    added = [p for p in sys.path if p not in _PATH_BEFORE]
    log(f"AC7 sys.path entries added at import (allowed, logged): {added}")
    log(f"AC7 sys.dont_write_bytecode = {sys.dont_write_bytecode}")

    # ---- AC1 / V1
    prespec_sha = sha256_file(REPO / "specifications/EXTFP_PRESPEC.md")
    v3_sha = sha256_file(V3_PATH)
    v4_sha_lf = hashlib.sha256(V4_PATH.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    log(f"AC1/V1 prespec {prespec_sha}  v3 {v3_sha}  v4(LF) {v4_sha_lf}")
    if prespec_sha != PRESPEC_SHA_CRLF or v3_sha != V3_SHA or v4_sha_lf != V4_SHA_LF:
        stop(2, "PRESPEC_MISMATCH", "prespec or amendment hash differs")

    # ---- AC2
    for line in (EXT / "raw_external_predictions.sha256").read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        digest, name = line.split()[:2]
        if sha256_file(EXT / name) != digest:
            stop(2, "INPUT_MISMATCH", name)
    log("AC2 raw prediction hashes: PASS")

    # ---- AC7
    for name, mod in IMPORTED.items():
        assert globals()[name] is getattr(mod, name), f"AC7_LOCAL_REDEFINITION: {name}"
    log(f"AC7 {len(IMPORTED)} names bound to the sealed modules: PASS")

    roster = [json.loads(l) for l in (EXT / "external_cohort_roster.jsonl").read_text(encoding="utf-8").splitlines()]
    roster_keys = [f"{r['split']}/{r['image']}" for r in roster]
    log(f"roster order == sorted key order: {roster_keys == sorted(roster_keys)}")
    data = {arm: load_arm(arm, roster) for arm in ARMS}
    taus = sealed_tau()
    primary = {a["arm"]: a for a in json.loads((EXT / "primary_endpoint.json").read_text(encoding="utf-8"))["arms"]}
    sec75 = {a["arm"]: a for a in json.loads((EXT / "secondary_75.json").read_text(encoding="utf-8"))["arms"]}

    # ---- AC3 / AC4: full-sample curves reproduce the sealed FROC CSVs
    for arm in PAIR:
        d = data[arm]
        tp, _, fn = counts_at_grid(d["pos_scores"], d["pos_is_tp"], int(d["gt"].sum()))
        assert np.array_equal(tp, d["tp"].sum(axis=0)), "PER_IMAGE_SUM"
        fpn = d["fp"].sum(axis=0)
        with (EXT / f"secondary_75_froc_{arm}.csv").open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if len(rows) != 999 or any(int(r["tp"]) != tp[i] or int(r["fn"]) != fn[i]
                                   or int(r["total_fp_1154"]) != fpn[i] for i, r in enumerate(rows)):
            stop(3, "FROC_REPRO_FAIL", arm)
    log("AC3 999-row tp/fn/total_fp_1154 reproduced for both arms: PASS")
    ac4 = {"yolov8m": (0.268, 649, 523), "yolo11l": (0.146, 893, 605)}
    for arm, (tau, fp_t, tp_t) in ac4.items():
        k = s75.grid_index(tau)
        got = (int(data[arm]["fp"][:, k].sum()), int(data[arm]["tp"][:, k].sum()))
        assert got == (fp_t, tp_t), f"AC4 {arm}: {got}"
    log("AC4 FP 649/893, TP 523/605 at tau 0.268/0.146: PASS")

    # ---- S1 full sample + AC5
    def full_s1(selector):
        out = {}
        for arm in PAIR:
            d = data[arm]
            k = int(selector(d["tp"].sum(axis=0), np.asarray(d["gt"].sum())))
            out[arm] = (round(float(GRID[k]), 3), d["tp"][:, k].sum() / d["gt"].sum(), d["fp"][:, k].sum() / N_NEG)
        return out

    def check_ac5(res):
        want = {"yolov8m": (0.192, 0.451152, 0.782496), "yolo11l": (0.182, 0.451946, 0.614385)}
        for arm, (t, r, f) in want.items():
            assert res[arm][0] == t and round(res[arm][1], 6) == r and round(res[arm][2], 6) == f, \
                f"AC5 {arm}: {res[arm]}"

    s1_full = full_s1(select_k)
    check_ac5(s1_full)
    s1_point = s1_full["yolov8m"][2] - s1_full["yolo11l"][2]
    log(f"AC5 full-sample R*=0.45: {s1_full}: PASS  (point delta {s1_point:+.6f})")

    # ---- AC6
    keys_neg, fp_v1 = load_negative_fp_counts(taus)
    boot_v1, idx_v1 = bootstrap(fp_v1)
    d01 = float(fp_v1["yolov8m"].mean() - fp_v1["yolo11l"].mean())
    ci01 = pct(boot_v1["yolov8m"] - boot_v1["yolo11l"])
    assert (round(d01, 4), round(ci01[0], 4), round(ci01[1], 4)) == (-0.2114, -0.2556, -0.1681), f"AC6 {d01} {ci01}"
    log(f"AC6 v1 delta0-1 {d01:.4f} [{ci01[0]:.4f}, {ci01[1]:.4f}]: PASS")

    # ---- AC8
    cases = [((-0.30, -0.10, 0, 1), "A"), ((0.10, 0.30, 0, 1), "B"), ((-0.10, 0.10, 0, 1), "C"),
             ((0.02, 0.08, 0, 1), "C"), ((0.02, 0.15, 0, 1), "D"), ((-0.05, 0.15, 0, 1), "D"),
             ((0.10, 0.30, 100, 9900), "D0"), ((None, None, 10_000, 0), "D0")]
    for args, want in cases:
        assert s1_verdict(*args) == want, f"AC8 {args} -> {s1_verdict(*args)} != {want}"
    assert {w for _, w in cases} == {"A", "B", "C", "D", "D0"}
    log("AC8 verdict function: 8 cases, all five branches reached: PASS")

    # ---- V2 / V3 step integral self-check
    toy_r = np.array([0.70, 0.50, 0.50, 0.40, 0.20])
    toy_f = np.array([3.0, 2.0, 1.5, 1.0, 0.5])

    def check_v2(a, pa):
        assert abs(a - 0.55) < 1e-12 and abs(pa - 0.55 / 0.30) < 1e-12, f"V2 A={a}"

    a_toy, pa_toy, _ = partial_area(toy_r, toy_f)
    check_v2(float(a_toy), float(pa_toy))
    log(f"V2 toy A={float(a_toy):.12f} PA={float(pa_toy):.12f}: PASS")

    # ---- full-sample S2 + V4
    full_pa = {}
    for arm in PAIR:
        d = data[arm]
        rec = d["tp"].sum(axis=0) / d["gt"].sum()
        _, pa, reach = partial_area(rec, d["fp"].sum(axis=0) / N_NEG)
        assert reach, f"V4 {arm} recall(1) {rec[0]}"
        full_pa[arm] = (float(rec[0]), float(pa))
    assert (round(full_pa["yolov8m"][0], 6), round(full_pa["yolo11l"][0], 6)) == (0.690230, 0.746624), f"V4 {full_pa}"
    s2_point = full_pa["yolov8m"][1] - full_pa["yolo11l"][1]
    log(f"V4 recall(1) {full_pa['yolov8m'][0]:.6f}/{full_pa['yolo11l'][0]:.6f} reached; "
        f"PA {full_pa['yolov8m'][1]:.6f}/{full_pa['yolo11l'][1]:.6f} (delta {s2_point:+.6f})")

    # ---- paired stratified bootstrap: S1 and S2 on the same P/N
    p_idx, n_idx = draw_pn(SEED_S1)
    check_strata(np.full(B, p_idx.shape[1]), np.full(B, n_idx.shape[1]))
    assert p_idx.max() < N_POS and n_idx.max() < N_NEG, "STRATUM_SIZE: index range"
    digest_s1 = index_digest(p_idx, n_idx)
    idx_by_model = {arm: (p_idx, n_idx) for arm in PAIR}
    check_paired(idx_by_model["yolov8m"][0], idx_by_model["yolo11l"][0])
    check_paired(idx_by_model["yolov8m"][1], idx_by_model["yolo11l"][1])

    s1_rows, s2_rows = [], []
    chunk = 500
    for start in range(0, B, chunk):
        stop_ = min(start + chunk, B)
        wp = np.stack([np.bincount(p_idx[b], minlength=N_POS) for b in range(start, stop_)]).astype(np.float64)
        wn = np.stack([np.bincount(n_idx[b], minlength=N_NEG) for b in range(start, stop_)]).astype(np.float64)
        res = {}
        for arm in PAIR:
            d = data[arm]
            stp = np.rint(wp @ d["tp"]).astype(np.int64)
            sgt = np.rint(wp @ d["gt"]).astype(np.int64)
            sfp = np.rint(wn @ d["fp"]).astype(np.int64)
            k = select_k(stp, sgt)
            rows_ = np.arange(stop_ - start)
            kk = np.clip(k, 0, None)
            rec = stp / sgt[:, None]
            fpi = sfp / N_NEG
            _, pa, reach = partial_area(rec, fpi)
            res[arm] = dict(k=k, recall=rec[rows_, kk], fp=fpi[rows_, kk], pa=pa, reach=reach)
        check_shared(digest_s1, index_digest(p_idx, n_idx))
        a, b_ = res["yolov8m"], res["yolo11l"]
        for i in range(stop_ - start):
            ok1 = a["k"][i] >= 0 and b_["k"][i] >= 0
            s1_rows.append((start + i, int(a["k"][i]) + 1, int(b_["k"][i]) + 1,
                            a["recall"][i], b_["recall"][i], a["fp"][i], b_["fp"][i],
                            a["fp"][i] - b_["fp"][i] if ok1 else float("nan"), ok1))
            ok2 = bool(a["reach"][i] and b_["reach"][i])
            s2_rows.append((start + i, a["pa"][i] - b_["pa"][i] if ok2 else float("nan"),
                            a["pa"][i], b_["pa"][i], ok2))
    log(f"bootstrap done: B={B}, seed {SEED_S1}, P/N digest {digest_s1[:16]}")

    # ---- AC9 / V3 mutation tests
    log("AC9/V3 mutation tests:")
    p2, _ = draw_pn(SEED_S1 + 1)
    expect_fail("AC9-1 per-model index", lambda: check_paired(p_idx, p2))
    expect_fail("AC9-2 smallest k", lambda: check_ac5(full_s1(
        lambda t, g: np.argmax(20 * t >= 9 * g[..., None], axis=-1))))
    pooled = np.random.default_rng(SEED_S1).integers(0, N_POS + N_NEG, size=(B, N_POS + N_NEG))
    n_pos_rows = (pooled < N_POS).sum(axis=1)
    expect_fail("AC9-3 unstratified 1,808 draw", lambda: check_strata(n_pos_rows, (N_POS + N_NEG) - n_pos_rows))

    def mutant_smallest():
        a = float(sum(toy_f[0] * (min(r, R_HI) - R_LO) for r in [toy_r[0]]))
        check_v2(a, a / 0.30)

    def mutant_shifted():
        prev = np.concatenate([[toy_r[0]], toy_r[:-1]])
        length = np.clip(np.minimum(prev, R_HI) - np.maximum(toy_r, R_LO), 0, None)
        a = float((toy_f * length).sum())
        check_v2(a, a / 0.30)

    expect_fail("V3-1 smallest k (A=0.90)", mutant_smallest)
    expect_fail("V3-2 shifted step (A=0.35)", mutant_shifted)
    p3, n3 = draw_pn(SEED_S1 + 2)
    expect_fail("V3-3 S2 own P/N", lambda: check_shared(digest_s1, index_digest(p3, n3)))

    # ---- S1 summary and verdict
    s1_valid = np.array([r[7] for r in s1_rows if r[8]])
    s1_fail = B - len(s1_valid)
    s1_ci = pct(s1_valid) if len(s1_valid) else None
    verdict = s1_verdict(*(s1_ci or (None, None)), s1_fail, len(s1_valid))

    # ---- S2 summary
    s2_valid = [r for r in s2_rows if r[4]]
    s2_fail = B - len(s2_valid)
    if s2_fail >= REACH_FAIL_LIMIT:
        s2 = {"status": "PA_REACH_FAIL", "reach_failures": s2_fail}
        s2_line = "S2_CI=PA_REACH_FAIL"
    else:
        dpa = np.array([r[1] for r in s2_valid])
        s2 = {"status": "OK", "reach_failures": s2_fail, "n_valid": len(s2_valid),
              "point_pa": {a: full_pa[a][1] for a in PAIR}, "point_delta": s2_point,
              "ci95_delta": pct(dpa),
              "ci95_pa": {"yolov8m": pct(np.array([r[2] for r in s2_valid])),
                          "yolo11l": pct(np.array([r[3] for r in s2_valid]))}}
        s2_line = f"S2_CI={s2['ci95_delta'][0]:.6f},{s2['ci95_delta'][1]:.6f}"

    # ---- v3 4.1a box decomposition, three arms
    lab = {f"{r['split']}/{r['image']}": r for r in roster}
    decomp, cat1_images_v8m = {}, []
    for arm in ARMS:
        n1 = n2 = 0
        for key, (boxes, scores, w, h) in data[arm]["neg_boxes"].items():
            keep = boxes[scores >= taus[arm]]
            ov = overlaps_any(keep, confusable_boxes(REPO / lab[key]["label_path"], w, h))
            n1 += int(ov.sum())
            n2 += int((~ov).sum())
            if arm == "yolov8m" and ov.any():
                cat1_images_v8m.append(key)
        assert n1 + n2 == primary[arm]["total_false_positives"], f"DECOMP_TOTAL {arm}"
        decomp[arm] = {"tau": taus[arm], "overlap_confusable": n1, "other": n2,
                       "overlap_per_image": n1 / N_NEG, "other_per_image": n2 / N_NEG,
                       "clean_negative_fp_per_image_sealed": primary[arm]["secondary"]["clean_fp_per_image_mean"]}

    # ---- v3 4.2 subgroup
    sub = {}
    sub_pos = [i for i, k in enumerate(data["yolov8m"]["pos_keys"]) if lab[k]["is_subgroup"]]
    sub_neg = [i for i, k in enumerate(data["yolov8m"]["neg_keys"]) if lab[k]["is_subgroup"]]
    assert (len(sub_pos), len(sub_neg)) == (48, 132), "SUBGROUP_SIZE"
    for arm in ARMS:
        k = s75.grid_index(taus[arm])
        d = data[arm]
        sub[arm] = {"negative_fp_per_image_132": float(d["fp"][sub_neg, k].mean()),
                    "positive_lesion_recall_48": float(d["tp"][sub_pos, k].sum() / d["gt"][sub_pos].sum())}
        assert abs(sub[arm]["negative_fp_per_image_132"]
                   - primary[arm]["secondary"]["subgroup_fp_per_image_mean"]) < 1e-12, f"SUBGROUP_FP {arm}"

    # ---- v4 4: positive-image FP and image-level specificity (+ v3 4.1d, v4 5 Fig 2)
    fp_pos, spec = {}, {}
    for arm in ARMS:
        k = s75.grid_index(taus[arm])
        d = data[arm]
        fpp = (d["nbox_pos"][:, k] - d["tp"][:, k]).astype(np.float64)
        tp_sum, fp_sum = int(d["tp"][:, k].sum()), int(fpp.sum())
        item1 = sec75[arm]["item1_positive_654"]["at_operating_point_tau"]
        if (tp_sum, fp_sum) != (item1["tp"], item1["fp"]):
            stop(3, "V6_FAIL", f"{arm} {tp_sum}/{fp_sum}")
        fp_pos[arm] = fpp
    want_zero = {"yolov8m": 718, "yolo11l": 650, "rtdetr-l": 549}
    for arm in ARMS:
        z = (fp_v1[arm] == 0).astype(np.float64)
        if int(z.sum()) != want_zero[arm]:
            stop(3, "V7_FAIL", f"{arm} {int(z.sum())}")
        spec[arm] = z
    log("V6 TP 523/605/539, FPpos 689/946/782; V7 zero-FP negatives 718/650/549: PASS")
    boot_pos, _ = bootstrap(fp_pos)
    boot_spec, idx_spec = bootstrap(spec)
    assert np.array_equal(idx_spec, idx_v1), "V8_INDEX: specificity index differs"
    pos_out = {arm: {"fp_per_positive_image": float(fp_pos[arm].mean()), "ci95": pct(boot_pos[arm]),
                     "tp": int(data[arm]["tp"][:, s75.grid_index(taus[arm])].sum()),
                     "fp_boxes": int(fp_pos[arm].sum())} for arm in ARMS}
    spec_out = {arm: {"image_specificity": float(spec[arm].mean()), "ci95": pct(boot_spec[arm]),
                      "zero_fp_images": int(spec[arm].sum())} for arm in ARMS}

    fig2 = {}
    sets = {"prespecified": taus, "internal_recall_0.50": TAU_R50, "internal_recall_0.70": TAU_R70}
    for label, tset in sets.items():
        for arm in ARMS:
            src = {"internal_recall_0.50": 0.5, "internal_recall_0.70": 0.7}.get(label)
            if src is not None:
                pt = next(p for p in sec75[arm]["item6_tau_sensitivity"]["points"]
                          if p["target_internal_recall"] == src)
                assert pt["tau"] == tset[arm], f"V8_TAU {label} {arm}"
    want_tot = {"internal_recall_0.50": {"yolov8m": 349, "yolo11l": 278, "rtdetr-l": 412},
                "internal_recall_0.70": {"yolov8m": 986, "yolo11l": 893, "rtdetr-l": 990}}
    for label, tset in sets.items():
        if label == "prespecified":
            keys_t, fp_t, boot_t, idx_t = keys_neg, fp_v1, boot_v1, idx_v1
        else:
            keys_t, fp_t = load_negative_fp_counts(tset)
            boot_t, idx_t = bootstrap(fp_t)
        assert keys_t == keys_neg and np.array_equal(idx_t, idx_v1), f"V8_INDEX {label}"
        for arm in ARMS:
            tot = int(fp_t[arm].sum())
            if label == "prespecified":
                assert abs(fp_t[arm].mean() - primary[arm]["fp_per_image_mean"]) < 1e-12, f"V8_PRIMARY {arm}"
            else:
                assert tot == want_tot[label][arm], f"V8_TOTAL {label} {arm} {tot}"
            fig2.setdefault(label, {})[arm] = {"tau": tset[arm], "fp_per_image": float(fp_t[arm].mean()),
                                               "total_fp": tot, "ci95": pct(boot_t[arm])}
    log("V8 Fig 2 totals, prespecified means and shared index matrix: PASS")

    # ---- v4 3 Welch
    import scipy
    from scipy import stats  # stop here if scipy is unusable (no approximation)
    seeds = json.loads((REPO / "runs/extfp/external_v2/endpoints_v2.json").read_text(encoding="utf-8"))[
        "descriptive_within_model_seed_variation"]
    xa = np.array(list(seeds["yolov8m"]["per_seed"].values()))
    xb = np.array(list(seeds["yolo11l"]["per_seed"].values()))
    va, vb = xa.var(ddof=1) / 3, xb.var(ddof=1) / 3
    se = float(np.sqrt(va + vb))
    nu = float(se ** 4 / (va ** 2 / 2 + vb ** 2 / 2))
    t2 = float(stats.t.ppf(0.975, 2))
    tq = float(stats.t.ppf(0.975, nu))
    dw = float(xa.mean() - xb.mean())
    assert (round(xa.mean(), 6), round(xb.mean(), 6), round(dw, 10), round(t2, 6)) == \
        (0.629694, 0.689197, -0.0595031774, 4.302653), f"V5 {xa.mean()} {xb.mean()} {dw} {t2}"
    welch = {"mean": {"yolov8m": float(xa.mean()), "yolo11l": float(xb.mean())},
             "sd": {"yolov8m": float(xa.std(ddof=1)), "yolo11l": float(xb.std(ddof=1))},
             "difference": dw, "se": se, "df_welch": nu, "t_0975": tq,
             "ci95": [dw - tq * se, dw + tq * se], "scipy": scipy.__version__}
    log(f"V5 Welch self-check: PASS (t(0.975,2)={t2:.6f})")

    # ---- v4 7 FP example selection (keys only; figures are drawn later)
    def select_examples():
        pool_a = [k for i, k in enumerate(data["yolov8m"]["neg_keys"]) if fp_v1_roster[i] >= 1]
        rng = np.random.default_rng(SEED_FIG)
        pick_a = [pool_a[i] for i in rng.choice(len(pool_a), size=2, replace=False)]
        pool_b = [k for k in cat1_roster if k not in pick_a]
        pick_b = [pool_b[int(i)] for i in rng.choice(len(pool_b), size=1)] if pool_b else []
        return len(pool_a), len(pool_b), pick_a, pick_b

    k268 = s75.grid_index(taus["yolov8m"])
    fp_v1_roster = data["yolov8m"]["fp"][:, k268]
    cat1_set = set(cat1_images_v8m)
    cat1_roster = [k for k in data["yolov8m"]["neg_keys"] if k in cat1_set]
    sel1, sel2 = select_examples(), select_examples()
    assert sel1 == sel2 and sel1[0] == 436, f"V9 {sel1[:2]}"
    log(f"V9 FP examples (logged before any figure): pool A {sel1[0]}, pool B {sel1[1]}, "
        f"A picks {sel1[2]}, B pick {sel1[3]}")

    # ---- outputs (AC10 / V10)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "s1_replicates.csv").open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("b,k_yolov8m,k_yolo11l,recall_yolov8m,recall_yolo11l,fp_yolov8m,fp_yolo11l,delta,reached\n")
        for r in s1_rows:
            fh.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.10f},{r[4]:.10f},{r[5]:.10f},{r[6]:.10f},{r[7]:.10f},{int(r[8])}\n")
    with (OUT / "s2_replicates.csv").open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("b,delta_pa,pa_yolov8m,pa_yolo11l,reached\n")
        for r in s2_rows:
            fh.write(f"{r[0]},{r[1]:.10f},{r[2]:.10f},{r[3]:.10f},{int(r[4])}\n")

    head = public_inputs.run_identifier().strip()
    v3_out = {
        "STATUS": "POST HOC (v3 amendment); does not change v1 P1 or v2 E1/E2",
        "amendment": "protocols/EXTFP_AMENDMENT_v3_POSTHOC_DRAFT.md", "amendment_sha256": v3_sha,
        "run_id": head, "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "s1": {"R_star": 0.45, "seed": SEED_S1, "B": B, "full_sample": {a: {"tau": v[0], "recall": v[1], "fp_per_image": v[2]}
                                                                    for a, v in s1_full.items()},
               "point_delta": s1_point, "reach_failures": s1_fail, "n_valid": int(len(s1_valid)),
               "ci95": s1_ci, "verdict": verdict, "index_digest": digest_s1},
        "s4_1a_box_decomposition": decomp,
        "s4_1d_arm_ci_prespecified_tau": fig2["prespecified"],
        "s4_2_subgroup_test_alte_cabinete": sub,
        "ac6_v1_reproduction": {"delta": d01, "ci95": ci01},
    }
    v4_out = {
        "STATUS": "POST HOC (v4 amendment); no verdicts",
        "amendment": "protocols/EXTFP_AMENDMENT_v4_POSTHOC.md", "amendment_sha256_lf": v4_sha_lf,
        "run_id": head,
        "s2_partial_area": {"interval": [R_LO, R_HI], "rule": "step, largest k", "seed": SEED_S1, "B": B,
                            "shares_s1_index": digest_s1, **s2},
        "s3_welch_seed_mean": welch,
        "s4_positive_image_fp": pos_out, "s4_image_specificity": spec_out,
        "s5_fig2_ci": fig2,
        "s7_fp_examples": {"seed": SEED_FIG, "pool_a": sel1[0], "pool_b": sel1[1],
                           "pool_a_picks": sel1[2], "pool_b_pick": sel1[3]},
        "s6_item3_sentence": ("The internal-to-external generalisation gap named in the protocol was not computed: "
                              "the protocol defines no metric for it, and computing one would have required new "
                              "inference on the internal test split, which this revision does not perform."),
    }
    (OUT / "posthoc_v3.json").write_text(json.dumps(v3_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "posthoc_v4.json").write_text(json.dumps(v4_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    log("")
    log(f"S1 full-sample delta {s1_point:+.6f}; reach failures {s1_fail}; CI {s1_ci}")
    log(f"S1_VERDICT={verdict}")
    log(f"S2 point PA {full_pa['yolov8m'][1]:.6f}/{full_pa['yolo11l'][1]:.6f}; reach failures {s2_fail}")
    log(s2_line)
    log(f"Welch d {dw:.6f}, SE {se:.6f}, df {nu:.4f}, CI [{welch['ci95'][0]:.6f}, {welch['ci95'][1]:.6f}]")
    (OUT / "posthoc_v3.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    names = ["posthoc_v3.json", "posthoc_v4.json", "s1_replicates.csv", "s2_replicates.csv", "posthoc_v3.log"]
    (OUT / "posthoc_v3.sha256").write_text(
        "".join(f"{sha256_file(OUT / n)}  {n}\n" for n in names), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
