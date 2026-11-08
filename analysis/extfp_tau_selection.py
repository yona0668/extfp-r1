"""EXTFP public analysis copy.

EDIT analysis/public_inputs.py or set EXTFP_DATA_ROOT and EXTFP_INPUT_MAP.
Provide your local inputs and their independently recorded SHA-256 values.
See README.md for input schemas and the analysis order.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import public_inputs
import sys
from pathlib import Path

import numpy as np

REPO = public_inputs.data_root()

# ---------------------------------------------------------------- frozen constants
GRID = np.round(np.arange(1, 1000) / 1000.0, 3)   # 0.001 .. 0.999, step 0.001 (5.3, 6.2)
STORE_CONF = 0.001                                 # 5.3 store threshold
INTERNAL_NMS_IOU = 1.0                             # 5.3 no suppression inside the predictor
INTERNAL_MAX_DET = 3000                            # 5.3 >= 3000 so no internal truncation
COMMON_NMS_IOU = 0.7                               # 5.3 common class-agnostic NMS
FINAL_MAX_DET = 300                                # 5.3 step (4)
MATCH_IOU = 0.50                                   # 6.2
IMGSZ = 1280                                       # 5 training protocol

# Configure the three analysis arms and local input paths below.
ARMS = {
    "yolov8m": {
        "rung": 0,
        "loader": "YOLO",
        "run_dir": "runs/extfp/yolov8m_rung0",
        "provenance": "runs/extfp/yolov8m_rung0.provenance.jsonl",
    },
    "yolo11l": {
        "rung": 1,
        "loader": "YOLO",
        "run_dir": "runs/extfp/yolo11l_rung1",
        "provenance": "runs/extfp/yolo11l_rung1.provenance.jsonl",
    },
    "rtdetr-l": {
        "rung": 3,
        "loader": "RTDETR",
        "run_dir": "runs/extfp/rtdetr-l_rung3",
        "provenance": "runs/extfp/rtdetr-l_rung3.provenance.jsonl",
    },
}

VAL_IMAGES = REPO / "data/periapical_yolo/images/val"
VAL_LABELS = REPO / "data/periapical_yolo/labels/val"

# Any path containing one of these is external corpus territory.  Touching it
# during tau selection is a G3 failure, so we refuse to run if it appears.
EXTERNAL_MARKERS = ("muresanu", "cluj", "external")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_internal_only(*paths: Path) -> None:
    """G3 guard: nothing we read may live in the external corpus."""
    for p in paths:
        low = str(p).lower()
        for marker in EXTERNAL_MARKERS:
            if marker in low:
                raise SystemExit(f"G3 VIOLATION: external path used in tau selection: {p}")


# ---------------------------------------------------------------- geometry
def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a (N,4) xyxy, b (M,4) xyxy -> (N,M) IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(union > 0, inter / union, 0.0)
    return out


def class_agnostic_nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Greedy class-agnostic NMS, score descending. Returns kept indices."""
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = iou_matrix(boxes[i][None, :], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thr]
    return np.asarray(keep, dtype=np.int64)


def common_postprocess(boxes, scores):
    """5.3 steps (1)-(4). ONE function, identical for every arm (G6 symmetry).

    (1) class-agnostic merge -- the caller already dropped class labels
    (2) common class-agnostic NMS at IoU 0.7
    (3) score descending sort
    (4) max_det 300 truncation; report whether it actually bit
    """
    if len(boxes) == 0:
        return boxes.reshape(0, 4), scores.reshape(0), False
    keep = class_agnostic_nms(boxes, scores, COMMON_NMS_IOU)          # (1)+(2)
    boxes, scores = boxes[keep], scores[keep]
    order = np.argsort(-scores, kind="stable")                        # (3)
    boxes, scores = boxes[order], scores[order]
    saturated = len(scores) > FINAL_MAX_DET                           # (4)
    return boxes[:FINAL_MAX_DET], scores[:FINAL_MAX_DET], saturated


# ---------------------------------------------------------------- ground truth
def load_gt(label_path: Path, width: int, height: int) -> np.ndarray:
    """YOLO normalised cxcywh -> absolute xyxy. Class label ignored (6.1)."""
    if not label_path.exists():
        return np.zeros((0, 4), dtype=np.float64)
    rows = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cx, cy, w, h = (float(v) for v in parts[1:5])
        rows.append([(cx - w / 2) * width, (cy - h / 2) * height,
                     (cx + w / 2) * width, (cy + h / 2) * height])
    return np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 4), dtype=np.float64)


def greedy_match(pred_boxes: np.ndarray, pred_scores: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """6.2 greedy 1:1 matching in score-descending order.

    Predictions arrive already sorted descending.  Returns a bool array marking
    each prediction TP (True) or FP (False).

    Because the match is greedy in score order, thresholding at tau simply drops
    a suffix: the TP/FP verdict of every surviving prediction is unchanged.  So
    one pass here serves the whole grid exactly.
    """
    is_tp = np.zeros(len(pred_boxes), dtype=bool)
    if len(pred_boxes) == 0 or len(gt) == 0:
        return is_tp
    ious = iou_matrix(pred_boxes, gt)
    taken = np.zeros(len(gt), dtype=bool)
    for i in range(len(pred_boxes)):
        row = ious[i].copy()
        row[taken] = -1.0
        j = int(np.argmax(row))
        if row[j] >= MATCH_IOU:
            is_tp[i] = True
            taken[j] = True
    return is_tp


# ---------------------------------------------------------------- per-arm run
def predict_arm(arm: str, spec: dict, device: str, out_jsonl: Path) -> dict:
    from ultralytics import RTDETR, YOLO

    weights = REPO / spec["run_dir"] / "weights" / "best.pt"
    if not weights.exists():
        raise SystemExit(f"[{arm}] missing weights: {weights}")
    assert_internal_only(weights, VAL_IMAGES, VAL_LABELS)

    loader = YOLO if spec["loader"] == "YOLO" else RTDETR
    model = loader(str(weights))

    images = sorted(VAL_IMAGES.glob("*.jpg"))
    if not images:
        raise SystemExit("no val images found")

    kwargs = dict(
        imgsz=IMGSZ,
        conf=STORE_CONF,
        max_det=INTERNAL_MAX_DET,
        device=device,
        verbose=False,
        save=False,
        stream=True,
    )
    # Source must be the DIRECTORY, not a list of paths: given a list,
    # Ultralytics relabels results image0.jpg, image1.jpg... and the label file
    # for each image can no longer be found.  A guard below catches that anyway.
    if spec["loader"] == "YOLO":
        # 5.3: neutralise the predictor-internal NMS so we get raw candidates.
        # RT-DETR is NMS-free and takes no iou/agnostic argument path at all.
        kwargs["iou"] = INTERNAL_NMS_IOU
        kwargs["agnostic_nms"] = True

    all_scores = []
    all_is_tp = []
    total_gt = 0
    saturated_images = 0
    raw_candidate_total = 0
    n_seen = 0
    missing_labels = []

    with out_jsonl.open("w", encoding="utf-8") as fh:
        for res in model.predict(source=str(VAL_IMAGES), **kwargs):
            img_path = Path(res.path)
            n_seen += 1
            label_path = VAL_LABELS / (img_path.stem + ".txt")
            if not label_path.exists():
                missing_labels.append(img_path.name)
            height, width = res.orig_shape
            b = res.boxes
            if b is None or len(b) == 0:
                boxes = np.zeros((0, 4), dtype=np.float64)
                scores = np.zeros((0,), dtype=np.float64)
            else:
                boxes = b.xyxy.cpu().numpy().astype(np.float64)
                scores = b.conf.cpu().numpy().astype(np.float64)
            raw_candidate_total += len(scores)

            boxes, scores, saturated = common_postprocess(boxes, scores)
            saturated_images += int(saturated)

            gt = load_gt(label_path, width, height)
            total_gt += len(gt)
            is_tp = greedy_match(boxes, scores, gt)

            all_scores.append(scores)
            all_is_tp.append(is_tp)

            fh.write(json.dumps({
                "image": img_path.name,
                "width": int(width),
                "height": int(height),
                "n_gt": int(len(gt)),
                "max_det_saturated": bool(saturated),
                # class-agnostic (6.1): only geometry and score are kept.
                # Full precision, NOT rounded: 5.3 requires that every
                # aggregate be re-derivable from this file, and rounding scores
                # moves boxes across tau at the low-score tail.
                "boxes_xyxy": boxes.tolist(),
                "scores": scores.tolist(),
                "is_tp_iou50": [bool(v) for v in is_tp.tolist()],
            }, ensure_ascii=False) + "\n")

    # ---- fail closed. A tau computed against no ground truth is not a tau.
    if n_seen != len(images):
        raise SystemExit(f"[{arm}] saw {n_seen} results for {len(images)} val images")
    if missing_labels:
        raise SystemExit(f"[{arm}] {len(missing_labels)} images had no label file, "
                         f"e.g. {missing_labels[:5]}")
    if total_gt == 0:
        raise SystemExit(f"[{arm}] ground truth is empty -- refusing to select tau")

    scores = np.concatenate(all_scores) if all_scores else np.zeros(0)
    is_tp = np.concatenate(all_is_tp) if all_is_tp else np.zeros(0, dtype=bool)

    # micro F1 across the whole split for every tau on the grid (6.2)
    order = np.argsort(-scores, kind="stable")
    s_sorted, tp_sorted = scores[order], is_tp[order]
    cum_tp = np.cumsum(tp_sorted)
    cum_all = np.arange(1, len(s_sorted) + 1)
    # number of predictions with score >= tau
    n_at = np.searchsorted(-s_sorted, -GRID, side="right")
    tp_at = np.where(n_at > 0, cum_tp[np.clip(n_at - 1, 0, None)], 0)
    all_at = np.where(n_at > 0, cum_all[np.clip(n_at - 1, 0, None)], 0)
    fp_at = all_at - tp_at
    fn_at = total_gt - tp_at
    denom = 2 * tp_at + fp_at + fn_at
    f1 = np.where(denom > 0, 2 * tp_at / np.maximum(denom, 1), 0.0)

    best_f1 = float(f1.max())
    # 6.2 tie-break: largest tau among the maxima, then smallest grid index.
    tied = np.flatnonzero(f1 == best_f1)
    pick = int(tied[np.argmax(GRID[tied])])
    tau = float(GRID[pick])

    return {
        "arm": arm,
        "rung": spec["rung"],
        "loader": spec["loader"],
        "weights": str(weights.relative_to(REPO)).replace("\\", "/"),
        "weights_sha256": sha256_file(weights),
        "n_images": len(images),
        "n_gt_boxes": int(total_gt),
        "raw_candidates_before_common_nms": int(raw_candidate_total),
        "max_det_saturated_images": int(saturated_images),
        "tau": tau,
        "f1_at_tau": best_f1,
        "tp_at_tau": int(tp_at[pick]),
        "fp_at_tau": int(fp_at[pick]),
        "fn_at_tau": int(fn_at[pick]),
        "precision_at_tau": float(tp_at[pick] / max(tp_at[pick] + fp_at[pick], 1)),
        "recall_at_tau": float(tp_at[pick] / max(total_gt, 1)),
        "n_tau_tied_at_max_f1": int(len(tied)),
        "tau_tied_min": float(GRID[tied].min()),
        "tau_tied_max": float(GRID[tied].max()),
        "predictions_jsonl": str(out_jsonl.relative_to(REPO)).replace("\\", "/"),
        "curve_csv": None,
        "_curve": (GRID, tp_at, fp_at, fn_at, f1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default="runs/extfp/tau_selection")
    args = ap.parse_args()

    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    import ultralytics

    records = []
    for arm, spec in ARMS.items():
        print(f"[tau] {arm} ...", flush=True)
        rec = predict_arm(arm, spec, args.device, out_dir / f"val_predictions_{arm}.jsonl")
        grid, tp_at, fp_at, fn_at, f1 = rec.pop("_curve")
        csv_path = out_dir / f"val_f1_curve_{arm}.csv"
        with csv_path.open("w", encoding="utf-8") as fh:
            fh.write("tau,tp,fp,fn,precision,recall,f1\n")
            for k in range(len(grid)):
                p = tp_at[k] / max(tp_at[k] + fp_at[k], 1)
                r = tp_at[k] / max(tp_at[k] + fn_at[k], 1)
                fh.write(f"{grid[k]:.3f},{tp_at[k]},{fp_at[k]},{fn_at[k]},"
                         f"{p:.12f},{r:.12f},{f1[k]:.12f}\n")
        rec["curve_csv"] = str(csv_path.relative_to(REPO)).replace("\\", "/")
        records.append(rec)
        print(f"[tau] {arm}: tau={rec['tau']:.3f} F1={rec['f1_at_tau']:.4f} "
              f"(TP {rec['tp_at_tau']} / FP {rec['fp_at_tau']} / FN {rec['fn_at_tau']}), "
              f"saturated {rec['max_det_saturated_images']}", flush=True)

    try:
        reference_id = public_inputs.run_identifier().strip()
    except Exception:
        reference_id = None

    summary = {
        # G3 evidence: the exact invocation and the exact code that produced tau.
        "invocation": {
            "executable": sys.executable,
            "argv": sys.argv,
            "cwd": str(Path.cwd()),
            "selector_sha256": sha256_file(Path(__file__).resolve()),
        },
        "prespec": "protocols/EXTFP_PRESPEC.md",
        "prespec_sha256": sha256_file(REPO / "specifications/EXTFP_PRESPEC.md"),
        "manifest_sha256": sha256_file(REPO / "specifications/EXTFP_PROTOCOL_MANIFEST.yaml"),
        "run_id": reference_id,
        "rules": {
            "cohort": "internal val split, 379 images (6)",
            "images_dir": "data/periapical_yolo/images/val",
            "labels_dir": "data/periapical_yolo/labels/val",
            "grid": "0.001..0.999 step 0.001 (5.3, 6.2)",
            "objective": "class-agnostic micro F1, matching IoU 0.50, greedy 1:1 score-descending (6.1, 6.2)",
            "tie_break": "largest tau among maxima, then smallest grid index (6.2)",
            "postprocess": "merge -> class-agnostic NMS IoU 0.7 -> sort desc -> max_det 300 (5.3)",
            "internal_nms_neutralised": "YOLO arms: iou=1.0, agnostic=True, internal max_det=3000 (5.3)",
            "external_data_used": False,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "arms": records,
    }
    (out_dir / "tau_selection.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[tau] wrote {out_dir / 'tau_selection.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
