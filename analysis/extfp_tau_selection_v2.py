"""EXTFP public analysis copy.

EDIT analysis/public_inputs.py or set EXTFP_DATA_ROOT and EXTFP_INPUT_MAP.
Provide your local inputs and their independently recorded SHA-256 values.
See README.md for input schemas and the analysis order.
"""
from __future__ import annotations

import argparse
import json
import platform
import public_inputs
import sys
from pathlib import Path

REPO = public_inputs.data_root()
sys.path.insert(0, str(Path(__file__).resolve().parent))

from extfp_tau_selection import predict_arm, sha256_file  # noqa: E402  frozen S5.3/S6 logic

# ------------------------------------------------------------ frozen document identity
# A mismatch means the frozen ground moved under us: refuse to select anything.
FROZEN_DOCS = {
    "specifications/EXTFP_PRESPEC.md":
        "c04a522514c4214b16724005d871a52695adb0557207615e031e3e4cf801bddb",
    "specifications/EXTFP_PRESPEC_v2_EXTENSION.md":
        "17af4cc43601d297df8e2fdb6379007db6969584e1397761f54a0fb7bc71bc82",
    "specifications/EXTFP_PROTOCOL_MANIFEST.yaml":
        "b2c8408d1aac5675e3e4f7d4d6e684715970721b604277efc0b07ec84700b2ee",
}

# v1 sealed property of the internal val split: 379 images, 678 class-agnostic
# GT boxes. The split is frozen,
# so every v2 run must see exactly the same ground truth.
EXPECTED_GT_BOXES = 678

# ------------------------------------------------------------ the nine canonical runs
# Order and best.pt sha256 are verbatim from the sealed training record
# frozen model order yolov8m -> yolov9c -> yolo11l within each seed.
# rung numbers follow SV2 (arm 0 = yolov8m, arm 4 = yolov9c, arm 1 = yolo11l).
# EDIT run_id to the local run identifiers; run directories follow {model}_rung{rung}__{run_id}.
RUNS = [
    dict(key="yolov8m-s0", model="yolov8m", seed=0, rung=0,
         run_id="yolov8m-s0",
         best_sha256="6515719cabd265feb4822fddec467db2adf14a3e9edb5b7ee61f1bc6228fb5b0"),
    dict(key="yolov9c-s0", model="yolov9c", seed=0, rung=4,
         run_id="yolov9c-s0",
         best_sha256="33b36a22c19bde4878e861b7acc43a0012bc9e28af0eb611e3c4e1c6059feccb"),
    dict(key="yolo11l-s0", model="yolo11l", seed=0, rung=1,
         run_id="yolo11l-s0",
         best_sha256="02432abf3e09e09ea7424076a4ad0660fc127d1a7b2aac0c0014bad69a486484"),
    dict(key="yolov8m-s1", model="yolov8m", seed=1, rung=0,
         run_id="yolov8m-s1",
         best_sha256="a92337791e502b9f948707baf3199d096d35a3a192b8d3d83ed560eb5c59b8ee"),
    dict(key="yolov9c-s1", model="yolov9c", seed=1, rung=4,
         run_id="yolov9c-s1",
         best_sha256="95e3def9d6125f4c75d21645713c79d77f7e4646963f6e257f1f76d70c7591df"),
    dict(key="yolo11l-s1", model="yolo11l", seed=1, rung=1,
         run_id="yolo11l-s1",
         best_sha256="c43281a15c1a32aa0e172b3c1e28595e98683ca6f9301e480540a2be05c2a772"),
    dict(key="yolov8m-s2", model="yolov8m", seed=2, rung=0,
         run_id="yolov8m-s2",
         best_sha256="9e04f6fccf7851d587b4523924bb6c6dc7f8934c5b04891bfd9bce0cdca020f4"),
    dict(key="yolov9c-s2", model="yolov9c", seed=2, rung=4,
         run_id="yolov9c-s2",
         best_sha256="1cc0de9ceea34c770a01eefd9f76a0511ccacb15e1b436ca092d7f2dfbd953a5"),
    dict(key="yolo11l-s2", model="yolo11l", seed=2, rung=1,
         run_id="yolo11l-s2",
         best_sha256="8b07c9458b253dc307cfbe745a87cbf71ff768914715daa766f4b784cb8847c7"),
]


def run_spec(r: dict) -> dict:
    """predict_arm() spec for one canonical run.

    All three v2 models are Detect-head YOLO family, so the SV5 rule
    "internal NMS neutralised on arms 0, 1 AND 4" maps every run to the
    loader='YOLO' branch of the frozen selector -- identical kwargs for all nine.
    """
    return {
        "rung": r["rung"],
        "loader": "YOLO",
        "run_dir": "runs/extfp/{m}_rung{g}__{i}".format(m=r["model"], g=r["rung"], i=r["run_id"]),
        "provenance": "runs/extfp/{i}.provenance.jsonl".format(i=r["run_id"]),
    }


def selector_source_path() -> Path:
    """Locate the imported selector independently of the configured data root."""
    import extfp_tau_selection
    return Path(extfp_tau_selection.__file__).resolve()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default="runs/extfp/tau_selection_v2")
    args = ap.parse_args()

    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    import ultralytics

    # ---- fail closed BEFORE any inference: frozen documents unchanged ...
    for rel, expected in FROZEN_DOCS.items():
        actual = sha256_file(REPO / rel)
        if actual != expected:
            raise SystemExit("FROZEN DOC MISMATCH: {rel}\n  sealed {e}\n  disk   {a}".format(
                rel=rel, e=expected, a=actual))
    print("[tau-v2] frozen docs verified: prespec v1.1, v2 extension, manifest", flush=True)

    # ---- ... and every best.pt is byte-identical to the sealed training record.
    mismatches = []
    for r in RUNS:
        weights = REPO / run_spec(r)["run_dir"] / "weights" / "best.pt"
        if not weights.exists():
            mismatches.append("{k}: missing {w}".format(k=r["key"], w=weights))
            continue
        actual = sha256_file(weights)
        if actual != r["best_sha256"]:
            mismatches.append("{k}: sealed {e}... disk {a}...".format(
                k=r["key"], e=r["best_sha256"][:16], a=actual[:16]))
    if mismatches:
        raise SystemExit("WEIGHTS MISMATCH vs configured weight identities:\n  " + "\n  ".join(mismatches))
    print("[tau-v2] all {n} best.pt match the sealed training record".format(n=len(RUNS)), flush=True)

    frozen_module_sha256 = sha256_file(selector_source_path())
    records = []
    for r in RUNS:
        print("[tau-v2] {k} ({i}) ...".format(k=r["key"], i=r["run_id"]), flush=True)
        spec = run_spec(r)
        rec = predict_arm(r["key"], spec, args.device,
                          out_dir / "val_predictions_{k}.jsonl".format(k=r["key"]))
        if rec["n_gt_boxes"] != EXPECTED_GT_BOXES:
            raise SystemExit("[{k}] ground truth drift: {n} boxes, sealed split has {e}".format(
                k=r["key"], n=rec["n_gt_boxes"], e=EXPECTED_GT_BOXES))
        if rec["weights_sha256"] != r["best_sha256"]:
            raise SystemExit("[{k}] weights changed between preflight and inference".format(k=r["key"]))

        grid, tp_at, fp_at, fn_at, f1 = rec.pop("_curve")
        csv_path = out_dir / "val_f1_curve_{k}.csv".format(k=r["key"])
        with csv_path.open("w", encoding="utf-8") as fh:
            fh.write("tau,tp,fp,fn,precision,recall,f1\n")
            for k in range(len(grid)):
                p = tp_at[k] / max(tp_at[k] + fp_at[k], 1)
                rr = tp_at[k] / max(tp_at[k] + fn_at[k], 1)
                fh.write("{t:.3f},{tp},{fp},{fn},{p:.12f},{r:.12f},{f:.12f}\n".format(
                    t=grid[k], tp=tp_at[k], fp=fp_at[k], fn=fn_at[k], p=p, r=rr, f=f1[k]))
        rec["curve_csv"] = str(csv_path.relative_to(REPO)).replace("\\", "/")
        rec.update(run_id=r["run_id"], model=r["model"], seed=r["seed"],
                   best_sha256_sealed=r["best_sha256"],
                   provenance=spec["provenance"])
        records.append(rec)
        print("[tau-v2] {k}: tau={t:.3f} F1={f:.4f} (TP {tp} / FP {fp} / FN {fn}), saturated {s}".format(
            k=r["key"], t=rec["tau"], f=rec["f1_at_tau"], tp=rec["tp_at_tau"],
            fp=rec["fp_at_tau"], fn=rec["fn_at_tau"], s=rec["max_det_saturated_images"]),
            flush=True)

    try:
        reference_id = public_inputs.run_identifier().strip()
    except Exception:
        reference_id = None

    summary = {
        "invocation": {
            "executable": sys.executable,
            "argv": sys.argv,
            "cwd": str(Path.cwd()),
            "selector_sha256": sha256_file(Path(__file__).resolve()),
            "frozen_module": "analysis/extfp_tau_selection.py",
            "frozen_module_sha256": frozen_module_sha256,
        },
        "prespec": "protocols/EXTFP_PRESPEC.md",
        "prespec_sha256": sha256_file(REPO / "specifications/EXTFP_PRESPEC.md"),
        "prespec_v2_extension": "protocols/EXTFP_PRESPEC_v2_EXTENSION.md",
        "prespec_v2_extension_sha256":
            sha256_file(REPO / "specifications/EXTFP_PRESPEC_v2_EXTENSION.md"),
        "manifest_sha256":
            sha256_file(REPO / "specifications/EXTFP_PROTOCOL_MANIFEST.yaml"),
        "run_id": reference_id,
        "rules": {
            "cohort": "internal val split, 379 images (6)",
            "images_dir": "data/periapical_yolo/images/val",
            "labels_dir": "data/periapical_yolo/labels/val",
            "grid": "0.001..0.999 step 0.001 (5.3, 6.2)",
            "objective": "class-agnostic micro F1, matching IoU 0.50, greedy 1:1 score-descending (6.1, 6.2)",
            "tie_break": "largest tau among maxima, then smallest grid index (6.2)",
            "postprocess": "merge -> class-agnostic NMS IoU 0.7 -> sort desc -> max_det 300 (5.3)",
            "internal_nms_neutralised": "YOLO arms: iou=1.0, agnostic=True, internal max_det=3000 (5.3; SV5 arms 0,1,4)",
            "per_run_tau": "each of the nine canonical v2 runs selects its own tau (SV5)",
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
    (out_dir / "tau_selection_v2.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[tau-v2] wrote {p}".format(p=out_dir / "tau_selection_v2.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
