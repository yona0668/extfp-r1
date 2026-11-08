"""EXTFP public analysis copy.

EDIT analysis/public_inputs.py or set EXTFP_DATA_ROOT and EXTFP_INPUT_MAP.
Provide your local inputs and their independently recorded SHA-256 values.
See README.md for input schemas and the analysis order.
"""
from __future__ import annotations

import hashlib
import json
import sys
from statistics import median as _median
from pathlib import Path
import public_inputs

REPO = public_inputs.data_root()

# Use all distributed classes and images: the primary endpoint requires
# the 1,154 lesion-negative images as well as the 654 positive images.
EXTERNAL_ROOT = REPO / "data/external/_incoming/muresanu"

# 3: all four distribution folders are one evaluation set.
SPLITS = {
    "train": "train",
    "valid": "valid",
    "test": "test",
    # the fourth folder nests one level deeper in the release
    "test_alte_cabinete": "test_alte_cabinete/Ext-validation",
}
SUBGROUP_SPLIT = "test_alte_cabinete"          # 7.5 item 4

LESION_CLASS = 7                                # legend.txt: Apical periodontitis

# 8: the confusable findings, counted for the prespecified secondary
# decomposition. They are NEVER removed from the primary denominator.
CONFUSABLE_CLASSES = {
    3: "Endodontic treatment",
    5: "Bone resorbtion",
    8: "Root fragment",
    9: "Furcation lesion",
    10: "Apical surgery",
    11: "Root resorption",
}

EXPECTED = {"total": 1808, "positive": 654, "negative": 1154, "subgroup": 180}


def read_label(path: Path) -> list[tuple[int, float, float, float, float]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        rows.append((int(float(parts[0])), *(float(v) for v in parts[1:5])))
    return rows


def build() -> dict:
    """Enumerate the cohort. Sorted by (split, filename) so the order is fixed."""
    images: list[dict] = []
    for split, rel in sorted(SPLITS.items()):
        img_dir = EXTERNAL_ROOT / rel / "images"
        lbl_dir = EXTERNAL_ROOT / rel / "labels"
        if not img_dir.is_dir():
            raise SystemExit(f"external split missing: {img_dir}")
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            label_path = lbl_dir / (img.stem + ".txt")
            if not label_path.exists():
                raise SystemExit(f"no label file for external image: {img}")
            rows = read_label(label_path)
            lesion = [r for r in rows if r[0] == LESION_CLASS]
            confus = [r for r in rows if r[0] in CONFUSABLE_CLASSES]
            images.append({
                "split": split,
                "image": img.name,
                "image_path": str(img.relative_to(REPO)).replace("\\", "/"),
                "label_path": str(label_path.relative_to(REPO)).replace("\\", "/"),
                "n_boxes_total": len(rows),
                "n_lesion_boxes": len(lesion),
                "n_confusable_boxes": len(confus),
                # 3 / 7.1: negative == carries no class 7 box at all
                "is_positive": len(lesion) > 0,
                "is_subgroup": split == SUBGROUP_SPLIT,
            })

    pos = [i for i in images if i["is_positive"]]
    neg = [i for i in images if not i["is_positive"]]
    hard = [i for i in neg if i["n_confusable_boxes"] > 0]

    # identity hash: the cohort membership and its labels, in fixed order
    h = hashlib.sha256()
    for i in images:
        h.update(f"{i['split']}|{i['image']}|{i['n_lesion_boxes']}|"
                 f"{i['n_boxes_total']}|{int(i['is_positive'])}\n".encode())
    membership_sha = h.hexdigest()

    # content hash of the image bytes themselves
    hc = hashlib.sha256()
    for i in images:
        hc.update(hashlib.sha256((REPO / i["image_path"]).read_bytes()).digest())
    content_sha = hc.hexdigest()

    return {
        "prespec": "protocols/EXTFP_PRESPEC.md 3 (frozen v1.1)",
        "external_root": str(EXTERNAL_ROOT.relative_to(REPO)).replace("\\", "/"),
        "splits_folded_into_one_evaluation_set": sorted(SPLITS),
        "lesion_class": LESION_CLASS,
        "confusable_classes": {str(k): v for k, v in sorted(CONFUSABLE_CLASSES.items())},
        "counts": {
            "total": len(images),
            "positive": len(pos),
            "negative": len(neg),
            "subgroup_test_alte_cabinete": sum(i["is_subgroup"] for i in images),
            "negative_hard": len(hard),
            "negative_clean": len(neg) - len(hard),
            "total_boxes": sum(i["n_boxes_total"] for i in images),
            "lesion_boxes": sum(i["n_lesion_boxes"] for i in images),
            "empty_label_files": sum(1 for i in images if i["n_boxes_total"] == 0),
            # 0(a) audited the NEGATIVE set's annotation density, not the corpus:
            # 14,375 boxes, median 11 per image, no empty label file.
            "negative_boxes": sum(i["n_boxes_total"] for i in neg),
            "negative_boxes_median": float(_median([i["n_boxes_total"] for i in neg])),
        },
        "membership_sha256": membership_sha,
        "image_content_sha256": content_sha,
        "images": images,
    }


def check(rec: dict) -> list[tuple[str, bool, str]]:
    """Compare the enumerated cohort against the frozen 3 / 8 numbers."""
    c = rec["counts"]
    out = [
        ("3 total = 1,808", c["total"] == EXPECTED["total"], str(c["total"])),
        ("3 positive = 654", c["positive"] == EXPECTED["positive"], str(c["positive"])),
        ("3 negative = 1,154", c["negative"] == EXPECTED["negative"], str(c["negative"])),
        ("7.5 subgroup = 180", c["subgroup_test_alte_cabinete"] == EXPECTED["subgroup"],
         str(c["subgroup_test_alte_cabinete"])),
        ("0(a) negative-set boxes = 14,375", c["negative_boxes"] == 14375,
         str(c["negative_boxes"])),
        ("0(a) negative-set median = 11", c["negative_boxes_median"] == 11.0,
         str(c["negative_boxes_median"])),
        ("0(a) empty label files = 0", c["empty_label_files"] == 0, str(c["empty_label_files"])),
        ("lesion boxes = 1,259 (class 7)", c["lesion_boxes"] == 1259,
         str(c["lesion_boxes"])),
        ("8 hard negatives = 872", c["negative_hard"] == 872, str(c["negative_hard"])),
        ("8 clean negatives = 282", c["negative_clean"] == 282, str(c["negative_clean"])),
    ]
    return out


def main() -> int:
    rec = build()
    checks = check(rec)
    width = max(len(n) for n, _, _ in checks)
    failed = 0
    for name, ok, detail in checks:
        if not ok:
            failed += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {name.ljust(width)}  {detail}")
    print(f"\n  membership_sha256    {rec['membership_sha256']}")
    print(f"  image_content_sha256 {rec['image_content_sha256']}")

    out_dir = REPO / "runs/extfp/external"
    out_dir.mkdir(parents=True, exist_ok=True)
    # the per-image roster is large; keep the identity record compact and put the
    # roster beside it so both are hashable and re-checkable.
    roster = rec.pop("images")
    (out_dir / "external_cohort_identity.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (out_dir / "external_cohort_roster.jsonl").open("w", encoding="utf-8") as fh:
        for i in roster:
            fh.write(json.dumps(i, ensure_ascii=False) + "\n")

    verdict = "PASS" if failed == 0 else "FAIL"
    print(f"\nExternal cohort identity {verdict} ({len(checks) - failed}/{len(checks)})")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
