#!/usr/bin/env python3
"""
prepare_data.py — turn the Magnetic Tile pixel masks into a YOLO detection set.

The dataset ships segmentation masks, not boxes. We take the tight bounding box of
each connected mask region as one detection over five classes (blowhole, break,
crack, fray, uneven). Defect-free tiles are kept as negatives with empty labels, so
the detector also learns what a clean surface looks like.

Stratified 70/20/10 split per defect type (seeded); the output tree is wiped first
so re-running replaces the split rather than overlaying it.

    python prepare_data.py --src Magnetic-Tile-Defect --out tile_yolo
"""
import argparse
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

DEFECT_TYPES = ["MT_Blowhole", "MT_Break", "MT_Crack", "MT_Fray", "MT_Uneven"]
NORMAL_TYPE = "MT_Free"
CLASS_NAMES = ["blowhole", "break", "crack", "fray", "uneven"]
TYPE_TO_ID = {t: i for i, t in enumerate(DEFECT_TYPES)}

MIN_AREA = 20  # px; below this a component is annotation speckle / JPEG fringe, not a defect


def mask_to_boxes(mask_path, img_w, img_h, cls_id):
    """One YOLO box per connected mask region; [] when the mask is empty or missing."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []
    if mask.shape[:2] != (img_h, img_w):
        mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    n, _, stats, _ = cv2.connectedComponentsWithStats((mask > 127).astype(np.uint8))
    lines = []
    for i in range(1, n):  # skip background component 0
        if stats[i, cv2.CC_STAT_AREA] < MIN_AREA:
            continue
        x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        lines.append(
            f"{cls_id} {(x + w / 2) / img_w:.6f} {(y + h / 2) / img_h:.6f} "
            f"{w / img_w:.6f} {h / img_h:.6f}"
        )
    return lines


def find_mask(img_path):
    """The .png segmentation mask sits next to the .jpg in Imgs/."""
    mask = img_path.with_suffix(".png")
    return mask if mask.exists() else None


def list_images(folder):
    imgs = folder / "Imgs"
    return sorted(imgs.glob("*.jpg")) if imgs.exists() else []


def collect(src):
    samples, fallback = [], 0
    for dtype in DEFECT_TYPES:
        folder = src / dtype
        cls_id = TYPE_TO_ID[dtype]
        imgs = list_images(folder)
        if not imgs:
            print(f"  WARNING: no images in {folder}")
            continue
        for img_path in imgs:
            im = cv2.imread(str(img_path))
            if im is None:
                continue
            h, w = im.shape[:2]
            mask = find_mask(img_path)
            lines = mask_to_boxes(mask, w, h, cls_id) if mask else []
            if not lines:
                # mask present but empty after thresholding — the tile is labelled
                # defective, so box the whole frame rather than drop the sample
                lines = [f"{cls_id} 0.5 0.5 0.95 0.95"]
                fallback += 1
            samples.append(dict(img=img_path, label="\n".join(lines), dtype=dtype))
    for img_path in list_images(src / NORMAL_TYPE):
        samples.append(dict(img=img_path, label="", dtype=NORMAL_TYPE))
    print(f"Collected {len(samples)} samples (whole-image fallback fired {fallback}x)")
    return samples


def stratified_split(samples, seed):
    rng = random.Random(seed)
    by = defaultdict(list)
    for s in samples:
        by[s["dtype"]].append(s)
    out = {"train": [], "val": [], "test": []}
    for items in by.values():
        rng.shuffle(items)
        n = len(items)
        out["train"] += items[: int(0.7 * n)]
        out["val"] += items[int(0.7 * n) : int(0.9 * n)]
        out["test"] += items[int(0.9 * n) :]
    for k in out:
        rng.shuffle(out[k])
    return out


def write(out_root, splits):
    # Wipe first: a re-run must replace the split, never overlay it. Overlaying a
    # second split onto an old one is how the same tile ends up in train and test.
    for sub in ["images", "labels"]:
        if (out_root / sub).exists():
            shutil.rmtree(out_root / sub)
    for split, items in splits.items():
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for s in items:
            name = f"{s['dtype']}_{s['img'].stem}"
            shutil.copy(s["img"], out_root / "images" / split / (name + ".jpg"))
            (out_root / "labels" / split / (name + ".txt")).write_text(s["label"])
        print(f"  {split}: {len(items)}")
    yaml = out_root / "tile.yaml"
    yaml.write_text(
        f"path: {out_root.resolve()}\ntrain: images/train\nval: images/val\n"
        f"test: images/test\n\nnc: {len(CLASS_NAMES)}\nnames: {CLASS_NAMES}\n"
    )
    print(f"tile.yaml -> {yaml}  (nc={len(CLASS_NAMES)}, names={CLASS_NAMES})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="Magnetic-Tile-Defect", type=Path)
    ap.add_argument("--out", default="tile_yolo", type=Path)
    ap.add_argument("--seed", default=42, type=int)
    args = ap.parse_args()

    samples = collect(args.src)
    if not samples:
        raise SystemExit("No samples found — check --src path and folder names.")
    splits = stratified_split(samples, args.seed)
    write(args.out, splits)

    counts = Counter(
        CLASS_NAMES[TYPE_TO_ID[s["dtype"]]] if s["dtype"] in TYPE_TO_ID else "good(neg)"
        for s in samples
    )
    print("\nPer-class image counts:")
    for k in CLASS_NAMES + ["good(neg)"]:
        print(f"  {k:12s} {counts.get(k, 0)}")


if __name__ == "__main__":
    main()
