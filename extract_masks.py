# -*- coding: utf-8 -*-
"""
extract_masks.py  ── 批次從 YOLO label 萃取黑底紙箱圖

用法：
  python extract_masks.py
  python extract_masks.py --images segmentation/images --labels segmentation/labels --out segmentation/mask
"""

import argparse
import os
from glob import glob

import cv2

from box_pipeline import BoxSegmenter

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def main():
    ap = argparse.ArgumentParser(description="批次從 YOLO-seg label 萃取黑底紙箱圖")
    ap.add_argument("--images", default="segmentation/images")
    ap.add_argument("--labels", default="segmentation/labels")
    ap.add_argument("--out",    default="segmentation/mask")
    args = ap.parse_args()

    img_files = []
    for ext in IMG_EXTS:
        img_files += glob(os.path.join(args.images, f"*{ext}"))
        img_files += glob(os.path.join(args.images, f"*{ext.upper()}"))
    img_files = sorted(img_files)

    if not img_files:
        raise SystemExit(f"在 {args.images} 找不到任何影像")

    total_boxes = 0
    for img_path in img_files:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(args.labels, stem + ".txt")

        if not os.path.exists(label_path):
            print(f"[略過] 找不到對應 label：{label_path}")
            continue

        image = cv2.imread(img_path)
        if image is None:
            print(f"[略過] 讀取失敗：{img_path}")
            continue

        boxes = BoxSegmenter.extract_from_label(image, label_path, args.out, stem=stem)
        print(f"{stem}：{len(boxes)} 箱")
        total_boxes += len(boxes)

    print(f"\n完成，共萃取 {total_boxes} 張黑底紙箱圖 → {args.out}")


if __name__ == "__main__":
    main()
