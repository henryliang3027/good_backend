import random
import time
from pathlib import Path

import cv2
import numpy as np

from ultralytics import YOLO


# ========== Model Config ==========
MODEL_PATH = "/home/ubuntu/Documents/Github/service_template/models/yolo_model/box_segmentation/best_26x_seg_20260702.pt"
CONF_THRESHOLD = 0.65
ALPHA = 0.5  # mask overlay transparency


# ========== Output Dir ==========
OUTPUT_DIR = Path("/home/ubuntu/Documents/Github/service_template/detected_seg_box")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def random_colors(n: int):
    colors = []
    used = set()
    while len(colors) < n:
        c = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        if c not in used:
            used.add(c)
            colors.append(c)
    return colors


def detect_seg(image_path: str, model: YOLO):
    t_start = time.time()

    results = model.predict(
        image_path, task="seg", imgsz=640, conf=CONF_THRESHOLD, verbose=False
    )
    t_infer = time.time()
    print(f"[YOLO seg] infer={round(t_infer - t_start, 3)}s", end="")

    img = cv2.imread(image_path)
    result = results[0]

    count = 0
    if result.masks is not None:
        polygons = result.masks.xy
        confs = result.boxes.conf.cpu().numpy()
        colors = random_colors(len(polygons))

        overlay = img.copy()
        for pts, color in zip(polygons, colors):
            pts = pts.astype(np.int32)
            cv2.fillPoly(overlay, [pts], color=color)

        img = cv2.addWeighted(overlay, ALPHA, img, 1 - ALPHA, 0)

        for pts, color, conf in zip(polygons, colors, confs):
            pts = pts.astype(np.int32)
            cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
            label = f"{conf:.2f}"
            label_pt = tuple(pts[pts[:, 1].argmin()])
            cv2.putText(
                img, label, label_pt,
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA,
            )
            count += 1

    print(f", found={count}")

    out_path = OUTPUT_DIR / Path(image_path).name
    cv2.imwrite(str(out_path), img)
    print(f"  saved → {out_path}  total={round(time.time()-t_start, 3)}s")


if __name__ == "__main__":
    model = YOLO(MODEL_PATH)
    print(f"✅ Seg YOLO 模型載入完成: {MODEL_PATH}")

    image_list = [
        "/home/ubuntu/Documents/Github/service_template/test_images/20260702_112112_jpg.rf.d1876f0eb76cfcb49cfb446fb8c151f6.jpg",
        "/home/ubuntu/Documents/Github/service_template/test_images/ct4.jpg"
    ]

    for image_path in image_list:
        detect_seg(image_path, model)
