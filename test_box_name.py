"""
test_box_name.py — YOLO seg inference → crop+背景去除 → GLM-OCR 文字擷取

前置條件：
  - llama-server 已在外部啟動並監聽 LLAMA_CPP_URL (預設 http://localhost:8000/v1)
  - YOLO 模型路徑 MODEL_PATH 存在

用法：
  python test_box_name.py
  LLAMA_CPP_URL=http://localhost:8000/v1 python test_box_name.py
"""

import base64
import io
import json
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from date_extractor import extract_dates

# ── Config ────────────────────────────────────────────────────────────────────
YOLO_MODEL_PATH = os.getenv(
    "YOLO_MODEL_PATH",
    "/home/ubuntu/Documents/Github/service_template/models/yolo_model/box_segmentation/best_26x_seg_20260702.pt",
)
CONF_THRESHOLD = 0.65

LLAMA_CPP_URL = os.getenv("LLAMA_CPP_URL", "http://localhost:8000/v1")
OCR_PROMPT = os.getenv("OCR_PROMPT", "請對這張圖片進行 OCR，輸出圖中所有文字。")

IMAGE_LIST = [
    "/home/ubuntu/Documents/Github/service_template/test_images/test2.jpg"
    # "/home/ubuntu/Documents/Github/service_template/test_images/20260702_112112_jpg.rf.d1876f0eb76cfcb49cfb446fb8c151f6.jpg",
    # "/home/ubuntu/Documents/Github/service_template/test_images/ct4.jpg",
    # "/home/ubuntu/Documents/Github/service_template/test_images/20260627_122740.jpg",
    # "/home/ubuntu/Documents/Github/service_template/test_images/1782439930281.jpg",
]

BOX_NAME_JSON = os.getenv(
    "BOX_NAME_JSON",
    os.path.join(os.path.dirname(__file__), "box_name.json"),
)

OUTPUT_DIR = Path(os.path.dirname(__file__)) / "detected_box_name"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_SIZE = 28

# ── Helpers ───────────────────────────────────────────────────────────────────
def _random_colors(n: int) -> list[tuple]:
    colors, used = [], set()
    while len(colors) < n:
        c = (random.randint(80, 255), random.randint(80, 255), random.randint(80, 255))
        if c not in used:
            used.add(c)
            colors.append(c)
    return colors


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(_FONT_PATH, size)
    except Exception:
        return ImageFont.load_default()


def draw_results(
    img_bgr: np.ndarray,
    polygons: list,
    labels: list[str],
    alpha: float = 0.4,
) -> Image.Image:
    """Mask 疊色 + 輪廓 + 品名文字，回傳 PIL RGB Image。"""
    colors = _random_colors(len(polygons))
    overlay = img_bgr.copy()

    for pts, color in zip(polygons, colors):
        cv2.fillPoly(overlay, [pts.astype(np.int32)], color)

    blended = cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0)

    for pts, color in zip(polygons, colors):
        cv2.polylines(blended, [pts.astype(np.int32)], isClosed=True, color=color, thickness=2)

    pil_img = Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = _load_font(FONT_SIZE)

    for pts, color, label in zip(polygons, colors, labels):
        pts_int = pts.astype(np.int32)
        x, y, bw, bh = cv2.boundingRect(pts_int)

        # 文字背景框
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx, ty = x, max(0, y - th - 6)
        draw.rectangle([tx, ty + th , tx + tw + 6, ty + 2*th + 6], fill=(0, 0, 0, 180))
        draw.text((tx + 3, ty + th + 3), label, fill=color, font=font)

    return pil_img


# ── Product DB ────────────────────────────────────────────────────────────────
def load_products(json_path: str) -> list[dict]:
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)["products"]


def match_product(ocr_text: str, products: list[dict]) -> dict | None:
    """最多 keywords 命中的產品勝出，平手取第一筆。"""
    best, best_score = None, 0
    for product in products:
        score = sum(1 for kw in product["keywords"] if kw in ocr_text)
        if score > best_score:
            best, best_score = product, score
    return best if best_score > 0 else None


# ── GLM-OCR client ────────────────────────────────────────────────────────────
_client = OpenAI(base_url=LLAMA_CPP_URL, api_key="no-key-needed")


def _pil_to_base64(image: Image.Image, fmt: str = "JPEG") -> str:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def call_glm_ocr(image: Image.Image) -> tuple[str, float]:
    """Returns (ocr_text, elapsed_seconds)."""
    b64 = _pil_to_base64(image)
    t0 = time.time()
    resp = _client.chat.completions.create(
        model="glm-ocr",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": OCR_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }
        ],
        temperature=0,
        max_tokens=2048,
    )
    elapsed = time.time() - t0
    return resp.choices[0].message.content, elapsed


# ── Mask crop + 去背 ──────────────────────────────────────────────────────────
def crop_with_mask(img_bgr: np.ndarray, polygon: np.ndarray) -> Image.Image:
    """
    給定原圖與 polygon (N,2) float，回傳：
      - 以 polygon 最小 bounding box 裁切
      - bounding box 外的背景設為黑色
    的 PIL Image (RGB)。
    """
    pts = polygon.astype(np.int32)
    h, w = img_bgr.shape[:2]

    # 建立單通道 mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    # 去背：mask 外設為黑
    masked = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)

    # 最小 bounding box
    x, y, bw, bh = cv2.boundingRect(pts)
    cropped = masked[y : y + bh, x : x + bw]

    return Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))


# ── Main pipeline ─────────────────────────────────────────────────────────────
def process_image(image_path: str, model: YOLO) -> None:
    print(f"\n{'='*60}")
    print(f"圖片：{image_path}")

    # YOLO inference
    t_infer_start = time.time()
    results = model.predict(
        image_path, task="seg", imgsz=640, conf=CONF_THRESHOLD, verbose=False
    )
    t_infer = time.time() - t_infer_start
    print(f"[YOLO] infer={t_infer:.3f}s")

    result = results[0]
    if result.masks is None:
        print("未偵測到任何 box")
        return

    polygons = result.masks.xy          # list of (N,2) float arrays
    confs    = result.boxes.conf.cpu().numpy()
    img_bgr  = cv2.imread(image_path)
    total    = len(polygons)
    products = load_products(BOX_NAME_JSON)

    print(f"偵測到 {total} 個 box\n")

    labels = []
    for idx, (polygon, conf) in enumerate(zip(polygons, confs), start=1):
        print(f"  ── Box {idx}/{total}  conf={conf:.2f}")

        cropped_pil = crop_with_mask(img_bgr, polygon)

        ocr_text, ocr_elapsed = call_glm_ocr(cropped_pil)
        # print(f"     GLM-OCR ({ocr_elapsed:.2f}s)：{ocr_text.strip()}")

        matched = match_product(ocr_text, products)
        name_line = (
            f"{matched['brand']} {matched['name']} ({conf:.2f})"
            if matched else f"未知品項 ({conf:.2f})"
        )
        print(f"     比對結果：{name_line}")

        dates = extract_dates(ocr_text)
        date_lines = []
        if dates["expiry"]:
            d = dates["expiry"]
            s = f"有效日期：{d['year']}/{d['month']}/{d['day']}"
            date_lines.append(s)
            print(f"     {s}")
        if dates["manufacture"]:
            d = dates["manufacture"]
            s = f"製造日期：{d['year']}/{d['month']}/{d['day']}"
            date_lines.append(s)
            print(f"     {s}")

        label = "\n".join([name_line] + date_lines)
        labels.append(label)

    # 繪製結果圖並儲存
    result_img = draw_results(img_bgr, polygons, labels)
    out_path = OUTPUT_DIR / Path(image_path).name
    result_img.save(str(out_path), "JPEG", quality=95)
    print(f"\n結果圖已儲存：{out_path}")
    print(f"總計偵測：{total} 個 box")


def main() -> None:
    print(f"載入 YOLO 模型：{YOLO_MODEL_PATH}")
    print(f"載入品名資料庫：{BOX_NAME_JSON} ({len(load_products(BOX_NAME_JSON))} 筆)")
    model = YOLO(YOLO_MODEL_PATH)

    for image_path in IMAGE_LIST:
        if not Path(image_path).exists():
            print(f"[略過] 找不到圖片：{image_path}")
            continue
        process_image(image_path, model)


if __name__ == "__main__":
    main()
