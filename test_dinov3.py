"""
test_dinov3.py — 對圖片跑 DINOv3 並列出 box_db.npz 中 Top-K 相似品項
======================================================================
用法：
  python test_dinov3.py                              # 使用預設圖片
  python test_dinov3.py path/to/image.jpg            # 指定圖片
  python test_dinov3.py path/to/image.jpg --top 5   # 顯示 Top-5
  python test_dinov3.py path/to/image.jpg --db other.npz
"""

import argparse
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# ── 預設值 ────────────────────────────────────────────────────────────────────
DEFAULT_IMAGE  = "detected_crop/unknown_box_o1.png"
DEFAULT_DB     = "box_db.npz"
DEFAULT_MODEL  = "models/dinov3"
DEFAULT_TOP_K  = 3


def load_model(model_path: str):
    print(f"載入模型：{model_path}")
    processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)
    print(f"載入模型：processor")
    model     = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    print(f"載入模型：model")
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"載入模型：device")
    model.to(device).eval()
    print(f"裝置：{device}")
    return processor, model, device


def embed(image_bgr: np.ndarray, processor, model, device: str) -> np.ndarray:
    """BGR numpy → L2 正規化 CLS token embedding（768 維）。"""
    rgb    = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil    = Image.fromarray(rgb)
    inputs = processor(images=pil, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs)

    feat = out.last_hidden_state[:, 0, :].squeeze().cpu().float().numpy()
    norm = np.linalg.norm(feat)
    return feat / norm if norm > 0 else feat


def main():
    ap = argparse.ArgumentParser(description="DINOv3 Top-K 品項比對")
    ap.add_argument("image",   nargs="?", default=DEFAULT_IMAGE, help="輸入圖片路徑")
    ap.add_argument("--db",    default=DEFAULT_DB,    help="box_db.npz 路徑")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="DINOv3 模型資料夾")
    ap.add_argument("--top",   type=int, default=DEFAULT_TOP_K, help="顯示前 N 名")
    args = ap.parse_args()

    # ── 讀取圖片 ──────────────────────────────────────────────────────────────
    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        sys.exit(f"找不到圖片：{args.image}")
    print(f"圖片：{args.image}  ({img_bgr.shape[1]}×{img_bgr.shape[0]})")

    # ── 載入 DB ───────────────────────────────────────────────────────────────
    data        = np.load(args.db, allow_pickle=True)
    feats       = data["feats"].astype(np.float32)       # (N, 768)
    type_names  = data["type_names"].astype(str)
    type_brands = data["type_brands"].astype(str)
    print(f"資料庫：{args.db}  ({len(type_names)} 筆 embedding)\n")

    # ── 推論 ──────────────────────────────────────────────────────────────────
    processor, model, device = load_model(args.model)
    print(f"finish loading model:{args.model}\n")
    query_feat = embed(img_bgr, processor, model, device)  # (768,)

    # ── 餘弦相似度（內積，已 L2 正規化）────────────────────────────────────────
    scores   = feats @ query_feat                          # (N,)
    top_idx  = np.argsort(scores)[::-1][:args.top]        # 由高到低取 Top-K

    # ── 輸出 ──────────────────────────────────────────────────────────────────
    print(f"{'排名':<4} {'相似度':>7}  {'品牌':<12} {'品名'}")
    print("-" * 55)
    for rank, idx in enumerate(top_idx, start=1):
        score = scores[idx]
        brand = type_brands[idx]
        name  = type_names[idx]
        print(f"#{rank:<3} {score:>7.4f}  {brand:<12} {name}")


if __name__ == "__main__":
    main()
