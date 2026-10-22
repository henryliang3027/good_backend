# -*- coding: utf-8 -*-
r"""
enrollment_hsv.py  ── 建立 HSV 色彩直方圖 embedding 資料庫
===========================================================
從 box_name.json 讀取品名、品牌與對應的 mask 圖片檔名，
對每張圖計算 HSV 色彩直方圖後存入 VectorDB（npz 格式）。

HSV 直方圖說明：
  H: 32 bins (0-180)  — 主要區分色相差異（紙箱顏色）
  S: 16 bins (0-256)  — 排除低飽和度雜訊
  V:  8 bins (0-256)  — 亮度參考用，影響最小
  合計特徵維度：32 + 16 + 8 = 56 維（L2 正規化）

背景排除：BGR 三通道最大值 <= 20 的像素（純黑填充）視為背景，
不納入直方圖計算，避免去背後的黑色背景污染色彩特徵。

資料夾結構：
  mask_enroll/
    1.png   2.png   ...   10.png    ← 檔名對應 box_name.json 的 "mask" 欄位

用法：
  cd /path/to/service_template
  python enrollment/enrollment_hsv.py \
      --data   enrollment/mask_enroll \
      --db     box_hsv_db.npz \
      --box-json box_name.json
"""

import argparse  # 命令列參數解析
import json      # 讀取 box_name.json
import os        # 路徑操作

import cv2       # 讀取圖片、色彩空間轉換
import numpy as np  # 數值運算

import sys
# enrollment_hsv.py 在 enroment/ 子資料夾，box_pipeline.py 也在同一層
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from box_pipeline import VectorDB  # 向量資料庫（npz 讀寫）

# ── HSV 直方圖參數 ────────────────────────────────────────────────────────────
H_BINS, S_BINS, V_BINS = 32, 16, 8   # 各通道的 bin 數，總維度 = 56


def hsv_histogram(bgr: np.ndarray) -> np.ndarray:
    """
    計算非背景區域的 HSV 色彩直方圖，L2 正規化後回傳。

    Args:
        bgr: BGR numpy array（mask 去背後，背景為純黑）

    Returns:
        shape=(56,) 的 float32 L2 正規化向量
    """
    # 前景像素遮罩：BGR 三通道最大值 > 20 視為前景
    fg_mask = bgr.max(axis=2) > 20

    # 若沒有任何前景像素（圖片全黑），回傳零向量
    if fg_mask.sum() == 0:
        return np.zeros(H_BINS + S_BINS + V_BINS, dtype=np.float32)

    # BGR → HSV 色彩空間
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # 分通道計算直方圖（只計算前景像素）
    h_hist = cv2.calcHist(
        [hsv], [0], fg_mask.astype(np.uint8), [H_BINS], [0, 180]
    ).flatten()
    s_hist = cv2.calcHist(
        [hsv], [1], fg_mask.astype(np.uint8), [S_BINS], [0, 256]
    ).flatten()
    v_hist = cv2.calcHist(
        [hsv], [2], fg_mask.astype(np.uint8), [V_BINS], [0, 256]
    ).flatten()

    # 串接三個直方圖為單一特徵向量
    feat = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)

    # L2 正規化（使餘弦相似度等於內積）
    norm = np.linalg.norm(feat)
    return feat / norm if norm > 0 else feat


def main() -> None:
    ap = argparse.ArgumentParser(description="紙箱品牌辨識：建立 HSV 色彩直方圖資料庫")
    ap.add_argument(
        "--data",
        default="enrollment/mask_enroll",
        help="mask 圖片所在資料夾（box_name.json 的 'mask' 欄位指向此資料夾的檔名）",
    )
    ap.add_argument(
        "--db",
        default="box_hsv_db.npz",
        help="輸出的向量資料庫路徑",
    )
    ap.add_argument(
        "--box-json",
        default="box_name.json",
        help="品名/品牌/mask 對應的 JSON 檔案路徑",
    )
    args = ap.parse_args()

    # ── 載入品名資料庫 JSON ──────────────────────────────────────────────────
    with open(args.box_json, encoding="utf-8") as f:
        products: list[dict] = json.load(f)["products"]
    print(f"[JSON] 讀取 {args.box_json}，共 {len(products)} 個品項")

    db = VectorDB()   # 空的向量資料庫
    total = 0         # 成功建檔的 embedding 總數

    # ── 逐一處理每個品項的 mask 圖片 ─────────────────────────────────────────
    for product in products:
        type_name  = product["name"]           # 品名，例如 "洋芋片 青檸口味"
        type_brand = product["brand"]          # 品牌，例如 "樂事"
        mask_files = product.get("mask", [])   # mask 圖片檔名列表

        if not mask_files:
            print(f"  [略過] {type_brand} {type_name}：box_name.json 無 'mask' 欄位")
            continue

        print(f"\n[品項] {type_brand} {type_name}（{len(mask_files)} 張圖）")

        for mask_filename in mask_files:
            img_path = os.path.join(args.data, mask_filename)

            if not os.path.exists(img_path):
                print(f"  ! 找不到圖片，略過：{img_path}")
                continue

            # 讀取圖片（支援中文路徑）
            bgr = cv2.imdecode(
                np.fromfile(img_path, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            if bgr is None:
                print(f"  ! 讀取失敗，略過：{img_path}")
                continue

            # 計算 HSV 直方圖特徵（56 維 L2 正規化）
            feat = hsv_histogram(bgr)

            # 存入向量資料庫（含品名與品牌）
            db.add(feat, type_name=type_name, type_brand=type_brand)
            total += 1
            print(f"  ✓ {mask_filename}  →  feat shape={feat.shape}")

    # ── 儲存 npz ─────────────────────────────────────────────────────────────
    db.save(args.db)

    # ── 結果摘要 ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  HSV 建檔完成")
    print("=" * 50)
    print(f"  輸出路徑  : {args.db}")
    print(f"  特徵維度  : {H_BINS + S_BINS + V_BINS} 維（H{H_BINS}+S{S_BINS}+V{V_BINS}）")
    print(f"  總 embedding 數 : {total}")
    print("  各品項數量：")
    for name, n in db.count_per_type().items():
        print(f"    {name}: {n} 組")


if __name__ == "__main__":
    main()
