# -*- coding: utf-8 -*-
r"""
enrollment.py  ── 建立 DINOv3 特徵 embedding 資料庫
=====================================================
從 box_name.json 讀取品名、品牌與對應的 mask 圖片檔名，
對每張圖跑 DINOv3 取得 embedding，存入 VectorDB（npz 格式）。

npz 每筆記錄包含：
  feats       — DINOv3 embedding（768 維，L2 正規化）
  type_names  — 對應 box_name.json 的 "name" 欄位（品名）
  type_brands — 對應 box_name.json 的 "brand" 欄位（品牌）

資料夾結構：
  mask_enroll/
    1.png   2.png   ...   10.png    ← 檔名對應 box_name.json 的 "mask" 欄位

用法：
  cd /path/to/service_template
  python enrollment/enrollment.py \
      --data   enrollment/mask_enroll \
      --db     box_db.npz \
      --box-json box_name.json \
      --model-id models/dinov3
"""

import argparse  # 命令列參數解析
import json      # 讀取 box_name.json
import os        # 路徑操作
import sys       # 加入專案根目錄到搜尋路徑

import cv2       # 讀取圖片
import numpy as np  # 數值運算

# enrollment.py 在 enroment/ 子資料夾，但 box_pipeline.py 也在同一資料夾
# 加入 enroment/ 資料夾到 sys.path，確保 import box_pipeline 能找到
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from box_pipeline import DinoV3Extractor, VectorDB  # DINOv3 萃取器與向量資料庫


def main() -> None:
    ap = argparse.ArgumentParser(description="紙箱品牌辨識：建立 DINOv3 embedding 資料庫")
    ap.add_argument(
        "--data",
        default="enrollment/mask_enroll",
        help="mask 圖片所在資料夾（box_name.json 的 'mask' 欄位指向此資料夾的檔名）",
    )
    ap.add_argument(
        "--db",
        default="box_db.npz",
        help="輸出的向量資料庫路徑",
    )
    ap.add_argument(
        "--box-json",
        default="box_name.json",
        help="品名/品牌/mask 對應的 JSON 檔案路徑",
    )
    ap.add_argument(
        "--model-id",
        default="models/dinov3",
        help="DINOv3 模型資料夾路徑（含 config.json 與 model.safetensors）",
    )
    ap.add_argument(
        "--device",
        default=None,
        help="推論裝置：cuda / cpu（預設自動偵測）",
    )
    args = ap.parse_args()

    # ── 載入品名資料庫 JSON ──────────────────────────────────────────────────
    with open(args.box_json, encoding="utf-8") as f:
        products: list[dict] = json.load(f)["products"]  # 取出 products 陣列
    print(f"[JSON] 讀取 {args.box_json}，共 {len(products)} 個品項")

    # ── 初始化 DINOv3 模型 ────────────────────────────────────────────────────
    print(f"[模型] 載入 DINOv3：{args.model_id}")
    extractor = DinoV3Extractor(model_id=args.model_id, device=args.device)
    db = VectorDB()  # 空的向量資料庫

    # ── 逐一處理每個品項的 mask 圖片 ─────────────────────────────────────────
    total = 0  # 成功建檔的 embedding 總數

    for product in products:
        type_name  = product["name"]   # 品名，例如 "洋芋片 青檸口味"
        type_brand = product["brand"]  # 品牌，例如 "樂事"
        mask_files = product.get("mask", [])  # mask 圖片檔名列表，例如 ["5.png"]

        if not mask_files:
            # 此品項尚未準備 mask 圖，跳過
            print(f"  [略過] {type_brand} {type_name}：box_name.json 無 'mask' 欄位")
            continue

        print(f"\n[品項] {type_brand} {type_name}（{len(mask_files)} 張圖）")

        for mask_filename in mask_files:
            # 組合完整路徑：mask_enroll/1.png 等
            img_path = os.path.join(args.data, mask_filename)

            if not os.path.exists(img_path):
                print(f"  ! 找不到圖片，略過：{img_path}")
                continue

            # 讀取圖片（支援中文路徑）
            image = cv2.imdecode(
                np.fromfile(img_path, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            if image is None:
                print(f"  ! 讀取失敗，略過：{img_path}")
                continue

            # DINOv3 推論，取得 L2 正規化的 embedding 向量
            feat = extractor.embed(image)

            # 存入向量資料庫（含品名與品牌）
            db.add(feat, type_name=type_name, type_brand=type_brand)
            total += 1
            print(f"  ✓ {mask_filename}  →  embedding shape={feat.shape}")

    # ── 儲存 npz ─────────────────────────────────────────────────────────────
    db.save(args.db)

    # ── 結果摘要 ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  DINOv3 建檔完成")
    print("=" * 50)
    print(f"  輸出路徑  : {args.db}")
    print(f"  總 embedding 數 : {total}")
    print(f"  各品項數量：")
    for name, n in db.count_per_type().items():
        print(f"    {name}: {n} 組")


if __name__ == "__main__":
    main()
