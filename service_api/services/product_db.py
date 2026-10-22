"""
services/product_db.py — 品名資料庫管理與比對
=============================================
負責從 box_name.json 載入品名清單，並提供關鍵字評分比對功能。
可在不重啟 API 的情況下呼叫 reload() 動態更新品名清單。

box_name.json 格式範例：
    {
        "products": [
            {"brand": "樂事", "name": "洋芋片 青檸口味", "keywords": ["樂事", "洋芋片", "青檸"]},
            {"brand": "華元", "name": "玉黍叔 甜辣口味", "keywords": ["玉黍叔"]},
            ...
        ]
    }
"""

import json                   # 解析 JSON 檔案
from pathlib import Path      # 跨平台路徑操作
from typing import Optional   # 表示回傳值可能為 None

from service_api import config  # 取得 BOX_NAME_JSON 設定


class ProductDatabase:
    """
    品名資料庫：從 JSON 載入品項，並提供 OCR 文字比對功能。

    使用方式：
        db = ProductDatabase()        # 自動載入 box_name.json
        product = db.match(ocr_text)  # 比對並回傳最佳匹配
        db.reload()                   # 動態重載（不重啟服務）
    """

    def __init__(self) -> None:
        self._products: list[dict] = []  # 品項列表，初始為空
        self.reload()                    # 建立時立即載入

    # ── 公開介面 ─────────────────────────────────────────────────────────────

    def reload(self) -> int:
        """
        從 JSON 檔案重新載入品名資料庫。

        適用情境：新增品項到 box_name.json 後，呼叫此方法立即生效，
        不需重啟 uvicorn。

        Returns:
            重載後的品項總數
        """
        path = Path(config.BOX_NAME_JSON)      # 取得 JSON 檔案的 Path 物件
        with open(path, encoding="utf-8") as f: # 開啟並讀取 JSON（UTF-8 支援中文）
            data = json.load(f)                # 解析 JSON 字串為 Python dict
        self._products = data["products"]      # 取出 products 陣列
        return len(self._products)             # 回傳品項數（供 log 或 API 回應使用）

    def match(self, ocr_text: str) -> Optional[dict]:
        """
        從 OCR 文字中比對品名資料庫，回傳命中關鍵字最多的產品。

        比對策略（評分制）：
          - 計算每個產品有幾個 keyword 出現在 ocr_text 中
          - 取最高分的產品（分數相同時，取資料庫中靠前的品項）
          - 最高分為 0（無任何 keyword 命中）時回傳 None

        評分制優於第一個命中的原因：
          同品牌（如華元）有多個品項，評分確保取到最相符的，
          而非因資料庫排序不同而拿錯。

        Args:
            ocr_text: GLM-OCR 輸出的原始文字（可含換行、雜訊）

        Returns:
            命中的產品 dict（含 brand, name, keywords），或 None（未知品項）
        """
        best_product: Optional[dict] = None  # 目前最佳命中產品
        best_score: int = 0                  # 目前最高關鍵字命中數

        for product in self._products:
            # 計算此產品有幾個 keyword 出現在 OCR 文字中
            score = sum(
                1 for kw in product["keywords"]  # 遍歷此產品的所有關鍵字
                if kw in ocr_text                # 關鍵字存在於 OCR 文字中
            )
            if score > best_score:               # 比目前最高分更高才更新
                best_product = product
                best_score = score

        # 只有至少命中一個關鍵字才算有效比對，否則視為未知品項
        return best_product if best_score > 0 else None

    # ── 屬性 ─────────────────────────────────────────────────────────────────

    @property
    def count(self) -> int:
        """品名資料庫目前的品項總數。"""
        return len(self._products)
