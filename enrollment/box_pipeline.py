# -*- coding: utf-8 -*-
"""
box_pipeline.py — DINOv3 特徵萃取器 + 向量資料庫
=================================================
提供兩個類別：
  DinoV3Extractor  — 載入 DINOv3 模型，將圖片轉為 L2 正規化的 embedding 向量
  VectorDB         — 儲存 embedding + type_name + type_brand，可存成 npz 並做餘弦相似度搜尋
"""

import numpy as np                       # 矩陣運算
import cv2                               # BGR 圖片讀取與色彩轉換
import torch                             # GPU 推論
from PIL import Image                    # transformers processor 需要 PIL 格式
from collections import Counter          # count_per_type 用
from transformers import AutoModel, AutoImageProcessor  # 載入 HuggingFace 模型


class DinoV3Extractor:
    """
    DINOv3 特徵萃取器。

    使用方式：
        ext = DinoV3Extractor(model_id="models/dinov3")
        feat = ext.embed(bgr_image)   # 回傳 (768,) float32 L2-normalized 向量
    """

    def __init__(self, model_id: str = "./dinov3_model", device: str | None = None) -> None:
        # 自動選擇 GPU（cuda / rocm）或 CPU
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # 載入 image processor（resize、normalize）
        self.processor = AutoImageProcessor.from_pretrained(
            model_id, trust_remote_code=True
        )
        # 載入 DINOv3 模型權重（.safetensors）
        self.model = AutoModel.from_pretrained(
            model_id, trust_remote_code=True
        )
        self.model.to(self.device).eval()  # 切換推論模式，不計算梯度

    def embed(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        將 BGR numpy array 轉為 L2 正規化的 DINOv3 embedding。

        Args:
            image_bgr: OpenCV 讀取的 BGR 圖片，shape=(H, W, 3)

        Returns:
            float32 numpy array，shape=(768,)，已做 L2 正規化
        """
        # OpenCV BGR → RGB → PIL（transformers processor 需要 RGB PIL）
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        # 前處理：resize 224x224、正規化（ImageNet mean/std）
        inputs = self.processor(images=pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}  # 搬到 GPU

        with torch.no_grad():                          # 推論時不計算梯度（省記憶體）
            outputs = self.model(**inputs)

        # 取 CLS token（位置 0）作為整張圖的 embedding
        feat = outputs.last_hidden_state[:, 0, :].squeeze().cpu().float().numpy()

        # L2 正規化：讓餘弦相似度等於向量內積，搜尋更快
        norm = np.linalg.norm(feat)
        return feat / norm if norm > 0 else feat


class VectorDB:
    """
    輕量向量資料庫：存 embedding + 品名 + 品牌，支援儲存/載入 npz 與餘弦相似度搜尋。

    npz 結構：
      feats       — shape (N, D)，float32，L2 正規化
      type_names  — shape (N,)，str，對應 box_name.json 的 "name"
      type_brands — shape (N,)，str，對應 box_name.json 的 "brand"

    使用方式：
        db = VectorDB()
        db.add(feat, "洋芋片 青檸口味", "樂事")
        db.save("box_db.npz")

        db2 = VectorDB.load("box_db.npz")
        results = db2.search(query_feat, top_k=3)
    """

    def __init__(self) -> None:
        self._feats: list[np.ndarray] = []   # embedding 列表
        self._type_names: list[str]   = []   # 品名列表
        self._type_brands: list[str]  = []   # 品牌列表

    def add(self, feat: np.ndarray, type_name: str, type_brand: str = "") -> None:
        """新增一筆 embedding 記錄。type_brand 預設為空（向下相容 enrollment_hsv.py）。"""
        self._feats.append(feat)
        self._type_names.append(type_name)
        self._type_brands.append(type_brand)

    def save(self, path: str) -> None:
        """將所有 embedding 存成 npz 檔案。"""
        np.savez(
            path,
            feats       = np.stack(self._feats),          # (N, D) float32
            type_names  = np.array(self._type_names),     # (N,) str
            type_brands = np.array(self._type_brands),    # (N,) str
        )

    @classmethod
    def load(cls, path: str) -> "VectorDB":
        """從 npz 檔案載入，回傳 VectorDB 實例。"""
        data = np.load(path, allow_pickle=True)
        db = cls()
        db._feats       = list(data["feats"])
        db._type_names  = list(data["type_names"].astype(str))
        db._type_brands = list(data["type_brands"].astype(str))
        return db

    def search(self, query: np.ndarray, top_k: int = 1) -> list[tuple]:
        """
        餘弦相似度搜尋（query 需已 L2 正規化）。

        Returns:
            list of (type_name, type_brand, score)，依相似度由高到低排列
        """
        if not self._feats:
            return []
        matrix = np.stack(self._feats)          # (N, D)
        scores = matrix @ query                  # 餘弦相似度 = 內積（L2 normalized）
        idx    = np.argsort(scores)[::-1][:top_k]
        return [
            (self._type_names[i], self._type_brands[i], float(scores[i]))
            for i in idx
        ]

    def count_per_type(self) -> dict[str, int]:
        """回傳每個 type_name 有幾筆 embedding。"""
        return dict(Counter(self._type_names))
