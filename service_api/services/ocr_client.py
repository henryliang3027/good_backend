"""
services/ocr_client.py — GLM-OCR 呼叫封裝
==========================================
透過 OpenAI-compatible API 呼叫外部 llama-server 執行 GLM-OCR。
所有 OCR 網路呼叫都集中在這裡，未來換成其他 OCR 服務只需修改此檔案。

前置條件：
  llama-server 已獨立啟動，監聽 config.LLAMA_CPP_URL（預設 http://localhost:8000/v1）
"""

import base64   # 將圖片編碼為 base64 字串，才能放進 JSON 請求
import io       # 記憶體緩衝區（不需寫到磁碟）
import time     # 計算推論耗時

from openai import OpenAI       # OpenAI Python SDK（相容 llama-server）
from PIL import Image           # PIL Image 格式

from service_api import config  # 取得 LLAMA_CPP_URL、OCR_PROMPT 等設定


class GlmOcrClient:
    """
    封裝對 GLM-OCR（llama-server）的 API 呼叫。

    使用方式：
        client = GlmOcrClient()
        text, elapsed = client.ocr(pil_image)
    """

    def __init__(self) -> None:
        # 建立 OpenAI client，指向本地 llama-server
        # api_key 填任意字串即可，llama-server 不驗證金鑰
        self._client = OpenAI(
            base_url=config.LLAMA_CPP_URL,
            api_key="no-key-needed",
        )

    def ocr(self, image: Image.Image) -> tuple[str, float]:
        """
        對一張 PIL Image 執行 OCR，回傳辨識文字與耗時。

        Args:
            image: PIL RGB Image（任意尺寸）

        Returns:
            (ocr_text, elapsed_seconds)
              ocr_text: GLM-OCR 輸出的原始文字
              elapsed_seconds: 從發送請求到收到完整回應的秒數
        """
        b64 = self._encode_image(image)  # 將圖片轉為 base64 字串
        t0  = time.time()                # 記錄請求開始時間

        # 使用 vision message 格式呼叫（文字提示 + 圖片 data URL）
        resp = self._client.chat.completions.create(
            model="glm-ocr",              # llama-server 載入的模型名稱
            messages=[{
                "role": "user",
                "content": [
                    # 第一個 content block：給 GLM-OCR 的指令
                    {
                        "type": "text",
                        "text": config.OCR_PROMPT,
                    },
                    # 第二個 content block：圖片（data URL 格式）
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}"
                        },
                    },
                ],
            }],
            temperature=config.OCR_TEMPERATURE,  # 0 = 固定輸出，不隨機
            max_tokens=config.OCR_MAX_TOKENS,    # 限制回應長度（避免無限輸出）
        )

        elapsed = time.time() - t0                    # 計算實際耗時
        text    = resp.choices[0].message.content     # 取第一個回應的文字內容
        return text, elapsed

    def is_reachable(self) -> bool:
        """
        確認 llama-server 是否可連線（用於健康檢查端點）。

        Returns:
            True 表示可連線，False 表示連線失敗（server 未啟動或 URL 設定錯誤）
        """
        try:
            self._client.models.list()  # 呼叫 /v1/models 端點來測試連線
            return True
        except Exception:
            return False                # 任何例外（連線失敗、timeout 等）都視為不可用

    @staticmethod
    def _encode_image(image: Image.Image, fmt: str = "JPEG") -> str:
        """
        將 PIL Image 轉為 base64 字串（用於 API 請求的 data URL）。

        Args:
            image: PIL Image（任意模式）
            fmt: 編碼格式，預設 JPEG（比 PNG 小、推論結果相近）

        Returns:
            base64 字串（不含 data:image/... 前綴）
        """
        buf = io.BytesIO()                          # 建立記憶體緩衝區（不寫到磁碟）
        image.save(buf, format=fmt)                 # 將圖片存入緩衝區
        raw = buf.getvalue()                        # 取出原始 bytes
        return base64.b64encode(raw).decode("utf-8")  # 轉 base64 後解碼成字串
