"""
config.py — 所有設定集中管理
=============================
修改設定時只需改這一個檔案。
也可透過環境變數覆蓋（適合 Docker / 不同環境部署）。

用法：
    from service_api import config
    print(config.YOLO_MODEL_PATH)
"""

import os                    # 讀取環境變數
from pathlib import Path     # 跨平台路徑操作

# ── 專案路徑 ──────────────────────────────────────────────────────────────────
# config.py 在 service_api/ 裡，上一層（parent.parent）才是專案根目錄
ROOT_DIR: Path = Path(__file__).parent.parent

# ── YOLO 設定 ─────────────────────────────────────────────────────────────────
# YOLO 模型權重路徑，可用環境變數 YOLO_MODEL_PATH 覆蓋
YOLO_MODEL_PATH: str = os.getenv(
    "YOLO_MODEL_PATH",
    str(ROOT_DIR / "models/yolo_model/box_segmentation/best_26x_seg_20260702.pt"),
)
# 低於此信心值的偵測結果會被過濾（0.0~1.0，越高越嚴格）
YOLO_CONF_THRESHOLD: float = float(os.getenv("YOLO_CONF", "0.80"))
# 推論時將圖片縮放至此邊長（須與訓練時一致）
YOLO_IMG_SIZE: int = int(os.getenv("YOLO_IMGSZ", "640"))

# ── GLM-OCR（llama-server）設定 ───────────────────────────────────────────────
# llama-server 的 OpenAI 相容 API base URL
LLAMA_CPP_URL: str = os.getenv("LLAMA_CPP_URL", "http://localhost:8000/v1")
# 傳給 GLM-OCR 的 system prompt
OCR_PROMPT: str = "Text Recognition:"
# 最大輸出 token 數（文字通常短，2048 已足夠）
OCR_MAX_TOKENS: int = 2048
# temperature=0 讓 OCR 輸出固定，不隨機抽樣
OCR_TEMPERATURE: float = 0.0

# ── 品名資料庫 ────────────────────────────────────────────────────────────────
# box_name.json 的路徑，可用環境變數 BOX_NAME_JSON 覆蓋
BOX_NAME_JSON: str = os.getenv(
    "BOX_NAME_JSON",
    str(ROOT_DIR / "box_name.json"),
)

# ── 視覺化設定 ────────────────────────────────────────────────────────────────
# 中文字型路徑（Ubuntu Noto Sans CJK）
FONT_PATH: str = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
# 標籤文字大小依原圖寬度動態換算（見 image_utils._dynamic_font_size），不再用固定值
# mask 填色的透明度（0.0=完全透明，1.0=完全不透明）
MASK_ALPHA: float = 0.4
# 文字標籤外框線寬（像素）
LABEL_BORDER_WIDTH: int = 2

# # ── DINOv3 第二層辨識設定 ────────────────────────────────────────────────────
# # DINOv3 模型資料夾路徑（enrollment.py 使用的同一個模型）
# DINO_MODEL_PATH: str = os.getenv(
#     "DINO_MODEL_PATH",
#     str(ROOT_DIR / "models/dinov3"),
# )
# # enrollment.py 產生的 embedding 資料庫路徑
# BOX_DB_NPZ: str = os.getenv(
#     "BOX_DB_NPZ",
#     str(ROOT_DIR / "box_db.npz"),
# )
# # 餘弦相似度門檻：低於此值視為未知品項（0.0~1.0，越高越嚴格）
# DINO_MATCH_THRESHOLD: float = float(os.getenv("DINO_MATCH_THRESHOLD", "0.7"))

# # ── HSV 第三層辨識設定（DINO top1/top2/top3 信心值太接近時的顏色輔助判斷）──────
# # enrollment_hsv.py 產生的 HSV 直方圖資料庫路徑
# BOX_HSV_DB_NPZ: str = os.getenv(
#     "BOX_HSV_DB_NPZ",
#     str(ROOT_DIR / "box_hsv_db.npz"),
# )
# # top1/top2、top2/top3 信心值差距門檻：小於此值視為「太接近，需要 HSV 輔助判斷」
# DINO_TOPK_GAP_THRESHOLD: float = float(os.getenv("DINO_TOPK_GAP_THRESHOLD", "0.05"))
# # HSV 相似度門檻：候選項中最高分低於此值視為未知品項
# HSV_MATCH_THRESHOLD: float = float(os.getenv("HSV_MATCH_THRESHOLD", "0.7"))

# # 是否啟用 HSV 第三層輔助判斷。False 時只取 DINO top1（備用版本，不比對 HSV）
# USE_HSV_MATCH: bool = os.getenv("USE_HSV_MATCH", "true").lower() in ("1", "true", "yes")

# ── 未知品項 mask 儲存路徑（供人工複核 / 之後補建 enrollment 用）──────────────
SEGMENTATION_UNKNOWN_DIR: str = os.getenv(
    "SEGMENTATION_UNKNOWN_DIR",
    str(ROOT_DIR / "enrollment/segmentation_unknown"),
)

# ── FastAPI 設定 ──────────────────────────────────────────────────────────────
# API 路徑前綴（所有偵測端點都掛在這個前綴下）
API_PREFIX: str = "/api/v1"

# ── 除錯設定 ──────────────────────────────────────────────────────────────────
# True 時，/detect/image 會印出圖片接收時間（hh:mm:ss）、
# YOLO 偵測耗時、以及所有 box 加總的 OCR 推論耗時
DEBUG: bool = os.getenv("DEBUG", "true").lower() in ("1", "true", "yes")
