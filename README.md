# Box Detection API

紙箱偵測 + OCR + 品名比對 + 日期解析 的整合 API。

**完整流程：** 上傳圖片 → YOLO 偵測紙箱 mask → GLM-OCR 辨識文字 → 品名比對 → 日期解析

---

## 目錄結構

```
service_template/
├── service_api/               ← API 主體
│   ├── main.py                ← FastAPI 入口、路由定義
│   ├── config.py              ← 所有設定常數（路徑、閾值等）
│   ├── schemas.py             ← API 請求/回應的 Pydantic 格式
│   ├── pipeline.py            ← 核心流程協調者（偵測→OCR→比對→解析）
│   ├── services/
│   │   ├── detector.py        ← YOLO 封裝
│   │   ├── ocr_client.py      ← GLM-OCR 封裝
│   │   └── product_db.py      ← 品名資料庫管理
│   └── utils/
│       └── image_utils.py     ← 影像處理工具（裁切、繪圖）
├── date_extractor.py          ← 日期解析模組
└── box_name.json              ← 品名資料庫
```

---

## 前置條件

### 1. 啟動 llama-server（GLM-OCR）

API 服務本身不載入 GLM-OCR 模型，需要先在外部啟動 llama-server：

```bash
./llama-b1287/llama-server \
    -m models/glm-ocr/GLM-OCR-f16.gguf \
    --mmproj models/glm-ocr/mmproj-GLM-OCR-Q8_0.gguf \
    --host 0.0.0.0 \
    --port 8000 \
    -ngl 99
```

確認 server 已就緒：

```bash
curl http://localhost:8000/v1/models
```

### 2. 確認 YOLO 模型存在

```
models/yolo_model/box_segmentation/best_26x_seg_20260702.pt
```

---

## 啟動 API 服務

在**專案根目錄**執行：

```bash
# 一般啟動
uvicorn service_api.main:app --host 0.0.0.0 --port 8080

# 開發模式（修改程式碼後自動重啟）
uvicorn service_api.main:app --host 0.0.0.0 --port 8080 --reload
```

啟動成功會看到：

```
🚀 [startup] 初始化 BoxDetectionPipeline，正在載入 YOLO 模型...
✅ [startup] 完成！  品名資料庫：10 筆
INFO:     Uvicorn running on http://0.0.0.0:8080
```

---

## API 端點

啟動後開啟 **http://localhost:8080/docs** 可看完整的 Swagger UI。

| 方法 | 路徑 | 說明 |
|------|------|------|
| GET  | `/health` | 健康檢查（YOLO / OCR server / 品名DB 狀態） |
| POST | `/api/v1/detect` | 偵測圖片，回傳 JSON 結果 |
| POST | `/api/v1/detect/image` | 偵測圖片，直接回傳標注後的 JPEG 圖片 |
| POST | `/admin/reload-product-db` | 熱重載 box_name.json（不重啟服務） |
| POST | `/admin/reload-date-extractor` | 熱重載 date_extractor.py（不重啟服務） |

### POST /api/v1/detect

```bash
curl -X POST http://localhost:8080/api/v1/detect \
     -F "file=@test_images/test2.jpg"
```

加上 `?include_image=true` 可在回應中附上 base64 標注圖：

```bash
curl -X POST "http://localhost:8080/api/v1/detect?include_image=true" \
     -F "file=@test_images/test2.jpg"
```

回應格式：

```json
{
  "filename": "test2.jpg",
  "total_boxes": 3,
  "boxes": [
    {
      "box_id": 1,
      "confidence": 0.9712,
      "product": { "brand": "樂事", "name": "洋芋片 青檸口味" },
      "expiry_date": { "year": "2027", "month": "01", "day": "13" },
      "manufacture_date": null
    }
  ],
  "annotated_image_base64": null
}
```

### POST /api/v1/detect/image

直接回傳標注後的 JPEG 圖片，適合用在 `<img>` 標籤或存檔：

```bash
curl -X POST http://localhost:8080/api/v1/detect/image \
     -F "file=@test_images/test2.jpg" \
     --output result.jpg
```

---

## 環境變數設定

所有設定都有預設值，可透過環境變數覆蓋（不需修改程式碼）：

| 環境變數 | 預設值 | 說明 |
|----------|--------|------|
| `YOLO_MODEL_PATH` | `models/yolo_model/.../best_26x_seg_20260702.pt` | YOLO 模型路徑 |
| `YOLO_CONF` | `0.65` | YOLO 信心值過濾閾值 |
| `LLAMA_CPP_URL` | `http://localhost:8000/v1` | llama-server 的 API base URL |
| `BOX_NAME_JSON` | `box_name.json` | 品名資料庫路徑 |

範例（llama-server 在另一台機器）：

```bash
LLAMA_CPP_URL=http://192.168.1.100:8000/v1 \
uvicorn service_api.main:app --host 0.0.0.0 --port 8080
```

---

## 不重啟服務更新設定

### 更新品名資料庫

編輯 `box_name.json` 後：

```bash
curl -X POST http://localhost:8080/admin/reload-product-db
```

### 更新日期解析邏輯

編輯 `date_extractor.py` 後：

```bash
curl -X POST http://localhost:8080/admin/reload-date-extractor
```

---

## 品名資料庫格式（box_name.json）

```json
{
  "products": [
    {
      "brand": "樂事",
      "name": "洋芋片 青檸口味",
      "keywords": ["樂事", "洋芋片", "青檸"]
    }
  ]
}
```

比對邏輯：計算 OCR 文字中命中的 `keywords` 數量，取最高分的產品。

---

## 新手上手順序

1. `service_api/config.py` — 所有可調參數在這裡，改設定從這裡開始
2. `service_api/schemas.py` — 看 API 的輸入輸出格式定義
3. `service_api/pipeline.py` — 看完整的處理流程（5 個步驟）
4. `service_api/main.py` — 看 HTTP 路由是怎麼定義的
