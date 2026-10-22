# POST /api/v1/detect — 前端串接指南

給前端工程師的 `/api/v1/detect` 回傳格式整理，含 TypeScript 型別、範例與各種邊界情況。

---

## 請求

```
POST /api/v1/detect?include_image=false
Content-Type: multipart/form-data

file: <圖片檔案 JPG/PNG>
```

| 參數 | 位置 | 型別 | 必填 | 說明 |
|------|------|------|------|------|
| `file` | form-data | binary | ✅ | 要偵測的圖片 |
| `include_image` | query | boolean | ❌（預設 `false`） | `true` 時回應會附上標注後的 base64 圖片 |

---

## TypeScript 型別定義

```ts
interface DateInfo {
  year: string;   // 4 碼西元年，例如 "2027"
  month: string;  // 2 碼補零，例如 "01"
  day: string;    // 2 碼補零，例如 "13"
}

interface ProductInfo {
  brand: string;  // 品牌名稱，例如 "樂事"
  name: string;   // 品名，例如 "洋芋片 青檸口味"
}

interface BoxResult {
  box_id: number;                        // 從 1 開始
  confidence: number;                    // YOLO 信心值 0~1，保留 4 位小數
  product: ProductInfo | null;           // 未比對到品項時為 null → 前端顯示「未知品項」
  expiry_date: DateInfo | null;          // 有效日期，解析失敗為 null
  manufacture_date: DateInfo | null;     // 製造日期，解析失敗為 null
}

interface DetectResponse {
  filename: string;
  total_boxes: number;                   // = boxes.length
  boxes: BoxResult[];
  annotated_image_base64: string | null; // include_image=true 時才有值；為 JPEG 的 base64（不含 data URI 前綴）
}
```

---

## 範例：正常情境（多個 box，部分未知品項）

```json
{
  "filename": "test2.jpg",
  "total_boxes": 2,
  "boxes": [
    {
      "box_id": 1,
      "confidence": 0.9712,
      "product": { "brand": "樂事", "name": "洋芋片 青檸口味" },
      "expiry_date": { "year": "2027", "month": "01", "day": "13" },
      "manufacture_date": null
    },
    {
      "box_id": 2,
      "confidence": 0.8531,
      "product": null,
      "expiry_date": null,
      "manufacture_date": null
    }
  ],
  "annotated_image_base64": null
}
```

## 範例：`include_image=true`

`annotated_image_base64` 為純 base64 字串（不含 `data:image/jpeg;base64,` 前綴），前端顯示時需自行補上：

```ts
const src = `data:image/jpeg;base64,${resp.annotated_image_base64}`;
```

## 範例：圖中沒有偵測到任何紙箱

```json
{
  "filename": "empty.jpg",
  "total_boxes": 0,
  "boxes": [],
  "annotated_image_base64": null
}
```

`include_image=true` 但沒偵測到 box 時，`annotated_image_base64` 仍為 `null`（沒有東西可標注）。

---

## 錯誤回應

| HTTP 狀態 | 情境 | 回應 body |
|-----------|------|-----------|
| `400` | 上傳檔案不是有效的圖片（無法被 OpenCV 解碼） | `{ "detail": "無效的圖片格式，請上傳 JPG 或 PNG 檔案。" }` |
| `422` | 缺少 `file` 欄位或格式不符（FastAPI 自動驗證） | FastAPI 標準驗證錯誤格式 |

前端建議：對非 2xx 一律讀取 `detail` 欄位顯示錯誤訊息。

---

## 前端顯示建議

- `product === null` → 顯示「未知品項」，可考慮加上人工標記/回報按鈕
- `expiry_date === null` 或 `manufacture_date === null` → 顯示「無法辨識」而非留白，避免使用者誤以為沒有日期
- `confidence` 可用於前端做信心值過濾或標示（例如 < 0.7 顯示警示色）
- `box_id` 僅在單次請求內唯一，不可跨請求當作永久識別碼
