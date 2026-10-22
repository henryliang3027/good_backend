"""
pipeline.py — 核心業務流程協調者
==================================
將 YOLO 偵測、GLM-OCR、品名比對、日期解析整合成一個完整流程。

API route 只需呼叫 pipeline.run()，不需知道各服務的細節。
每新增一個處理步驟，只需在這裡修改，不影響 main.py（路由層）。

流程圖：
  上傳圖片
    → (1) YOLO 偵測 → 取得所有 box 的 mask polygon
    → (2) 逐一 crop 每個 box（去背）
    → (3) GLM-OCR 辨識文字
    → (4) OCR 文字品名比對（box_name.json 關鍵字）
         ↓ 失敗
    → "未知品項"
    → (5) 日期解析（有效日期、製造日期）
    → (6) 組合回應（含選用的標注圖片）
"""

import base64                   # 將標注圖片編碼為 base64 字串
import importlib                # 動態重新載入 date_extractor 模組（熱重載用）
import sys                      # 用於檢查 date_extractor 是否已在 sys.modules
import time                     # 計算 YOLO / OCR 耗時（DEBUG 用）
from pathlib import Path        # 專案根路徑操作

import cv2                      # 解碼上傳的圖片 bytes
import numpy as np              # numpy array 操作

from service_api.services import date_extractor # 日期解析模組（保留模組引用，而非 from ... import）
                                # 用模組引用呼叫（date_extractor.extract_dates）
                                # 才能在 reload() 後自動使用新邏輯

from service_api.schemas import (   # Pydantic 回應格式
    BoxResult,
    DateInfo,
    DetectResponse,
    ProductInfo,
)
from service_api import config
from service_api.services.detector     import BoxDetector     # YOLO 偵測器
from service_api.services.ocr_client  import GlmOcrClient   # GLM-OCR 客戶端
from service_api.services.product_db  import ProductDatabase # 品名資料庫（第一層）
# from service_api.services.dino_matcher import DinoMatcher    # DINOv3 比對（第二層）
# from service_api.services.hsv_matcher  import HsvMatcher     # HSV 色彩比對（第三層）
from service_api.utils.image_utils   import (
    crop_with_mask,           # 裁切 mask 區域並去背
    draw_results,             # 在原圖上繪製偵測結果
    enhance_contrast_pillow,  # 提高圖片對比度（OCR 前處理）
    pil_to_bytes,             # 將 PIL Image 轉為 bytes
)


class BoxDetectionPipeline:
    """
    紙箱偵測完整流程的協調者（Orchestrator Pattern）。

    此類別持有所有服務的引用，由 FastAPI lifespan 在啟動時建立，
    整個應用程式生命週期只建立一次（避免重複載入 YOLO 模型）。

    使用方式：
        pipeline = BoxDetectionPipeline()          # startup 時呼叫一次
        result   = pipeline.run(img_bytes, fname)  # 每次請求呼叫
    """

    def __init__(self) -> None:
        # 初始化 YOLO 偵測器（會在此時載入模型，需幾秒）
        self.detector   = BoxDetector()
        # 初始化 GLM-OCR 客戶端（不載入模型，連外部 llama-server）
        self.ocr        = GlmOcrClient()
        # 初始化品名資料庫（第一層比對，讀取 box_name.json 關鍵字）
        self.product_db = ProductDatabase()
        # # 初始化 DINOv3 比對器（第二層比對，載入模型與 box_db.npz embedding）
        # self.dino       = DinoMatcher()
        # # 初始化 HSV 比對器（第三層比對，DINO top1~top3 信心值太接近時輔助判斷顏色）
        # self.hsv        = HsvMatcher()

    def run(
        self,
        image_bytes:   bytes,       # 上傳的圖片原始 bytes
        filename:      str,         # 原始檔名（放進回應供識別）
        include_image: bool = False, # 是否在回應中附上標注圖片
    ) -> DetectResponse:
        """
        對上傳圖片執行完整偵測流程，回傳結構化結果。

        Args:
            image_bytes:   上傳圖片的原始 bytes（JPG/PNG）
            filename:      原始檔名
            include_image: True 時回應會附上 base64 編碼的標注圖片

        Returns:
            DetectResponse（所有 box 的結果 + 可選的標注圖片）
        """
        print(f"filename={filename}")
        # ── 步驟 1：將 bytes 轉為 OpenCV BGR numpy array ────────────────────
        nparr   = np.frombuffer(image_bytes, np.uint8)    # bytes → uint8 array
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)   # 解碼為 BGR 三通道圖
        img_h, img_w = img_bgr.shape[:2]                  # 圖片高寬（mask 正規化用）

        # ── 步驟 2：YOLO 推論，取得所有 box 的 mask polygon ─────────────────
        yolo_t0   = time.time()
        detection = self.detector.detect(img_bgr)          # 傳入 BGR numpy array
        if config.DEBUG:
            print(f"[DEBUG] YOLO segmentation time: {time.time() - yolo_t0:.3f}s")

        # ── 步驟 3：逐一處理每個偵測到的 box ────────────────────────────────
        box_results: list[BoxResult] = []  # 收集每個 box 的結構化結果
        labels: list[str]            = []  # 同步收集文字標籤（繪圖用）
        ocr_total_time: float        = 0.0  # 累計所有 box 的 OCR 推論耗時（DEBUG 用）

        for idx, (polygon, conf) in enumerate(
            zip(detection.polygons, detection.confs), start=1  # box_id 從 1 開始
        ):
            # 3a. 裁切 mask 區域並去背
            #     crop_with_mask 回傳 PIL RGB（OCR 用）
            #     同時保留 BGR numpy（DINO 第二層比對用）
            cropped_pil = crop_with_mask(img_bgr, polygon)
            cropped_bgr = cv2.cvtColor(            # PIL RGB → BGR numpy
                np.array(cropped_pil), cv2.COLOR_RGB2BGR
            )
            
            # 儲存裁切後的圖片供除錯用
            crop_dir = Path("/home/ubuntu/Documents/Github/service_template/detected_crop")
            crop_dir.mkdir(parents=True, exist_ok=True)
            cropped_pil.save(crop_dir / f"{Path(filename).stem}_box_o{idx}.png")

            # 增強對比度，提升 OCR 辨識率
            enhanced_pil = enhance_contrast_pillow(cropped_pil)

            # 儲存裁切後的圖片供除錯用
            # crop_dir = Path("/home/ubuntu/Documents/Github/service_template/detected_crop")
            # crop_dir.mkdir(parents=True, exist_ok=True)
            # enhanced_pil.save(crop_dir / f"{Path(filename).stem}_box{idx}.png")

            # 3b. 將增強後的裁切圖送進 GLM-OCR，取得辨識文字
            ocr_text, ocr_elapsed = self.ocr.ocr(enhanced_pil)
            ocr_total_time += ocr_elapsed
            print(f"  OCR (box {idx}): {ocr_text.strip()}")


            # 3c. OCR 文字品名比對
            matched = self.product_db.match(ocr_text)

            # 記錄比對結果供 log 使用
            if matched:
                print(f"  比對（ocr）：{matched['brand']} {matched['name']}")
            else:
                print(f"  比對：未知品項")

            product = (                            # 轉為 Pydantic 格式（或 None）
                ProductInfo(brand=matched["brand"], name=matched["name"])
                if matched else None
            )

            if config.DEBUG:
                name = f"{matched['brand']} {matched['name']}" if matched else "未知品項"
                print(f"[DEBUG] box {idx} OCR time: {ocr_elapsed:.3f}s  品名: {name}")

            # # 3c. 第一層：OCR 文字品名比對
            # matched     = self.product_db.match(ocr_text)
            # match_layer = "ocr"                    # 記錄是哪一層比對到的

            # # 3c-2. 第二層：OCR 失敗時改用 DINOv3 視覺特徵比對
            # #       + 第三層：DINO top1~top3 信心值太接近時，改用 HSV 顏色輔助判斷
            # #       （DINO 對顏色不敏感，容易把同款包裝、不同口味的商品混淆）
            # if matched is None:
            #     if config.USE_HSV_MATCH:
            #         matched, match_layer = self._match_dino_hsv(cropped_bgr)
            #     else:
            #         matched, match_layer = self._match_dino_only(cropped_bgr)

            # # 記錄比對結果供 log 使用
            # if matched:
            #     print(f"  比對（{match_layer}）：{matched['brand']} {matched['name']}")
            # else:
            #     print(f"  比對：未知品項")
            #     match_layer = None

            # product = (                            # 轉為 Pydantic 格式（或 None）
            #     ProductInfo(brand=matched["brand"], name=matched["name"])
            #     if matched else None
            # )

            # 3d. 日期解析：同時找有效日期與製造日期
            # 注意：用 date_extractor.extract_dates 而非直接呼叫 extract_dates，
            # 這樣 importlib.reload(date_extractor) 後新邏輯會自動生效
            dates    = date_extractor.extract_dates(ocr_text)
            expiry   = _to_date_info(dates.get("expiry"))      # 有效日期（可 None）
            mfg      = _to_date_info(dates.get("manufacture")) # 製造日期（可 None）

            # 3e. mask 輪廓點正規化（除以圖片寬高，值域 0.0~1.0，與圖片尺寸無關）
            normalized_mask = [
                [round(float(x) / img_w, 4), round(float(y) / img_h, 4)]
                for x, y in polygon
            ]

            # 3f. 組合此 box 的結構化結果
            box_results.append(BoxResult(
                box_id           = idx,
                confidence       = round(float(conf), 4),  # 保留 4 位小數
                mask             = normalized_mask,
                product          = product,
                expiry_date      = expiry,
                manufacture_date = mfg,
            ))

            # 3g. 建立文字標籤（box_id + 品名 + 信心值 + 日期，多行顯示）
            if matched:
                name_line = f"#{idx} {matched['brand']} {matched['name']} ({conf:.2f})"
            else:
                name_line = f"#{idx} 未知品項 ({conf:.2f})"
            date_lines = []
            if expiry:   # 有效日期有解析到才顯示
                date_lines.append(f"有效日期：{expiry.year}/{expiry.month}/{expiry.day}")
            if mfg:      # 製造日期有解析到才顯示
                date_lines.append(f"製造日期：{mfg.year}/{mfg.month}/{mfg.day}")
            labels.append("\n".join([name_line] + date_lines))  # 用換行合併多行

            # # 3f. 建立文字標籤（品名 + 信心值 + 比對方式 + 日期，多行顯示）
            # if matched and match_layer == "ocr":
            #     name_line = f"{matched['brand']} {matched['name']}"
            # elif matched and match_layer == "dino":
            #     name_line = f"{matched['brand']} {matched['name']}"
            # elif matched and match_layer == "hsv":
            #     name_line = f"{matched['brand']} {matched['name']}"
            # else:
            #     name_line = f"未知品項"
            # date_lines = []
            # if expiry:   # 有效日期有解析到才顯示
            #     date_lines.append(f"有效日期：{expiry.year}/{expiry.month}/{expiry.day}")
            # if mfg:      # 製造日期有解析到才顯示
            #     date_lines.append(f"製造日期：{mfg.year}/{mfg.month}/{mfg.day}")
            # labels.append("\n".join([name_line] + date_lines))  # 用換行合併多行

        if config.DEBUG:
            print(f"[DEBUG] OCR total inference time ({detection.total} boxes): {ocr_total_time:.3f}s")

        # ── 步驟 4：視覺化（選用）─────────────────────────────────────────────
        annotated_b64: str | None = None
        if include_image and detection.total > 0:  # 有偵測到 box 才繪製
            # 在原圖上疊加所有 box 的 mask 與標籤
            result_img = draw_results(img_bgr, detection.polygons, labels)
            # 轉為 JPEG bytes → base64 字串，放入 JSON 回應
            annotated_b64 = base64.b64encode(pil_to_bytes(result_img)).decode("utf-8")

        # ── 步驟 5：組合並回傳完整回應 ───────────────────────────────────────
        return DetectResponse(
            filename                = filename,
            total_boxes             = detection.total,
            boxes                   = box_results,
            annotated_image_base64  = annotated_b64,
        )

    def _match_dino_hsv(self, cropped_bgr: np.ndarray) -> tuple[dict | None, str | None]:
        """
        第二層（DINOv3）+ 第三層（HSV）品名比對。

        邏輯：
          1. 取 DINO top1~top3（依信心值排序）。
          2. top1 分數 < DINO_MATCH_THRESHOLD → DINO 連最高分都不夠確定，
             不用建待比較清單，直接視為未知品項。
          3. 建立「待比較清單」：
             - top1、top2 信心值差距 < DINO_TOPK_GAP_THRESHOLD → top2 加入清單
             - 承上，若 top2 也加入了，且 top2、top3 差距 < 門檻 → top3 也加入
             - 待比較清單最多 3 筆
          4. 若清單只有 top1（DINO 已經很確定）→ 直接採用 top1。
          5. 若清單有多筆（DINO 分不清楚，通常是顏色相近的品項撞名）
             → 只在這幾個候選品項之間比對 HSV 顏色相似度：
                - 最高分 >= HSV_MATCH_THRESHOLD → 回傳該品項（match_layer="hsv"）
                - 否則 → 未知品項

        Returns:
            (matched, match_layer)：matched 為 {"brand", "name"} 或 None
        """
        top3 = self.dino.match_topk(cropped_bgr, k=3)  # 由高到低排序
        if not top3:
            return None, None

        # top1 分數未達門檻：DINO 連最高分都不夠有信心，不用再建待比較清單，直接視為未知品項
        if top3[0]["score"] < config.DINO_MATCH_THRESHOLD:
            return None, None

        # 建立待比較清單（最多 3 筆）
        compare_list = [top3[0]]
        if len(top3) >= 2 and abs(top3[0]["score"] - top3[1]["score"]) < config.DINO_TOPK_GAP_THRESHOLD:
            compare_list.append(top3[1])
            if len(top3) >= 3 and abs(top3[1]["score"] - top3[2]["score"]) < config.DINO_TOPK_GAP_THRESHOLD:
                compare_list.append(top3[2])

        print(compare_list)  # log 用，觀察 top1~top3 信心值差距

        # 只有 top1：DINO 已經足夠確定（top1 已通過門檻），直接採用
        if len(compare_list) == 1:
            top1 = compare_list[0]
            return {"brand": top1["brand"], "name": top1["name"]}, "dino"

        # 候選品項太接近：改用 HSV 顏色比對，只在這幾個候選項之間找最佳匹配
        hsv_result = self.hsv.match(cropped_bgr, candidates=compare_list)
        if hsv_result is None:
            return None, None
        return {"brand": hsv_result["brand"], "name": hsv_result["name"]}, "hsv"

    def _match_dino_only(self, cropped_bgr: np.ndarray) -> tuple[dict | None, str | None]:
        """
        備用版本：只取 DINOv3 top1，不使用 HSV 輔助比對。

        邏輯：
          1. 取 DINO top1。
          2. top1 分數 >= DINO_MATCH_THRESHOLD → 直接採用 top1。
          3. 否則 → 未知品項。

        Returns:
            (matched, match_layer)：matched 為 {"brand", "name"} 或 None
        """
        top1_list = self.dino.match_topk(cropped_bgr, k=1)
        if not top1_list:
            return None, None

        top1 = top1_list[0]
        if top1["score"] < config.DINO_MATCH_THRESHOLD:
            return None, None

        return {"brand": top1["brand"], "name": top1["name"]}, "dino"

    def reload_date_extractor(self) -> None:
        """
        熱重載 date_extractor 模組（不重啟服務）。

        使用情境：修改 date_extractor.py（如新增日期格式 regex）後，
        呼叫 POST /admin/reload-date-extractor 讓新邏輯立即生效。

        原理：importlib.reload 會重新執行 date_extractor.py，
        之後透過 date_extractor.extract_dates 呼叫的都是新邏輯。
        """
        importlib.reload(date_extractor)  # 重新執行 date_extractor.py


# ── 內部輔助函式 ──────────────────────────────────────────────────────────────

def _to_date_info(d: dict | None) -> DateInfo | None:
    """
    將 date_extractor.extract_dates 回傳的 dict 轉為 Pydantic DateInfo。
    d 為 None 時直接穿透（回傳 None），避免呼叫端重複判斷。
    """
    if d is None:
        return None                                             # None 直接穿透
    return DateInfo(year=d["year"], month=d["month"], day=d["day"])  # 轉為 Pydantic 物件
