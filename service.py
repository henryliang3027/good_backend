import os
import base64
import io
import time
from collections import Counter
from contextlib import asynccontextmanager
import signal
import ollama
from PIL import Image, ImageDraw, ImageFont
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from ultralytics import YOLO
from openai import OpenAI

import subprocess
from nicegui import ui
from utils.date_validator import DateValidator
from dependencies import set_collection

from datetime import datetime

_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
debug_font = ImageFont.truetype(_FONT_PATH, size=18)


# ========== Model & DB Config ==========
YOLO_MODEL_PATH = "14_bottles_yolo/bottle_detector/best105.pt"
CAP_YOLO_MODEL_PATH = "caps_yolo/cap_detector/best596.pt"
CAP_CONF_THRESHOLD = 0.5
BOTTLE_CONF_THRESHOLD = 0.65

LABEL_NAMES = {
    0:  "冷山茶王",
    1:  "茶裏王台式綠茶",
    2:  "茶裏王日式無糖綠茶",
    3:  "茶裏王白毫烏龍",
    4:  "茶裏王半熟金萱",
    5:  "原萃台灣青茶",
    6:  "原萃烏龍茶",
    7:  "原萃鐵觀音",
    8:  "無加糖LP33機能優酪乳",
    9:  "御茶園特上檸檬茶",
    10: "每朝健康双纖綠茶",
    11: "每朝健康熟藏紅茶",
    12: "愛之味油切分解茶四季春風味",
    13: "濃韻無糖烏龍茶",
}

CLASS_COLORS = [
    (  0, 204, 255),  #  0 冷山茶王             - 黃
    ( 57, 219,  83),  #  1 茶裏王台式綠茶        - 綠
    ( 34, 139,  34),  #  2 茶裏王日式無糖綠茶    - 深綠
    (180, 180, 180),  #  3 茶裏王白毫烏龍        - 灰
    (  0, 165, 255),  #  4 茶裏王半熟金萱        - 橙
    ( 94, 212,  94),  #  5 原萃台灣青茶          - 青綠
    (139,  69,  19),  #  6 原萃烏龍茶            - 棕
    (148,   0, 211),  #  7 原萃鐵觀音            - 紫
    (255, 255,   0),  #  8 無加糖LP33機能優酪乳  - 青
    (255, 128,   0),  #  9 御茶園特上檸檬茶      - 藍
    ( 50, 205,  50),  # 10 每朝健康双纖綠茶      - 草綠
    (  0,   0, 200),  # 11 每朝健康熟藏紅茶      - 紅
    (255,  20, 147),  # 12 愛之味油切分解茶      - 粉
    ( 20,  20, 220),  # 13 濃韻無糖烏龍茶        - 深紅
]


class Base64ImageRequest(BaseModel):
    image_base64: str
    question: str = "請統計商品"

# ========== Global Objects ==========
yolo_model = None
cap_yolo_model = None
chroma_client = None
collection = None


SYSTEM_PROMPT_TEMPLATE = """你是一位專業的超商貨架分析員。請根據以下掃描結果清單回答用戶問題。

【掃描結果清單】
{scan_list}

【輸出格式——絕對遵守】
每筆商品資訊必須嚴格使用下列格式，注意「有」字與空格，禁止使用冒號（: 或 ：）：
  [商品名稱] 有 [數量] 瓶

正確：茶裏王台式綠茶 有 2 瓶
禁止：茶裏王台式綠茶: 2 瓶
禁止：茶裏王台式綠茶：2 瓶
禁止：茶裏王台式綠茶 2 瓶

【回答規則】
1. 「統計商品」——列出清單中所有商品，每行一個，格式如上。不加任何標題或列點符號。

2. 「有幾瓶 [品牌]」——品牌前綴查詢（如：茶裏王、原萃、每朝）：
   - 從掃描清單中找出所有名稱「以該品牌為開頭」的商品。
   - 每行一個，格式如上。必須列出所有符合的商品，不可遺漏任何一項。
   - 若清單中完全沒有符合的商品，僅回答：沒有找到您指定的商品

3. 「有幾瓶 [完整商品名稱]」——完整名稱查詢：
   - 若清單中有該商品，回答：[商品名稱] 有 [數量] 瓶
   - 若清單中沒有該商品，回答：沒有找到您指定的商品

4. 禁止輸出任何額外說明、前言或結尾客套話。
5. 必須使用繁體中文。
6. 若遇到語音辨識諧音詞，自動對應到清單中最相似的商品名稱。
"""

client = OpenAI(
    base_url="http://127.0.0.1:8881/v1",
    api_key="no-key-needed",  # 本地通常不驗證，填任意字串即可
)


# llama-server 進程
llama_process = None

LLAMA_SERVER_CMD = [
    "./llama.cpp/build/bin/llama-server",
    "-m",
    "ministral/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf",
    "--mmproj",
    "ministral/mmproj-F16.gguf",
    "--host",
    "0.0.0.0",
    "--port",
    "8881",
    "--ctx-size",
    "4096",
    "-ngl",
    "-1",
]


def start_llama_server():
    global llama_process
    print("Starting llama-server...")
    llama_process = subprocess.Popen(
        LLAMA_SERVER_CMD,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # 建立獨立 process group
    )
    # 等待 llama-server 啟動
    time.sleep(5)
    print(f"llama-server started with PID: {llama_process.pid}")


def stop_llama_server():
    global llama_process
    if llama_process:
        print(f"Stopping llama-server (PID: {llama_process.pid})...")
        try:
            os.killpg(os.getpgid(llama_process.pid), signal.SIGTERM)
            llama_process.wait(timeout=10)
        except (subprocess.TimeoutExpired, ProcessLookupError):
            try:
                os.killpg(os.getpgid(llama_process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        print("llama-server stopped.")
        llama_process = None


def _signal_handler(sig, frame):
    stop_llama_server()
    raise SystemExit(0)

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global yolo_model, cap_yolo_model, chroma_client, collection
    print("🚀 正在啟動系統並載入模型...")

    # 1. 載入自訓練 YOLO 偵測模型
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print(f"✅ YOLO 模型載入完成: {YOLO_MODEL_PATH}")

    cap_yolo_model = YOLO(CAP_YOLO_MODEL_PATH)
    print(f"✅ Cap YOLO 模型載入完成: {CAP_YOLO_MODEL_PATH}")

    start_llama_server()
    yield
    stop_llama_server()


app = FastAPI(
    title="Good API v1",
    description="test",
    version="1.0.0",
    lifespan=lifespan,
)



class Base64ImageRequest(BaseModel):
    image_base64: str
    question: str = "請統計圖中的商品"


# ========== Helper Functions ==========

DEBUG_DIR = "detected_bottle"


def bbox_iou(a: tuple, b: tuple) -> float:
    """計算兩個 bbox (x1,y1,x2,y2) 的 IoU。"""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def group_overlapping_bboxes(bboxes: list[tuple]) -> list[list[int]]:
    """將互相有 overlap 的 bbox 以 Union-Find 歸為同一群，回傳各群的 index 列表。"""
    n = len(bboxes)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if bbox_iou(bboxes[i], bboxes[j]) > 0:
                parent[find(i)] = find(j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def detect_and_label(pil_image: Image.Image) -> tuple[list[str], list[tuple]]:
    """用 bottle YOLO 偵測，回傳 (商品名稱列表, [(cls_id, name, conf, (x1,y1,x2,y2)), ...])。"""
    results = yolo_model(pil_image, conf=BOTTLE_CONF_THRESHOLD, verbose=False)
    detected = []
    bottle_bboxes = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = LABEL_NAMES.get(cls_id, f"未知({cls_id})")
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            detected.append(name)
            bottle_bboxes.append((cls_id, name, conf, (x1, y1, x2, y2)))
            print(f"[YOLO bottle] {name} (cls={cls_id}, conf={conf:.2f})")

    return detected, bottle_bboxes


@app.get("/")
async def root():
    return {
        "message": "OCR API",
        "version": "1.0.0",
    }


@app.post("/inventory_base64")
async def inventory_base64(request: Base64ImageRequest):
    start_time = time.time()
    
    print("image received")
    # 1. 解碼圖片
    try:
        t0 = time.time()
        image_data = base64.b64decode(request.image_base64)
        pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")
        print(f"[IMAGE] decode={round(time.time()-t0, 3)}s, size={len(image_data)} bytes, width={pil_image.width}, height={pil_image.height}")
    except:
        raise HTTPException(status_code=400, detail="圖片解碼失敗")

    # 2. YOLO bottle 偵測
    t0 = time.time()
    detected_names, bottle_bboxes = detect_and_label(pil_image)
    print(f"[YOLO bottle] detect={round(time.time()-t0, 3)}s, found={len(detected_names)}")
    if not detected_names:
        return {"status": 1, "data": "貨架上看起來沒有瓶子。"}

    # 3. YOLO cap 偵測，取得所有瓶蓋 bbox
    t0 = time.time()
    cap_results = cap_yolo_model(pil_image, conf=CAP_CONF_THRESHOLD, verbose=False)
    cap_bboxes = []
    for result in cap_results:
        for box in result.boxes:
            cap_conf = float(box.conf[0])
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            cap_bboxes.append((cap_conf, (x1, y1, x2, y2)))
            # print(f"[YOLO cap] bbox=({x1},{y1},{x2},{y2})")
    print(f"[YOLO cap] detect={round(time.time()-t0, 3)}s, found={len(cap_bboxes)}")

    # 3-1. Debug: 儲存標註圖 bottle and cap
    if bottle_bboxes or cap_bboxes:
        t0 = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        pil_image.save(os.path.join(DEBUG_DIR, f"input_{timestamp}.jpg"))

        overview = pil_image.copy()
        draw = ImageDraw.Draw(overview)
        for cls_id, name, conf, (x1, y1, x2, y2) in bottle_bboxes:
            bgr = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
            color = (bgr[2], bgr[1], bgr[0])
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.text((x1, max(0, y1 - 30)), f"{name} {conf:.2f}", fill=color, font=debug_font)
        for (cap_conf, (x1, y1, x2, y2)) in cap_bboxes:
            draw.rectangle([x1, y1, x2, y2], outline="blue", width=2)
            draw.text((x1, max(0, y1 - 30)), f"cap {cap_conf:.2f}", fill="blue", font=debug_font)
        overview.save(os.path.join(DEBUG_DIR, f"overview_{timestamp}.jpg"))

        iw, ih = pil_image.width, pil_image.height

        # label: bottle+cap (class 0 = cap, class 1..N = bottle cls_id+1)
        cap_bottle_lines = []
        for cls_id, _name, _conf, (x1, y1, x2, y2) in bottle_bboxes:
            cx, cy = (x1 + x2) / 2 / iw, (y1 + y2) / 2 / ih
            w,  h  = (x2 - x1) / iw,      (y2 - y1) / ih
            cap_bottle_lines.append(f"{cls_id + 1} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        for _conf, (x1, y1, x2, y2) in cap_bboxes:
            cx, cy = (x1 + x2) / 2 / iw, (y1 + y2) / 2 / ih
            w,  h  = (x2 - x1) / iw,      (y2 - y1) / ih
            cap_bottle_lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        with open(os.path.join(DEBUG_DIR, f"input_{timestamp}_cap.txt"), "w") as f:
            f.write("\n".join(cap_bottle_lines))

        # label: bottle only
        bottle_lines = []
        for cls_id, _name, _conf, (x1, y1, x2, y2) in bottle_bboxes:
            cx, cy = (x1 + x2) / 2 / iw, (y1 + y2) / 2 / ih
            w,  h  = (x2 - x1) / iw,      (y2 - y1) / ih
            bottle_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        with open(os.path.join(DEBUG_DIR, f"input_{timestamp}_bottle.txt"), "w") as f:
            f.write("\n".join(bottle_lines))

        print(f"image saving time={round(time.time()-t0, 3)}s")

    # 4. 將有 overlap 的 cap bbox 歸為一群，再與 bottle bbox 比對，計算各 bottle 類別瓶數
    t0 = time.time()
    if cap_bboxes and bottle_bboxes:
        cap_bbox_coords = [bbox for _, bbox in cap_bboxes]
        cap_groups = group_overlapping_bboxes(cap_bbox_coords)
        bottle_counts: Counter = Counter()

        for group_indices in cap_groups:
            # 計算此 cap group 的 union bbox
            gx1 = min(cap_bbox_coords[i][0] for i in group_indices)
            gy1 = min(cap_bbox_coords[i][1] for i in group_indices)
            gx2 = max(cap_bbox_coords[i][2] for i in group_indices)
            gy2 = max(cap_bbox_coords[i][3] for i in group_indices)
            group_bbox = (gx1, gy1, gx2, gy2)

            # 找 IoU 最大的 bottle
            best_iou, best_name = 0.0, None
            for _cls_id, name, _conf, bbox in bottle_bboxes:
                score = bbox_iou(group_bbox, bbox)
                if score > best_iou:
                    best_iou, best_name = score, name

            if best_name:
                bottle_counts[best_name] += len(group_indices)
                # print(f"[CAP→BOTTLE] group={group_bbox} → {best_name} (iou={best_iou:.2f}, caps={len(group_indices)})")

        counts = dict(bottle_counts)
    else:
        # cap 模型無偵測結果時，退回 bottle 直接計數
        counts = dict(Counter(detected_names))
    print(f"matching time={round(time.time()-t0, 6)}ms")

    # 5. 組合成文字給 llama.cpp
    scan_list_str = "\n".join([f"- {k}: {v} 瓶" for k, v in counts.items()])
    print(f"=====SYSTEM_PROMPT=====")
    print(f"{scan_list_str}")
    print(f"==========")

    # 5-1. 若問題為盤點/統計，直接格式化輸出，不經過 ministral
    if any(kw in request.question for kw in ("盤點商品", "統計商品")):
        answer = "\n".join([f"{k} 有 {v} 瓶" for k, v in counts.items()])
        print(f"⚡ 耗時: {round(time.time() - start_time, 2)}s")
        print(f"=====回答(直接輸出)======")
        print(answer)
        print(f"==============")
        return {"status": 1, "data": answer}

    # 5-2. 其他問題走 ministral 推理
    t0 = time.time()
    response = client.chat.completions.create(
        model="ministral_3_3b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE.format(scan_list=scan_list_str)},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": request.question},
                ],
            },
        ],
        temperature=0,
    )

    print(f"vlm response time={round(time.time()-t0, 3)}s")

    print(f"⚡ 耗時: {round(time.time() - start_time, 2)}s")
    print(f"=====回答======")
    print(f"{response.choices[0].message.content}")
    print(f"==============")
    return {"status": 1, "data": response.choices[0].message.content}



def glm_ocr_ollama(base64_image):
    response = ollama.chat(
        model="glm-ocr:q8_0",
        messages=[
            {
                "role": "user",
                "content": "Text Recognition:",
                "images": [base64_image],
            }
        ],
    )

    return response["message"]["content"]


@app.post("/glm_ocr_inference_base64")
async def glm_ocr_inference_base64(request: Base64ImageRequest):
    output = ""

    try:
        output = glm_ocr_ollama(request.image_base64)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    print("OCR Result:", output)

    elements = output.split("\n")

    if len(elements) == 0:
        return JSONResponse(content={"count": 0, "date": None})
    elif len(elements) == 1:
        result = DateValidator.extract_expiry_date(output)
        print(f"1 result:{result}")
        return JSONResponse(content=result)
    elif len(elements) > 1:
        result = DateValidator.extract_multiple_dates(output)
        print(f"2 result:{result}")
        return JSONResponse(content=result)


if __name__ == "__main__":
    import uvicorn

    ui.run_with(app, title="Good API", favicon="🍵", dark=False)
    uvicorn.run(app, host="0.0.0.0", port=8888)
