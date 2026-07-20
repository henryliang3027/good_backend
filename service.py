import os
import base64
import io
import time
from collections import Counter
from contextlib import asynccontextmanager
import signal
from PIL import Image, ImageDraw, ImageFont, ImageOps
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from ultralytics import YOLO
from openai import OpenAI

import subprocess
from nicegui import ui
from utils.date_validator import DateValidator

from datetime import datetime

_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
debug_font = ImageFont.truetype(_FONT_PATH, size=18)


# ========== Model & DB Config ==========
YOLO_MODEL_PATH = "yolo_model/bottle_detector/best_M_130_20260425.pt"
CAP_YOLO_MODEL_PATH = "yolo_model/cap_detector/best_L_101_20260424.pt"
DATE_YOLO_MODEL_PATH = "yolo_model/date_detector/best_v11m_1280_20260602.pt"
BOX_OBB_YOLO_MODEL_PATH = "yolo_model/box_obb_detector/best_26m_obb_640_20260624.pt"
CAP_CONF_THRESHOLD = 0.80
BOTTLE_CONF_THRESHOLD = 0.80
DATE_CONF_THRESHOLD = 0.80
BOX_OBB_CONF_THRESHOLD = 0.60

BOX_OBB_LABEL_NAMES = {
    0: "來一客牛肉蔬菜風味",
    1: "來一客韓式泡菜風味",
    2: "來一客鮮蝦魚板風味",
    3: "黑松蜜桃C",
    4: "維他露P",
    5: "樂事洋芋片青檸享清新口味",
    6: "科學麵",
    7: "真魷味紅燒口味",
    8: "玉黍叔甜辣口味",
    9: "Mos Burger",
    10: "益生菌蒟蒻果凍"
}

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
date_yolo_model = None
box_obb_yolo_model = None


SYSTEM_PROMPT_RULES = """
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
   - 禁止列出名稱不以該品牌為開頭的商品。例如詢問「原萃」時，不可列出「茶裏王」開頭的商品。
   - 若清單中完全沒有符合的商品，僅回答：沒有找到您指定的商品

3. 「有幾瓶 [完整商品名稱]」——完整名稱查詢（問題中包含完整商品名，例如「有幾瓶原萃鐵觀音」）：
   - 只回答該指定商品，不可列出其他商品。
   - 若清單中有該商品，回答：[商品名稱] 有 [數量] 瓶
   - 若清單中沒有該商品，回答：沒有找到您指定的商品

4. 禁止輸出任何額外說明、前言或結尾客套話。
5. 必須使用繁體中文。
6. 若遇到語音辨識諧音詞，自動對應到清單中最相似的商品名稱。
7. 商品名稱必須與掃描清單完全一致，逐字照抄，禁止增加、刪除或重複任何文字。
"""


def filter_answer_by_scan_list(answer: str, scan_list: list[tuple[str, int]]) -> str:
    """將模型回答逐行與 scan_list 比對，移除不在清單中的商品行。
    若全部移除則回傳「沒有找到您指定的商品」。"""
    scan_names = {name for name, _ in scan_list}
    lines = [line for line in answer.splitlines() if line]

    valid_lines = []
    for line in lines:
        if "沒有找到您指定的商品" in line:
            valid_lines.append(line)
        elif " 有 " in line and line.endswith(" 瓶"):
            product_name = line.split(" 有 ")[0]
            if product_name in scan_names:
                valid_lines.append(line)

    if not valid_lines:
        return "沒有找到您指定的商品"
    return "\n".join(valid_lines)


def build_system_prompt(scan_list: list[tuple[str, int]]) -> str:
    items = "\n".join(f"- {name}: {qty} 瓶" for name, qty in scan_list)
    return f"你是一位專業的超商貨架分析員。請根據以下掃描結果清單回答用戶問題。\n\n【掃描結果清單】\n{items}\n{SYSTEM_PROMPT_RULES}"


client = OpenAI(
    base_url="http://127.0.0.1:8881/v1",
    api_key="no-key-needed",
)

glm_ocr_client = OpenAI(
    base_url="http://127.0.0.1:8882/v1",
    api_key="no-key-needed",
)


# llama-server 進程
llama_process = None
glm_ocr_process = None

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

GLM_OCR_SERVER_CMD = [
    "./llama.cpp/build/bin/llama-server",
    "-m",
    "glm_ocr/GLM-OCR-Q8_0.gguf",
    "--mmproj",
    "glm_ocr/mmproj-GLM-OCR-Q8_0.gguf",
    "--host",
    "0.0.0.0",
    "--port",
    "8882",
    "--ctx-size",
    "4096",
    "-ngl",
    "-1",
]


def _start_server(cmd: list[str], name: str) -> subprocess.Popen:
    print(f"Starting {name}...")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    time.sleep(5)
    print(f"{name} started with PID: {proc.pid}")
    return proc


def _stop_server(proc: subprocess.Popen | None, name: str) -> None:
    if proc:
        print(f"Stopping {name} (PID: {proc.pid})...")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=10)
        except (subprocess.TimeoutExpired, ProcessLookupError):
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        print(f"{name} stopped.")


def start_llama_server():
    global llama_process
    llama_process = _start_server(LLAMA_SERVER_CMD, "llama-server (ministral)")


def stop_llama_server():
    global llama_process
    _stop_server(llama_process, "llama-server (ministral)")
    llama_process = None


def start_glm_ocr_server():
    global glm_ocr_process
    glm_ocr_process = _start_server(GLM_OCR_SERVER_CMD, "llama-server (glm-ocr)")


def stop_glm_ocr_server():
    global glm_ocr_process
    _stop_server(glm_ocr_process, "llama-server (glm-ocr)")
    glm_ocr_process = None


def _signal_handler(sig, frame):
    stop_llama_server()
    stop_glm_ocr_server()
    raise SystemExit(0)

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global yolo_model, cap_yolo_model, date_yolo_model, box_obb_yolo_model
    print("🚀 正在啟動系統並載入模型...")

    yolo_model = YOLO(YOLO_MODEL_PATH)
    print(f"✅ YOLO 模型載入完成: {YOLO_MODEL_PATH}")

    cap_yolo_model = YOLO(CAP_YOLO_MODEL_PATH)
    print(f"✅ Cap YOLO 模型載入完成: {CAP_YOLO_MODEL_PATH}")


    date_yolo_model = YOLO(DATE_YOLO_MODEL_PATH)
    print(f"✅ Date YOLO 模型載入完成: {DATE_YOLO_MODEL_PATH}")

    box_obb_yolo_model = YOLO(BOX_OBB_YOLO_MODEL_PATH)
    print(f"✅ Box OBB YOLO 模型載入完成: {BOX_OBB_YOLO_MODEL_PATH}")

    start_llama_server()
    start_glm_ocr_server()
    yield
    stop_llama_server()
    stop_glm_ocr_server()


app = FastAPI(
    title="Good API v1",
    description="test",
    version="1.0.0",
    lifespan=lifespan,
)



class Base64ImageRequest(BaseModel):
    image_base64: str
    question: str = "請統計圖中的商品"
    mode: int = 1


# ========== Helper Functions ==========

DEBUG_DIR = "detected_bottle"
LABEL_CAPS_DIR = "label_caps"
LABEL_BOTTLES_DIR = "label_bottles"
ORIGINAL_BOTTLE_AND_CAP_IMAGES_DIR = "original_bottle_and_cap_images"
ORIGINAL_BOX_IMAGES_DIR = "original_box_images"
DETECTED_BOX_DIR = "detected_box"
LABEL_OBB_BOXES_DIR = "label_obb_boxes"
LABEL_BOXES_DATE_DIR = "label_boxes_date"
CROPPED_DATE_IMAGE_DIR = "cropped_date_image"
CROPPED_BOXES_DIR = "cropped_boxes"


for _dir in [DEBUG_DIR, LABEL_CAPS_DIR, LABEL_BOTTLES_DIR, 
             ORIGINAL_BOTTLE_AND_CAP_IMAGES_DIR, 
             ORIGINAL_BOX_IMAGES_DIR, DETECTED_BOX_DIR,
             LABEL_OBB_BOXES_DIR, LABEL_BOXES_DATE_DIR, CROPPED_DATE_IMAGE_DIR,
             CROPPED_BOXES_DIR]:
    os.makedirs(_dir, exist_ok=True)


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



def point_in_polygon(px: float, py: float, polygon: list) -> bool:
    """Ray casting：判斷點 (px,py) 是否落在多邊形 polygon (含旋轉框) 內。"""
    n = len(polygon)
    inside = False
    x1, y1 = polygon[0]
    for i in range(1, n + 1):
        x2, y2 = polygon[i % n]
        if ((y1 > py) != (y2 > py)) and (px < (x2 - x1) * (py - y1) / (y2 - y1 + 1e-9) + x1):
            inside = not inside
        x1, y1 = x2, y2
    return inside


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
    pil_image = None
    try:
        t0 = time.time()
        image_data = base64.b64decode(request.image_base64)
        if request.mode == 2:
            pil_image = ImageOps.exif_transpose(Image.open(io.BytesIO(image_data))).convert("RGB")
            # shelf_ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            # pil_image.save(os.path.join(DETECTED_SHELF_DIR, f"shelf_{shelf_ts}.jpg"))
        else:
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
    t0 = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    pil_image.save(os.path.join(ORIGINAL_BOTTLE_AND_CAP_IMAGES_DIR, f"input_{timestamp}.jpg"))

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

    # label: cap (class 0 = cap, class 1..N = bottle cls_id+1)
    cap_lines = []
    for _conf, (x1, y1, x2, y2) in cap_bboxes:
        cx, cy = (x1 + x2) / 2 / iw, (y1 + y2) / 2 / ih
        w,  h  = (x2 - x1) / iw,      (y2 - y1) / ih
        cap_lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    with open(os.path.join(LABEL_CAPS_DIR, f"input_{timestamp}.txt"), "w") as f:
        f.write("\n".join(cap_lines))

    # label: bottle only
    bottle_lines = []
    for cls_id, _name, _conf, (x1, y1, x2, y2) in bottle_bboxes:
        cx, cy = (x1 + x2) / 2 / iw, (y1 + y2) / 2 / ih
        w,  h  = (x2 - x1) / iw,      (y2 - y1) / ih
        bottle_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    with open(os.path.join(LABEL_BOTTLES_DIR, f"input_{timestamp}.txt"), "w") as f:
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

    counts = dict(sorted(counts.items(), key=lambda x: x[0]))

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
            {"role": "system", "content": build_system_prompt(list(counts.items()))},
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

    answer = filter_answer_by_scan_list(
        response.choices[0].message.content, list(counts.items())
    )

    print(f"⚡ 耗時: {round(time.time() - start_time, 2)}s")
    print(f"=====回答======")
    print(f"{answer}")
    print(f"==============")
    return {"status": 1, "data": answer}




def glm_ocr_llama(base64_image: str) -> str:
    response = glm_ocr_client.chat.completions.create(
        model="glm-ocr",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Text Recognition:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ],
            }
        ],
        temperature=0,
    )
    return response.choices[0].message.content


@app.post("/glm_ocr_inference_base64")
async def glm_ocr_inference_base64(request: Base64ImageRequest):
    output = ""

    try:
        output = glm_ocr_llama(request.image_base64)
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


BOX_COLORS = [
    (255,   0,   0),  # red
    (  0, 200,   0),  # green
    (  0,   0, 255),  # blue
    (255, 165,   0),  # orange
    (128,   0, 128),  # purple
    (  0, 200, 200),  # cyan
    (255, 105, 180),  # pink
    (139,  69,  19),  # brown
]


class BoxDetectionRequest(BaseModel):
    image_base64: str


def _parse_date_from_ocr(ocr_text: str) -> dict | None:
    result = DateValidator.extract_expiry_date(ocr_text)
    if result["count"] == 0:
        return None
    return result["date"]["expiration"]


@app.post("/box_date_detection")
async def box_date_detection(request: BoxDetectionRequest):
    try:
        image_data = base64.b64decode(request.image_base64)
        pil_image = ImageOps.exif_transpose(Image.open(io.BytesIO(image_data))).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="圖片解碼失敗")

    print(f"[IMAGE] width={pil_image.width}, height={pil_image.height}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    pil_image.save(os.path.join(ORIGINAL_BOX_IMAGES_DIR, f"input_{timestamp}.jpg"))
    overview = pil_image.copy()
    draw = ImageDraw.Draw(overview)

    # 1. OBB YOLO 偵測商品箱子
    box_obb_results = box_obb_yolo_model(pil_image, conf=BOX_OBB_CONF_THRESHOLD, verbose=False)
    box_detections = []  # list of {"name": str, "bbox": (x1,y1,x2,y2), "obb": [[x,y],...]}

    iw, ih = pil_image.width, pil_image.height
    obb_label_lines = []

    for result in box_obb_results:
        if result.obb is None:
            continue
        for i in range(len(result.obb)):
            cls_id = int(result.obb.cls[i])
            conf = float(result.obb.conf[i])
            name = BOX_OBB_LABEL_NAMES.get(cls_id, f"未知({cls_id})")
            x1, y1, x2, y2 = (int(v) for v in result.obb.xyxy[i].tolist())
            obb_pts = [[int(v[0]), int(v[1])] for v in result.obb.xyxyxyxy[i].tolist()]
            box_detections.append({"name": name, "bbox": [x1, y1, x2, y2], "obb": obb_pts})
            color = BOX_COLORS[cls_id % len(BOX_COLORS)]
            pts = [(int(v[0]), int(v[1])) for v in result.obb.xyxyxyxy[i].tolist()]
            draw.polygon(pts, outline=color, width=4)
            draw.text((x1, max(0, y1 - 30)), f"{name} {conf:.2f}", fill=color, font=debug_font)
            print(f"[BOX OBB] {name} conf={conf:.2f} bbox=({x1},{y1},{x2},{y2})")

            box_crop = pil_image.crop((x1, y1, x2, y2))
            box_crop.save(os.path.join(CROPPED_BOXES_DIR, f"box_{timestamp}_{i}_1.jpg"), quality=95, subsampling=0)

            corners = result.obb.xyxyxyxy[i].tolist()
            norm_pts = " ".join(f"{v[0]/iw:.6f} {v[1]/ih:.6f}" for v in corners)
            obb_label_lines.append(f"{cls_id} {norm_pts}")

    with open(os.path.join(LABEL_OBB_BOXES_DIR, f"input_{timestamp}.txt"), "w") as f:
        f.write("\n".join(obb_label_lines))

    # 2. 在整張圖上跑 date YOLO，偵測 date region
    date_results = date_yolo_model(pil_image, conf=DATE_CONF_THRESHOLD, verbose=False)
    date_regions = []  # list of {"bbox": [x1,y1,x2,y2], "conf": float}

    for result in date_results:
        for box in result.boxes:
            conf = float(box.conf[0])
            dx1, dy1, dx2, dy2 = (int(v) for v in box.xyxy[0].tolist())
            date_regions.append({"bbox": [dx1, dy1, dx2, dy2], "conf": conf})
            draw.rectangle([dx1, dy1, dx2, dy2], outline=(0, 220, 0), width=2)
            draw.text((dx1, max(0, dy1 - 15)), f"date {conf:.2f}", fill=(0, 220, 0), font=debug_font)
            print(f"[DATE] conf={conf:.2f} bbox=({dx1},{dy1},{dx2},{dy2})")

    # 3. 配對每個 date region，看落在哪個 box 的 obb 內，再對該區域做 OCR
    box_best_date = [None] * len(box_detections)  # {"date": ..., "date_bbox": ...}

    for date_idx, date_region in enumerate(date_regions):
        dx1, dy1, dx2, dy2 = date_region["bbox"]
        cx, cy = (dx1 + dx2) / 2, (dy1 + dy2) / 2

        matched_box_idx = None
        for box_idx, box_det in enumerate(box_detections):
            if point_in_polygon(cx, cy, box_det["obb"]):
                matched_box_idx = box_idx
                break

        if matched_box_idx is None:
            print(f"[MATCH] date region {date_idx} 未落在任何 box 內，略過")
            continue

        date_crop = pil_image.crop((dx1, dy1, dx2, dy2))
        date_crop.save(os.path.join(CROPPED_DATE_IMAGE_DIR, f"date_{timestamp}_{date_idx}_1.jpg"), quality=95, subsampling=0)
        buf = io.BytesIO()
        date_crop.save(buf, format="JPEG", quality=95, subsampling=0)
        date_crop_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        ocr_text = glm_ocr_llama(date_crop_b64)
        parsed_date = _parse_date_from_ocr(ocr_text)
        print(f"[DATE OCR] box={box_detections[matched_box_idx]['name']} text={ocr_text!r} → {parsed_date}")

        if parsed_date is not None and box_best_date[matched_box_idx] is None:
            box_best_date[matched_box_idx] = {
                "date": parsed_date,
                "date_bbox": [dx1, dy1, dx2, dy2],
            }

    output = []
    for box_idx, box_det in enumerate(box_detections):
        matched = box_best_date[box_idx] or {"date": None, "date_bbox": None}
        output.append({
            "name": box_det["name"],
            "obb": box_det["obb"],
            "date": matched["date"],
            "date_bbox": matched["date_bbox"],
        })
        print(f"[MATCH] {box_det['name']} → {matched['date']}")

    overview.save(os.path.join(DETECTED_BOX_DIR, f"box_{timestamp}.jpg"), quality=95, subsampling=0)

    return {"status": "1", "data": output}




# @app.post("/box_date_detection2")
# async def box_date_detection2(request: BoxDetectionRequest):
#     try:
#         image_data = base64.b64decode(request.image_base64)
#         pil_image = ImageOps.exif_transpose(Image.open(io.BytesIO(image_data))).convert("RGB")
#     except Exception:
#         raise HTTPException(status_code=400, detail="圖片解碼失敗")

#     print(f"[IMAGE] width={pil_image.width}, height={pil_image.height}")
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
#     pil_image.save(os.path.join(ORIGINAL_BOX_IMAGES_DIR, f"input2_{timestamp}.jpg"))
#     overview = pil_image.copy()
#     draw = ImageDraw.Draw(overview)

#     # 1. OBB YOLO 偵測商品箱子
#     iw, ih = pil_image.width, pil_image.height
#     box_obb_results = box_obb_yolo_model(pil_image, conf=BOX_OBB_CONF_THRESHOLD, verbose=False)
#     box_detections = []
#     obb_label_lines = []

#     for result in box_obb_results:
#         if result.obb is None:
#             continue
#         for i in range(len(result.obb)):
#             cls_id = int(result.obb.cls[i])
#             conf = float(result.obb.conf[i])
#             name = BOX_OBB_LABEL_NAMES.get(cls_id, f"未知({cls_id})")
#             x1, y1, x2, y2 = (int(v) for v in result.obb.xyxy[i].tolist())
#             obb_pts = [[int(v[0]), int(v[1])] for v in result.obb.xyxyxyxy[i].tolist()]
#             box_detections.append({"name": name, "bbox": [x1, y1, x2, y2], "obb": obb_pts, "cls_id": cls_id, "conf": conf})
#             color = BOX_COLORS[cls_id % len(BOX_COLORS)]
#             pts = [(int(v[0]), int(v[1])) for v in result.obb.xyxyxyxy[i].tolist()]
#             draw.polygon(pts, outline=color, width=4)
#             draw.text((x1, max(0, y1 - 30)), f"{name} {conf:.2f}", fill=color, font=debug_font)
#             print(f"[BOX OBB] {name} conf={conf:.2f} bbox=({x1},{y1},{x2},{y2})")

#             corners = result.obb.xyxyxyxy[i].tolist()
#             norm_pts = " ".join(f"{v[0]/iw:.6f} {v[1]/ih:.6f}" for v in corners)
#             obb_label_lines.append(f"{cls_id} {norm_pts}")

#     with open(os.path.join(LABEL_OBB_BOXES_DIR, f"input2_{timestamp}.txt"), "w") as f:
#         f.write("\n".join(obb_label_lines))

#     # 2. 對每個偵測到的箱子裁切，再個別跑 date YOLO + OCR
#     date_label_lines = []
#     output = []
#     date_number = 0
#     for box_det in box_detections:
#         bx1, by1, bx2, by2 = box_det["bbox"]
#         box_crop = pil_image.crop((bx1, by1, bx2, by2))
#         bcw, bch = box_crop.width, box_crop.height

#         date_results = date_yolo_model(box_crop, conf=DATE_CONF_THRESHOLD, verbose=False)

#         best_date, best_date_bbox = None, None
        

#         for result in date_results:
#             for date_idx, box in enumerate(result.boxes):
#                 conf = float(box.conf[0])
#                 dx1, dy1, dx2, dy2 = (int(v) for v in box.xyxy[0].tolist())

#                 # 座標轉回原圖空間
#                 abs_x1 = bx1 + dx1
#                 abs_y1 = by1 + dy1
#                 abs_x2 = bx1 + dx2
#                 abs_y2 = by1 + dy2

#                 draw.rectangle([abs_x1, abs_y1, abs_x2, abs_y2], outline=(0, 220, 0), width=2)
#                 draw.text((abs_x1, max(0, abs_y1 - 15)), f"date {conf:.2f}", fill=(0, 220, 0), font=debug_font)
#                 print(f"[DATE in {box_det['name']}] conf={conf:.2f} bbox=({dx1},{dy1},{dx2},{dy2})")

#                 cx = ((dx1 + dx2) / 2) / bcw
#                 cy = ((dy1 + dy2) / 2) / bch
#                 w  = (dx2 - dx1) / bcw
#                 h  = (dy2 - dy1) / bch
#                 date_label_lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

#                 date_crop = box_crop.crop((dx1, dy1, dx2, dy2))
#                 print(f"date:{date_idx}")
#                 date_crop.save(os.path.join(CROPPED_DATE_IMAGE_DIR, f"date_{timestamp}_{date_number}_2.jpg"), quality=95, subsampling=0)
#                 date_number+=1
#                 buf = io.BytesIO()
#                 date_crop.save(buf, format="JPEG", quality=95, subsampling=0)
#                 crop_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

#                 ocr_text = glm_ocr_llama(crop_b64)
#                 # ocr_text = "20260101"
#                 parsed_date = _parse_date_from_ocr(ocr_text)
#                 print(f"[DATE OCR] text={ocr_text!r} → {parsed_date}")

#                 if best_date is None and parsed_date is not None:
#                     best_date = parsed_date
#                     best_date_bbox = [abs_x1, abs_y1, abs_x2, abs_y2]

#         output.append({
#             "name": box_det["name"],
#             "obb": box_det["obb"],
#             "date": best_date,
#             "date_bbox": best_date_bbox,
#         })
#         print(f"[MATCH] {box_det['name']} → {best_date}")

#     with open(os.path.join(LABEL_BOXES_DATE_DIR, f"input2_{timestamp}.txt"), "w") as f:
#         f.write("\n".join(date_label_lines))

#     # overview.save(os.path.join(DETECTED_BOX_DIR, f"box2_{timestamp}.jpg"), quality=95, subsampling=0)

#     return {"status": "1", "data": output}


if __name__ == "__main__":
    import uvicorn

    ui.run_with(app, title="Good API",  favicon="🍵", dark=False)
    uvicorn.run(app, host="0.0.0.0", port=8888)
