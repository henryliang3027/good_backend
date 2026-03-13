import os
import base64
import io
import time
from collections import Counter
from contextlib import asynccontextmanager
from datetime import datetime
import signal
import cv2
import numpy as np
import chromadb
import ollama
from PIL import ImageDraw, ImageFont
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
from ultralytics import YOLO
from sentence_transformers import SentenceTransformer
from openai import OpenAI

import subprocess
from utils.date_validator import DateValidator

# ========== Model & DB Config ==========
BOTTLE_CLASS_ID = 39
OLLAMA_MODEL = "ministral-3:3b"
CONF_THRESHOLD = 0.8

# --- Cosine 門檻值建議 ---
# 0.0 ~ 0.2: 極度相似 (同一產品)
# 0.2 ~ 0.35: 相似 (同系列不同角度)
# > 0.35: 視為未知商品
COSINE_THRESHOLD = 0.35


class Base64ImageRequest(BaseModel):
    image_base64: str
    question: str = "請統計商品"

# ========== Global Objects ==========
yolo_model = None
clip_model = None
chroma_client = None
collection = None

SYSTEM_PROMPT_TEMPLATE = """你是一位專業的超商貨架分析員。請根據以下掃描結果清單回答用戶問題。

【掃描結果清單】
{scan_list}

【回答規則與範例】
1. 若用戶詢問「統計商品」或類似整體盤點的問題，嚴格遵守以下格式：
   根據掃描結果清單，以下是各商品的數量統計：
   [商品名稱] 有 [數量] 瓶
   (以此類推，每行一個，不使用列點符號或顏色前綴)

2. 若用戶詢問「有幾瓶 [特定商品]」，嚴格遵守以下格式：
   [特定商品] 有 [數量] 瓶
   (如果該商品完全不存在，請回：沒有找到您指定的商品)

3. 輸出禁止包含額外的解釋或結尾客套話。
4. 忽略顏色前綴（例如「灰色茶裏王」僅回答「茶裏王」），以掃描清單中的商品名稱為主。
5. 必須使用繁體中文。
6. 用戶輸入可能來自語音轉文字（STT），若遇到諧音詞，請自動對應到清單中最接近的商品

【範例】
用戶：統計商品
回答：
根據掃描結果清單，以下是各商品的數量統計：
原萃台灣青茶 有 1 瓶
茶裏王半熟金萱 有 1 瓶
茶裏王白毫烏龍 有 1 瓶
無加糖LP33機能優酪乳 有 2 瓶

用戶：有幾瓶茶裏王白毫烏龍？
回答：茶裏王白毫烏龍 有 1 瓶
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
    global yolo_model, clip_model, chroma_client, collection
    print("🚀 正在啟動系統並載入模型...")
    
    # 1. 載入視覺模型
    yolo_model = YOLO("yolo11m.pt")
    clip_model = SentenceTransformer('clip-ViT-B-32')
    
    # 2. 初始化 ChromaDB (持久化儲存於本地資料夾)
    chroma_client = chromadb.PersistentClient(path="./drink_vector_db")
    # collection = chroma_client.get_or_create_collection(name="drink_catalog", metadata={"hnsw:space": "cosine"})
    collection = chroma_client.get_or_create_collection(name="drink_catalog")
    
    existing_count = collection.count()
    print(f"📦 ChromaDB 已就緒，目前資料庫包含 {existing_count} 筆特徵資料。")

    # Debug: 檢查 DB 中每筆 CLIP 特徵的維度與幾何長度（L2 norm）
    if existing_count > 0:
        db_items = collection.get(include=["embeddings", "metadatas"])
        print("[DEBUG] DB 特徵向量資訊:")
        for item_id, meta, emb in zip(db_items['ids'], db_items['metadatas'], db_items['embeddings']):
            vec = np.array(emb)
            label = f"{meta.get('brand','')}{meta.get('flavor','')}" or item_id
            print(f"  {label} | dim={vec.shape[0]} | L2 norm={np.linalg.norm(vec):.6f}")

    # 啟動時執行
    start_llama_server()
    yield
    # 關閉時執行
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
_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
_debug_font = ImageFont.truetype(_FONT_PATH, size=14)

def detect_and_crop_bottles(pil_image: Image.Image):
    results = yolo_model(pil_image, conf=CONF_THRESHOLD, verbose=False)
    cropped_images = []
    boxes_found = []

    for result in results:
        for box in result.boxes:
            if int(box.cls[0]) == BOTTLE_CLASS_ID:
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cropped_images.append(pil_image.crop((x1, y1, x2, y2)))
                boxes_found.append((x1, y1, x2, y2, conf))

    # Debug: 為每張輸入圖建立資料夾，存原圖 bbox 標註 + 各 crop
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    debug_folder = os.path.join(DEBUG_DIR, timestamp)
    if boxes_found:
        os.makedirs(debug_folder, exist_ok=True)

        # 儲存原始輸入圖
        pil_image.save(os.path.join(debug_folder, "input.jpg"))

        # 儲存原圖並標上 bbox
        overview_img = pil_image.copy()
        draw = ImageDraw.Draw(overview_img)
        for i, (x1, y1, x2, y2, conf) in enumerate(boxes_found):
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, max(0, y1 - 15)), f"#{i} {conf:.2f}", fill="red", font=_debug_font)
        overview_img.save(os.path.join(debug_folder, "overview.jpg"))

        # 儲存所有 cropped bottle 原圖
        for i, crop in enumerate(cropped_images):
            crop.save(os.path.join(debug_folder, f"crop_{i:02d}_raw.jpg"))

        print(f"[DEBUG] 偵測到 {len(boxes_found)} 個瓶子，debug 資料夾: {debug_folder}")

    return cropped_images, debug_folder


# ========== Feature Extraction ==========

def get_hsv_features(image_pil):
    image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)
    # 3D 直方圖: H=8, S=2, V=2 → 32 維
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 2, 2], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist.tolist()


def combine_features(clip_emb, hsv_emb, color_weight=1.5):
    clip_np = np.array(clip_emb)
    clip_norm = clip_np / np.linalg.norm(clip_np)
    hsv_np = np.array(hsv_emb)
    hsv_weighted = (hsv_np / np.linalg.norm(hsv_np)) * color_weight
    # 512 + 32 = 544 維
    return np.hstack((clip_norm, hsv_weighted)).tolist()


def match_with_chroma(pil_image: Image.Image, debug_folder: str, crop_index: int):
    """CLIP+HSV 544 維向量比對，找出 cosine distance 最小的商品。"""
    clip_emb = clip_model.encode(pil_image).tolist()
    hsv_emb = get_hsv_features(pil_image)
    img_emb = combine_features(clip_emb, hsv_emb)

    total = collection.count()
    all_results = collection.query(
        query_embeddings=[img_emb],
        n_results=max(total, 1),
        include=["metadatas", "distances"]
    )
    distances = list(zip(all_results['metadatas'][0], all_results['distances'][0]))

    print(f"[DEBUG] crop #{crop_index} CLIP 距離:")
    for meta, dist in distances:
        label = f"{meta.get('brand','')}{meta.get('flavor','')}"
        print(f"  {label}: {dist:.4f}")

    if debug_folder:
        crop_debug = pil_image.copy()
        draw = ImageDraw.Draw(crop_debug)
        line_height = 14
        y_offset = 4
        for meta, dist in distances:
            name = f"{meta.get('brand','')}{meta.get('flavor','')}"
            text = f"{name}: {dist:.4f}"
            bbox = draw.textbbox((4, y_offset), text, font=_debug_font)
            draw.rectangle(bbox, fill="white")
            draw.text((4, y_offset), text, fill="red", font=_debug_font)
            y_offset += line_height
        crop_debug.save(os.path.join(debug_folder, f"crop_{crop_index:02d}.jpg"))

    best_meta, best_dist = distances[0]
    # if best_dist > COSINE_THRESHOLD:
    #     return "未知商品"

    return f"{best_meta.get('brand','')}{best_meta.get('flavor','')}"

# ========== CRUD Endpoints (管理資料庫) ==========

@app.post("/db/add", summary="[CRUD] 新增飲料特徵到資料庫")
async def add_to_db(
    brand: str = Form(...),
    flavor: str = Form(...),
    color: str = Form(""),
    file: UploadFile = File(...)
):
    """上傳一張 crop 好的瓶子，存入 ChromaDB。

    - brand: 品牌，例如「茶裏王」
    - flavor: 口味，例如「台式綠茶」
    - color: 瓶身顏色，例如「黃色」
    """
    item_id = f"{brand}{flavor}"  # 以 brand+flavor 作為唯一 ID
    image = Image.open(file.file).convert("RGB")
    clip_emb = clip_model.encode(image).tolist()
    hsv_emb = get_hsv_features(image)
    embedding = combine_features(clip_emb, hsv_emb)

    collection.upsert(
        ids=[item_id],
        embeddings=[embedding],
        metadatas=[{
            "brand": brand,
            "flavor": flavor,
            "color": color,
        }]
    )
    return {"status": "success", "message": f"已存入: {brand} {flavor} ({color})"}

@app.get("/db/list", summary="[CRUD] 列出目前所有商品")
async def list_db():
    results = collection.get()
    return {"total": len(results['ids']), "items": results['metadatas']}

@app.delete("/db/{name}", summary="[CRUD] 刪除特定商品")
async def delete_item(name: str):
    collection.delete(ids=[name])
    return {"status": "deleted", "item": name}


@app.get("/")
async def root():
    return {
        "message": "OCR API",
        "version": "1.0.0",
    }


@app.post("/inventory_base64")
async def inventory_base64(request: Base64ImageRequest):
    start_time = time.time()
    
    # 1. 解碼圖片
    try:
        image_data = base64.b64decode(request.image_base64)
        pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")
    except:
        raise HTTPException(status_code=400, detail="圖片解碼失敗")

    # 2. YOLO 偵測與裁切
    crops, debug_folder = detect_and_crop_bottles(pil_image)
    if not crops:
        return {"status": 1, "data": "貨架上看起來沒有瓶子。"}

    # 3. CLIP 向量比對
    detected_names = [match_with_chroma(img, debug_folder, i) for i, img in enumerate(crops)]
    counts = dict(Counter(detected_names))
    
    # 4. 組合成文字給 Ollama
    scan_list_str = "\n".join([f"- {k}: {v} 瓶" for k, v in counts.items()])
    print(f"=====SYSTEM_PROMPT=====")
    print(f"{scan_list_str}")
    print(f"==========")

    # 5. llama.cpp 推理
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

    uvicorn.run(app, host="0.0.0.0", port=8888)
