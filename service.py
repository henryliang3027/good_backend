import os
import base64
import io
import time
from collections import Counter
from contextlib import asynccontextmanager
from datetime import datetime
import signal
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
from rapidfuzz import process as fuzz_process, fuzz
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

# OCR + Fuzzy 比對後的 CLIP 驗證門檻
# 模糊比對找到候選後，用 CLIP cosine distance 做最終確認
FUZZY_CLIP_THRESHOLD = 0.15

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
    collection = chroma_client.get_or_create_collection(name="drink_catalog", metadata={"hnsw:space": "cosine"})
    
    existing_count = collection.count()
    print(f"📦 ChromaDB 已就緒，目前資料庫包含 {existing_count} 筆特徵資料。")

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


def fuzzy_match_ocr_to_db(ocr_text: str):
    """
    從 DB 取出所有 brand+flavor，用 rapidfuzz 模糊比對 OCR 文字，
    回傳最符合的 DB item ID（即 brand+flavor 字串）。
    處理形近字錯誤，例如「線→綠」、「古→甘」。
    """
    all_items = collection.get(include=["metadatas"])
    if not all_items['ids']:
        return None

    # 建立 {id: "brand+flavor"} 候選字典
    candidates = {}
    for item_id, meta in zip(all_items['ids'], all_items['metadatas']):
        brand = meta.get('brand', '')
        flavor = meta.get('flavor', '')
        candidates[item_id] = f"{brand}{flavor}"

    # partial_ratio 對形近字和部分匹配效果最好
    result = fuzz_process.extractOne(
        ocr_text,
        candidates,
        scorer=fuzz.partial_ratio,
        score_cutoff=50,
    )

    if result is None:
        print(f"[Fuzzy] 無法匹配 OCR 文字: {repr(ocr_text[:60])}")
        return None

    _, score, matched_id = result
    print(f"[Fuzzy] OCR 文字匹配 -> '{matched_id}' (score={score:.1f})")
    return matched_id


def match_bottle(pil_image: Image.Image, debug_folder: str, crop_index: int):
    """
    新版比對流程：
    1. CLIP encode crop 取得向量
    2. GLM OCR 辨識標籤文字
    3. rapidfuzz 模糊比對 DB 的 brand+flavor，找出候選商品
    4. 取 DB 該筆的 CLIP cosine distance，< FUZZY_CLIP_THRESHOLD 才確認命中
    """
    # Step 1: CLIP encode
    img_emb = clip_model.encode(pil_image).tolist()

    # Step 2: 查詢 DB 所有商品距離（供 debug 及後續驗證用）
    total = collection.count()
    all_results = collection.query(
        query_embeddings=[img_emb],
        n_results=max(total, 1),
        include=["metadatas", "distances"]
    )
    id_dist_map = {
        item_id: dist
        for item_id, dist in zip(all_results['ids'][0], all_results['distances'][0])
    }
    distances = list(zip(all_results['metadatas'][0], all_results['distances'][0]))

    print(f"[DEBUG] crop #{crop_index} CLIP 距離:")
    for meta, dist in distances:
        label = f"{meta.get('brand','')}{meta.get('flavor','')}"
        print(f"  {label}: {dist:.4f}")

    # Step 3: GLM OCR
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG")
    crop_b64 = base64.b64encode(buf.getvalue()).decode()
    try:
        ocr_text = glm_ocr_ollama(crop_b64)
        print(f"[OCR] crop #{crop_index}: {repr(ocr_text[:80])}")
    except Exception as e:
        print(f"[OCR] crop #{crop_index} 失敗: {e}")
        ocr_text = ""

    # Step 4: Fuzzy match OCR 文字 -> 候選 DB ID
    matched_id = fuzzy_match_ocr_to_db(ocr_text) if ocr_text else None

    # Step 5: CLIP 驗證 cosine distance < FUZZY_CLIP_THRESHOLD
    if matched_id is not None:
        cosine_dist = id_dist_map.get(matched_id)
        print(f"[Verify] '{matched_id}' CLIP distance = {cosine_dist:.4f} (threshold={FUZZY_CLIP_THRESHOLD})")
        if cosine_dist is not None and cosine_dist < FUZZY_CLIP_THRESHOLD:
            matched_name = matched_id  # DB id == brand+flavor
        else:
            print(f"[Verify] CLIP 驗證未通過，標記為未知商品")
            matched_name = "未知商品"
    else:
        matched_name = "未知商品"

    # Debug: 將 crop 圖片標註距離後儲存
    if debug_folder:
        crop_debug = pil_image.copy()
        draw = ImageDraw.Draw(crop_debug)
        line_height = 14
        y_offset = 4
        for meta, dist in distances:
            name = f"{meta.get('brand','')}{meta.get('flavor','')}"
            marker = " <--" if name == matched_name else ""
            text = f"{name}: {dist:.4f}{marker}"
            bbox = draw.textbbox((4, y_offset), text, font=_debug_font)
            draw.rectangle(bbox, fill="white")
            color = "green" if marker else "red"
            draw.text((4, y_offset), text, fill=color, font=_debug_font)
            y_offset += line_height
        # 也標上 OCR 結果摘要
        ocr_summary = ocr_text.replace("\n", " ")[:50]
        draw.text((4, y_offset + 4), f"OCR: {ocr_summary}", fill="blue", font=_debug_font)
        crop_debug.save(os.path.join(debug_folder, f"crop_{crop_index:02d}.jpg"))

    return matched_name

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
    embedding = clip_model.encode(image).tolist()

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

    # 3. OCR + Fuzzy + CLIP 比對
    detected_names = [match_bottle(img, debug_folder, i) for i, img in enumerate(crops)]
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
