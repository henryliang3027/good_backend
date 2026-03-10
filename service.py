import os
import base64
import io
import time
from collections import Counter
from contextlib import asynccontextmanager

import numpy as np
import chromadb
import ollama
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
from ultralytics import YOLO
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import subprocess

# ========== Model & DB Config ==========
BOTTLE_CLASS_ID = 39
OLLAMA_MODEL = "ministral-3:3b"
CONF_THRESHOLD = 0.5

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
    )
    # 等待 llama-server 啟動
    time.sleep(5)
    print(f"llama-server started with PID: {llama_process.pid}")


def stop_llama_server():
    global llama_process
    if llama_process:
        print(f"Stopping llama-server (PID: {llama_process.pid})...")
        llama_process.terminate()
        try:
            llama_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            llama_process.kill()
        print("llama-server stopped.")
        llama_process = None


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

# ========== Helper Functions ==========

def detect_and_crop_bottles(pil_image: Image.Image):
    results = yolo_model(pil_image, conf=CONF_THRESHOLD, verbose=False)
    cropped_images = []
    for result in results:
        for box in result.boxes:
            if int(box.cls[0]) == BOTTLE_CLASS_ID:
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                cropped_images.append(pil_image.crop((x1, y1, x2, y2)))
    return cropped_images

def match_with_chroma(pil_image: Image.Image):
    """取代原本的 .pt 比對，改用 ChromaDB 查詢"""
    img_emb = clip_model.encode(pil_image).tolist()
    
    results = collection.query(
        query_embeddings=[img_emb],
        n_results=1,
        include=["metadatas", "distances"]
    )

    print(f"result: {results}")
    
    if not results['ids'][0]:
        return "未知商品"
    
    dist = results['distances'][0][0]
    metadata = results['metadatas'][0][0]
    
    return f"{metadata['color']}{metadata['display_name']}"

# ========== CRUD Endpoints (管理資料庫) ==========

@app.post("/db/add", summary="[CRUD] 新增飲料特徵到資料庫")
async def add_to_db(
    name: str = Form(...),
    color: str = Form(""),
    file: UploadFile = File(...)
):
    """上傳一張 crop 好的瓶子，存入 ChromaDB"""
    image = Image.open(file.file).convert("RGB")
    embedding = clip_model.encode(image).tolist()
    
    collection.upsert(
        ids=[name], # 以品名作為唯一 ID
        embeddings=[embedding],
        metadatas=[{"display_name": name, "color": color}]
    )
    return {"status": "success", "message": f"已存入: {color}{name}"}

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
    crops = detect_and_crop_bottles(pil_image)
    if not crops:
        return {"status": 1, "data": "貨架上看起來沒有瓶子。"}

    # 3. ChromaDB 向量比對
    detected_names = [match_with_chroma(img) for img in crops]
    counts = dict(Counter(detected_names))
    
    # 4. 組合成文字給 Ollama
    scan_list_str = "\n".join([f"- {k}: {v} 瓶" for k, v in counts.items()])
    print(f"=====SYSTEM_PROMPT=====")
    print(f"{SYSTEM_PROMPT_TEMPLATE.format(scan_list=scan_list_str)}")
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




@app.post("/glm_ocr_inference_base64")
async def glm_ocr_inference_base64(request: Base64ImageRequest):
    tmp_path = None
    try:
        output = glm_ocr_ollama(request.image_base64)
        print("OCR Result:", output)

        elements = output.split("\n")

        # output 範例: [{'rec_texts': ['2023/12/31'], 'rec_scores': [0.998]}]
        test = ""
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

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8888)
