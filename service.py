import os
import base64
import io
import time
from collections import Counter
from contextlib import asynccontextmanager
import signal
import chromadb
import ollama
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from ultralytics import YOLO
from openai import OpenAI

import subprocess
from nicegui import ui
from utils.date_validator import DateValidator
from dependencies import set_collection
import routes.admin  # noqa: F401 — registers @ui.page('/admin')

# ========== Model & DB Config ==========
YOLO_MODEL_PATH = "14_bottles_yolo/bottle_detector/best.pt"
CONF_THRESHOLD = 0.5

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


class Base64ImageRequest(BaseModel):
    image_base64: str
    question: str = "請統計商品"

# ========== Global Objects ==========
yolo_model = None
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
    global yolo_model, chroma_client, collection
    print("🚀 正在啟動系統並載入模型...")

    # 1. 載入自訓練 YOLO 偵測模型
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print(f"✅ YOLO 模型載入完成: {YOLO_MODEL_PATH}")

    # 2. 初始化 ChromaDB（供 /db/* CRUD 端點使用）
    chroma_client = chromadb.PersistentClient(path="./drink_vector_db")
    collection = chroma_client.get_or_create_collection(name="drink_catalog")
    set_collection(collection)
    print(f"📦 ChromaDB 已就緒，目前資料庫包含 {collection.count()} 筆資料。")

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

def detect_and_label(pil_image: Image.Image) -> list[str]:
    """用 best.pt 偵測，直接回傳每個 bbox 對應的商品名稱列表。"""
    results = yolo_model(pil_image, conf=CONF_THRESHOLD, verbose=False)
    detected = []
    boxes_info = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = LABEL_NAMES.get(cls_id, f"未知({cls_id})")
            detected.append(name)
            boxes_info.append((int(v) for v in box.xyxy[0].tolist()) )
            print(f"[YOLO] {name} (cls={cls_id}, conf={conf:.2f})")

    # Debug: 儲存標註圖
    # if boxes_info:
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    #     debug_folder = os.path.join(DEBUG_DIR, timestamp)
    #     os.makedirs(debug_folder, exist_ok=True)
    #     pil_image.save(os.path.join(debug_folder, "input.jpg"))

    #     overview = pil_image.copy()
    #     draw = ImageDraw.Draw(overview)
    #     for result in results:
    #         for box in result.boxes:
    #             cls_id = int(box.cls[0])
    #             conf = float(box.conf[0])
    #             x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
    #             name = LABEL_NAMES.get(cls_id, f"未知({cls_id})")
    #             draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    #             draw.text((x1, max(0, y1 - 15)), f"{name} {conf:.2f}", fill="red", font=_debug_font)
    #     overview.save(os.path.join(debug_folder, "overview.jpg"))
    #     print(f"[DEBUG] debug 資料夾: {debug_folder}")

    return detected

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
    embedding = cnn_encoder.encode(image)

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

    # 2. YOLO 偵測 + 直接取得商品名稱
    detected_names = detect_and_label(pil_image)
    if not detected_names:
        return {"status": 1, "data": "貨架上看起來沒有瓶子。"}

    counts = dict(Counter(detected_names))
    
    # 4. 組合成文字給 llama.cpp
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

    ui.run_with(app, title="Good API", favicon="🍵", dark=False)
    uvicorn.run(app, host="0.0.0.0", port=8888)
