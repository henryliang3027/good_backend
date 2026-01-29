from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from paddleocr import PaddleOCR
import tempfile
import os
import logging
import base64
from openai import OpenAI

from utils.date_validator import DateValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.root.setLevel(logging.INFO)


app = FastAPI(
    title="PaddleOCR V5 API",
    description="test",
    version="1.0.0",
)

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)


class Base64ImageRequest(BaseModel):
    image_base64: str


SYSTEM_PROMPT = """你是零售貨架的商品統計助手。

輸出格式（CSV）：
商品名稱,顏色,數量

規則：

1. 每個商品一行
2. 使用最短的中文商品名稱
3. 顏色用中文單字,顏色必須是該商品的主要顏色
4. 數量只用整數

範例：
冷山茶王,藍,2
飲冰室茶集,綠,1
可口可樂,紅,3"""


client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="no-key-needed",  # 本地通常不驗證，填任意字串即可
)


def convert_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded_string}"


def inference(base64_image, question):
    response = client.chat.completions.create(
        model="ministral_custom",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": base64_image}},
                ],
            },
        ],
        temperature=0,
    )

    return response.choices[0].message.content


@app.get("/")
async def root():
    return {
        "message": "OCR API",
        "version": "1.0.0",
    }


@app.post("/inventory_base64")
async def inventory_base64(request: Base64ImageRequest):
    try:
        image_data = base64.b64decode(request.image_base64)
        base64_image = f"data:image/jpeg;base64,{request.image_base64}"
        result = inference(base64_image, "請統計圖中的商品")
        return JSONResponse(content={"status": 1, "data": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ocr_inference_base64")
async def ocr_inference_base64(request: Base64ImageRequest):
    tmp_path = None
    try:
        image_data = base64.b64decode(request.image_base64)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(image_data)
            tmp_path = tmp.name
            logging.info(f"Saved base64 image to {tmp_path}")

        output = ocr.predict(tmp_path)
        print("OCR Result:", output)
        for result in output:
            print("OCR rec_texts Result:", output[0]["rec_texts"])
        # output 範例: [{'rec_texts': ['2023/12/31'], 'rec_scores': [0.998]}]
        test = ""
        if len(output[0]["rec_texts"]) == 0:
            return JSONResponse(content={"count": 0, "date": None})
        else:
            text = output[0]["rec_texts"][0]

            result = DateValidator.extract_date(text)
            return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# @app.post("/ocr_inference")
# async def ocr_inference(image: UploadFile = File(...)):
#     if not image.content_type.startswith("image/"):
#         raise HTTPException(status_code=400, detail="File must be an image")

#     tmp_path = None
#     try:
#         with tempfile.NamedTemporaryFile(
#             delete=False, suffix=os.path.splitext(image.filename)[1]
#         ) as tmp:
#             content = await image.read()
#             tmp.write(content)
#             tmp_path = tmp.name
#             logging.info(f"Saved uploaded image to {tmp_path}")

#         output = ocr.predict(tmp_path)
#         texts = []
#         for result in output:
#             if "rec_texts" in result and result["rec_texts"]:
#                 texts.extend(result["rec_texts"])

#         return JSONResponse(content={"texts": texts})

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

#     finally:
#         if tmp_path and os.path.exists(tmp_path):
#             os.unlink(tmp_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8888)
