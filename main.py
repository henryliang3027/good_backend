from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from paddleocr import PaddleOCR
import tempfile
import os

app = FastAPI(title="PaddleOCR API")

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)


@app.post("/ocr_inference")
async def ocr_inference(image: UploadFile = File(...)):
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image.filename)[1]) as tmp:
            content = await image.read()
            tmp.write(content)
            tmp_path = tmp.name

        output = ocr.predict(tmp_path)

        texts = []
        for result in output:
            if "rec_texts" in result and result["rec_texts"]:
                texts.extend(result["rec_texts"])

        return JSONResponse(content={"texts": texts})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)