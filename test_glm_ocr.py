"""
GLM-OCR inference test via local llama.cpp server (OpenAI-compatible API).

Usage:
    pip install fastapi uvicorn openai python-multipart pillow httpx
    python test_glm_ocr.py

    # Test via Swagger: http://localhost:8001/docs
"""

import base64
import io
import os
import subprocess
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel

# ── Paths (relative to repo root) ────────────────────────────────────────────
_REPO_ROOT   = os.path.abspath(os.path.dirname(__file__))
LLAMA_SERVER = os.getenv("LLAMA_SERVER",  os.path.join(_REPO_ROOT, "llama-b1287", "llama-server"))
MODEL_PATH   = os.getenv("MODEL_PATH",    os.path.join(_REPO_ROOT, "models", "glm-ocr", "GLM-OCR-Q8_0.gguf"))
MMPROJ_PATH  = os.getenv("MMPROJ_PATH",   os.path.join(_REPO_ROOT, "models", "glm-ocr", "mmproj-GLM-OCR-Q8_0.gguf"))
LIB_PATH     = os.getenv("LD_LIBRARY_PATH", os.path.join(_REPO_ROOT, "llama-b1287"))

LLAMA_PORT     = int(os.getenv("LLAMA_PORT", "8000"))
LLAMA_CPP_URL  = os.getenv("LLAMA_CPP_URL", f"http://localhost:{LLAMA_PORT}/v1")
OCR_PROMPT     = os.getenv("OCR_PROMPT", "Text Recognition:")
N_GPU_LAYERS   = os.getenv("N_GPU_LAYERS", "-1")
CTX_SIZE       = os.getenv("CTX_SIZE", "4096")

_server_proc: subprocess.Popen | None = None


def _start_llama_server() -> subprocess.Popen:
    cmd = [
        LLAMA_SERVER,
        "-m",        MODEL_PATH,
        "--mmproj",  MMPROJ_PATH,
        "--host",    "0.0.0.0",
        "--port",    str(LLAMA_PORT),
        "--ctx-size", CTX_SIZE,
        "-ngl",      N_GPU_LAYERS,
    ]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = LIB_PATH + ":" + env.get("LD_LIBRARY_PATH", "")
    print(f"[llama-server] starting: {' '.join(cmd)}")
    return subprocess.Popen(cmd, env=env)


def _wait_for_server(timeout: int = 120) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(f"http://localhost:{LLAMA_PORT}/health", timeout=2)
            if r.status_code == 200:
                print("[llama-server] ready")
                return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"llama-server did not become ready within {timeout}s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _server_proc
    _server_proc = _start_llama_server()
    try:
        _wait_for_server()
    except RuntimeError as e:
        _server_proc.terminate()
        raise e
    yield
    print("[llama-server] shutting down")
    _server_proc.terminate()
    _server_proc.wait()


app = FastAPI(
    title="GLM-OCR Test",
    description="GLM-OCR inference via local llama.cpp server",
    lifespan=lifespan,
)

_client = OpenAI(base_url=LLAMA_CPP_URL, api_key="no-key-needed")


def _image_to_base64(image: Image.Image, fmt: str = "JPEG") -> str:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def _call_glm_ocr(b64_image: str, prompt: str, mime: str = "image/jpeg") -> str:
    resp = _client.chat.completions.create(
        model="glm-ocr",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64_image}"}},
                ],
            }
        ],
        temperature=0,
        max_tokens=2048,
    )
    return resp.choices[0].message.content


# ── Request / Response models ─────────────────────────────────────────────────

class OcrBase64Request(BaseModel):
    image_base64: str
    prompt: str = OCR_PROMPT


class OcrResponse(BaseModel):
    result: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Check if llama.cpp server is reachable."""
    try:
        models = _client.models.list()
        return {"status": "ok", "models": [m.id for m in models.data]}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"llama.cpp server unreachable: {e}")


@app.post("/ocr/upload", response_model=OcrResponse)
async def ocr_upload(
    file: UploadFile = File(..., description="Image file to OCR"),
    prompt: str = Form(default=OCR_PROMPT),
):
    """OCR an uploaded image file."""
    try:
        data = await file.read()
        img  = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    b64 = _image_to_base64(img)
    result = _call_glm_ocr(b64, prompt)
    return OcrResponse(result=result)


@app.post("/ocr/base64", response_model=OcrResponse)
async def ocr_base64(request: OcrBase64Request):
    """OCR a base64-encoded image."""
    try:
        data = base64.b64decode(request.image_base64)
        img  = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image")

    b64 = _image_to_base64(img)
    result = _call_glm_ocr(b64, request.prompt)
    return OcrResponse(result=result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
