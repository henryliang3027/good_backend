import base64
import io
import os
import signal
import subprocess
import time

from openai import OpenAI
from PIL import Image

IMAGE_DIR = "/home/b40351/Documents/Github/good_backend/cropped_boxes"

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

glm_ocr_client = OpenAI(
    base_url="http://127.0.0.1:8882/v1",
    api_key="no-key-needed",
)


def start_glm_ocr_server() -> subprocess.Popen:
    print("Starting llama-server (glm-ocr)...")
    proc = subprocess.Popen(
        GLM_OCR_SERVER_CMD,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    time.sleep(5)
    print(f"llama-server (glm-ocr) started with PID: {proc.pid}")
    return proc


def stop_glm_ocr_server(proc: subprocess.Popen | None) -> None:
    if proc:
        print(f"Stopping llama-server (glm-ocr) (PID: {proc.pid})...")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=10)
        except (subprocess.TimeoutExpired, ProcessLookupError):
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        print("llama-server (glm-ocr) stopped.")


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


def test_image(image_path: str):
    print(f"\n[IMAGE] {image_path}")
    with Image.open(image_path) as img:
        print(f"[SIZE]  {img.width}x{img.height}")
        if img.width == 314:
            img = img.resize((img.width * 2, img.height * 2))
            print(f"[RESIZE] {img.width}x{img.height}")
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=95)
        image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    start = time.perf_counter()
    result = glm_ocr_llama(image_base64)
    elapsed = time.perf_counter() - start

    print(f"[OCR]   {result!r}")
    print(f"[TIME]  {elapsed:.3f}s")
    return elapsed


if __name__ == "__main__":
    images = sorted(
        os.path.join(IMAGE_DIR, f)
        for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    if not images:
        print(f"No images found in {IMAGE_DIR}")
    else:
        glm_ocr_process = start_glm_ocr_server()
        try:
            total_start = time.perf_counter()
            elapsed_times = [test_image(path) for path in images]
            total_elapsed = time.perf_counter() - total_start

            print(f"\n[TOTAL] {total_elapsed:.3f}s for {len(images)} image(s)")
            print(f"[AVG]   {sum(elapsed_times) / len(elapsed_times):.3f}s/image")
        finally:
            stop_glm_ocr_server(glm_ocr_process)
