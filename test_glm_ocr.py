import base64
import os
import ollama

IMAGE_DIR = "/home/b40351/Documents/Github/good_backend/cropped_date_image"


def glm_ocr_ollama(base64_image: str) -> str:
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


def test_image(image_path: str):
    print(f"\n[IMAGE] {image_path}")
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

    result = glm_ocr_ollama(image_base64)
    print(f"[OCR]   {result!r}")


if __name__ == "__main__":
    images = sorted(
        os.path.join(IMAGE_DIR, f)
        for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    if not images:
        print(f"No images found in {IMAGE_DIR}")
    else:
        for path in images:
            test_image(path)
