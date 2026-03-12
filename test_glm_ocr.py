import base64
import glob
import os
import ollama


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


def image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def main():
    crops_dir = "my_crops"
    image_paths = glob.glob(os.path.join(crops_dir, "**", "*.jpg"), recursive=True)
    image_paths += glob.glob(os.path.join(crops_dir, "**", "*.png"), recursive=True)
    image_paths.sort()

    print(f"找到 {len(image_paths)} 張圖片\n")

    for image_path in image_paths:
        label = os.path.basename(os.path.dirname(image_path))
        print(f"[{label}] {image_path}")
        b64 = image_to_base64(image_path)
        result = glm_ocr_ollama(b64)
        print(f"  OCR 結果: {result.strip()}")
        print()


if __name__ == "__main__":
    main()
