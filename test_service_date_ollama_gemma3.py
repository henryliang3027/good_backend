import base64
import requests

IMAGE_PATH_LIST = [
    "/home/b40351/Documents/Github/good_backend/date_image/20261028.png",
    # "/home/b40351/Documents/Github/good_backend/date_image/r8.jpg",
    # "/home/b40351/Documents/Github/good_backend/date_image/r9.jpg",
    # "/home/b40351/Documents/Github/good_backend/date_image/r10.jpg",
    # "/home/b40351/Documents/Github/good_backend/date_image/r11.jpg",
    # "/home/b40351/Documents/Github/good_backend/date_image/r12.jpg",
    # "/home/b40351/Documents/Github/good_backend/date_image/r13.jpg",
]

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "glm-ocr:q8_0"
PROMPT = "Text Recognition: "


for image_path in IMAGE_PATH_LIST:
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "images": [image_base64],
        "stream": False,
    }

    print(f"\nImage: {image_path}")
    response = requests.post(OLLAMA_URL, json=payload)
    print(f"status code: {response.status_code}")
    if response.ok:
        result = response.json()
        print(f"date: {result.get('response', '').strip()}")
    else:
        print(f"error: {response.text}")
