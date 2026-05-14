import base64
import requests

IMAGE_PATH = "/home/b40351/Documents/Github/good_backend/date_image/r7.png"
URL = "http://127.0.0.1:8888/glm_ocr_inference_base64"

with open(IMAGE_PATH, "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "image_base64": image_base64,
}

response = requests.post(URL, json=payload)
print(f"status code: {response.status_code}")
print(f"response: {response.json()}")
