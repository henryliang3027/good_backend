import base64
import requests

IMAGE_PATH = "label_images/input_20260414_090657_286494.jpg"
URL = "http://127.0.0.1:8888/inventory_base64"

with open(IMAGE_PATH, "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "image_base64": image_base64,
    "question": "統計商品",
}

response = requests.post(URL, json=payload)
print(f"status code: {response.status_code}")
print(f"response: {response.json()}")
