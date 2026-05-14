import base64
import requests

IMAGE_PATH = "/home/b40351/Documents/Github/good_backend/test_images/g_box_test.jpeg"
URL = "http://127.0.0.1:8888/box_detection"

with open(IMAGE_PATH, "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "image_base64": image_base64,
}

response = requests.post(URL, json=payload)
print(f"status code: {response.status_code}")

data = response.json()
print(f"response: {data}")

if response.status_code == 200 and data.get("status") == "1":
    number = data["data"]
    print(f"偵測到 {number} 個 box")
