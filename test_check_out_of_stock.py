import base64
import requests

IMAGE_PATH = "/home/b40351/Documents/Github/good_backend/test_images/gtest.jpeg"
URL = "http://127.0.0.1:8888/check_out_of_stock"

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
    for shelf in data["data"]:
        position = shelf["position"]
        out_of_stock = shelf["out_of_stock"]
        if out_of_stock:
            print(f"[{position}] 缺貨: {out_of_stock}")
        else:
            print(f"[{position}] 無缺貨")
