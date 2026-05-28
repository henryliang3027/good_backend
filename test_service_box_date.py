import base64
import requests

URL = "http://127.0.0.1:8888/box_date_detection"

IMAGE_PATH_LIST = [
    "/home/b40351/Documents/Github/good_backend/box_date_image/box0.jpg",
]


def test_image(image_path: str):
    print(f"\n[IMAGE] {image_path}")

    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

    response = requests.post(URL, json={"image_base64": image_base64})
    print(f"[STATUS] {response.status_code}")

    if response.status_code != 200:
        print(f"[ERROR] {response.text}")
        return

    data = response.json()
    if data.get("status") != "1":
        print(f"[FAIL] {data}")
        return

    results = data["data"]
    print(f"[BOXES] 偵測到 {len(results)} 個箱子")
    for i, item in enumerate(results):
        print(f"  [{i+1}] name={item['name']}  date={item['date']}")


if __name__ == "__main__":
    for path in IMAGE_PATH_LIST:
        test_image(path)
