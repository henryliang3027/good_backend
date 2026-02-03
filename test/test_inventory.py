import requests
import base64
import time


def test_inventory_base64(image_path: str, api_url: str = "http://localhost:8888"):
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

    start_time = time.time()
    response = requests.post(
        f"{api_url}/inventory_base64",
        json={"image_base64": image_base64},
    )
    end_time = time.time()
    print(f"Time elapse for {image_path}: {end_time - start_time:.3f} seconds")

    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.json()


if __name__ == "__main__":

    print("=== Testing /ocr_inference_base64 ===")
    image_path = "images/inventory1.jpg"

    test_inventory_base64(image_path)
