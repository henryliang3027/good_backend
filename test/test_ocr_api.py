import requests
import base64

image_paths = [
    "images/empty.jpg",
    # "images/img_00001.jpg",
    # "images/img_00002.jpg",
    # "images/img_00003.jpg",
    # "images/img_00004.jpg",
    # "images/img_00005.jpg",
    # "images/img_00006.jpg",
    # "images/img_00007.jpg",
    # "images/img_00008.jpg",
    # "images/img_00009.jpg",
    # "images/img_00010.jpg",
]


def test_ocr_inference_base64(image_path: str, api_url: str = "http://localhost:8888"):
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

    response = requests.post(
        f"{api_url}/ocr_inference_base64",
        json={"image_base64": image_base64},
    )

    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.json()


def test_ocr_inference_upload(image_path: str, api_url: str = "http://localhost:8888"):
    with open(image_path, "rb") as f:
        files = {"image": (image_path, f, "image/jpeg")}
        response = requests.post(f"{api_url}/ocr_inference", files=files)

    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.json()


if __name__ == "__main__":

    print("=== Testing /ocr_inference_base64 ===")

    for image_path in image_paths:
        test_ocr_inference_base64(image_path)

    # print("\n=== Testing /ocr_inference (upload) ===")
    # test_ocr_inference_upload(image_path)
