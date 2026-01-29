import base64
import os
from mistralai import Mistral

api_key = "h4nh7h78eUj36PNw71DglaocbxHlfg28"

client = Mistral(api_key=api_key)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


image_paths = [
    "images/img_00001.jpg",
    "images/img_00002.jpg",
    "images/img_00003.jpg",
    "images/img_00004.jpg",
    "images/img_00005.jpg",
    "images/img_00006.jpg",
    "images/img_00007.jpg",
    "images/img_00008.jpg",
    "images/img_00009.jpg",
    "images/img_00010.jpg",
]


for image_path in image_paths:
    base64_image = encode_image(image_path)

    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{base64_image}",
        },
        # table_format=None,
        include_image_base64=True,
    )

    page = ocr_response.pages[0]
    print(f"{image_path}, {page.markdown}")
