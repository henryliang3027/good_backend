
import base64
import os
import ollama

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")





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



if __name__ == "__main__":
    image_path = "images/img_00001.jpg"
    base64_image = encode_image(image_path)
    ocr_result = glm_ocr_ollama(base64_image)
    print(f"OCR Result for {image_path}:\n{ocr_result}")