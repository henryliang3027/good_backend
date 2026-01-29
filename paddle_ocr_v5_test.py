import time
from paddleocr import PaddleOCR, PaddleOCRVL
import re

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

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)

for image_path in image_paths:
    # calculate time elapse

    start_time = time.time()
    output = ocr.predict(image_path)
    end_time = time.time()
    print(f"Time elapse for {image_path}: {end_time - start_time:.3f} seconds")
    for result in output:
        print(result["rec_texts"][0])
