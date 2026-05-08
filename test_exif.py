import base64
import io
from PIL import Image

IMAGE_PATH = "/home/b40351/Documents/Github/good_backend/test_images/gtest.jpeg"

with open(IMAGE_PATH, "rb") as f:
    raw_bytes = f.read()

original = Image.open(IMAGE_PATH)
original_orientation = original.getexif().get(274)
print(f"Original EXIF orientation: {original_orientation}")
print(f"Original size: {original.size}")

image_base64 = base64.b64encode(raw_bytes).decode("utf-8")
decoded_bytes = base64.b64decode(image_base64)
decoded = Image.open(io.BytesIO(decoded_bytes))
decoded_orientation = decoded.getexif().get(274)
print(f"After base64 round-trip EXIF orientation: {decoded_orientation}")
print(f"After base64 round-trip size: {decoded.size}")

print(f"\nEXIF preserved: {original_orientation == decoded_orientation}")
