import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
_font = ImageFont.truetype(_FONT_PATH, size=18)


def draw_label(frame, text, x1, y1, color_bgr):
    """用 PIL 渲染 label，背景填 color，文字黑色，貼回 OpenCV frame。"""
    color_rgb = color_bgr[::-1]
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    bbox = draw.textbbox((0, 0), text, font=_font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    lx, ly = x1, max(0, y1 - th - 6)

    draw.rectangle([lx, ly, lx + tw + 6, ly + th + 6], fill=color_rgb)
    draw.text((lx + 3, ly + 3), text, font=_font, fill=(0, 0, 0))

    frame[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


MODEL_PATH = "/home/b40351/Documents/Github/good_backend/caps_yolo/cap_detector/best458.pt"
IMAGE_PATH = "/home/b40351/Documents/Github/good_backend/label_images/input_20260413_171722_013004.jpg"
CONF_THRESHOLD = 0.90

LABEL_NAMES = {
    0: "cap",
}

CLASS_COLORS = [
    (0, 200, 255),  # 0 cap - 黃
]

model = YOLO(MODEL_PATH)
frame = cv2.imread(IMAGE_PATH)

results = model(frame, conf=CONF_THRESHOLD, verbose=False)

for result in results:
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        label = f"{LABEL_NAMES.get(cls_id, str(cls_id))} {conf:.2f}"
        color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        draw_label(frame, label, x1, y1, color)

cv2.imshow("Caps YOLO", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
