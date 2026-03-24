import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
_font = ImageFont.truetype(_FONT_PATH, size=18)


def draw_label(frame, text, x1, y1, color_bgr):
    """用 PIL 渲染中文 label，背景填 color，文字白色，貼回 OpenCV frame。"""
    # BGR -> RGB
    color_rgb = color_bgr[::-1]
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    bbox = draw.textbbox((0, 0), text, font=_font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    lx, ly = x1, max(0, y1 - th - 6)

    draw.rectangle([lx, ly, lx + tw + 6, ly + th + 6], fill=color_rgb)
    draw.text((lx + 3, ly + 3), text, font=_font, fill=(0, 0, 0))

    frame[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

MODEL_PATH = "14_bottles_yolo/bottle_detector/best105.pt"
CAP_MODEL_PATH = "caps_yolo/cap_detector/best596.pt"
VIDEO_PATH = "14_bottles_yolo/20260323_103316.mp4"
BOTTLE_CONF_THRESHOLD = 0.65
CAP_CONF_THRESHOLD = 0.5

LABEL_NAMES = {
    0: "冷山茶王", 
    1: "茶裏王台式綠茶", 
    2: "茶裏王日式無糖綠茶", 
    3: "茶裏王白毫烏龍", 
    4: "茶裏王半熟金萱", 
    5: "原萃台灣青茶", 
    6: "原萃烏龍茶",
    7: "原萃鐵觀音", 
    8: "無加糖LP33機能優酪乳", 
    9: "御茶園特上檸檬茶", 
    10: "每朝健康双纖綠茶", 
    11: "每朝健康熟藏紅茶", 
    12: "愛之味油切分解茶四季春風味", 
    13: "濃韻無糖烏龍茶",
}

CAP_LABEL_NAME = {
    0: "cap",
}

CAP_CLASS_COLOR = (255,255,128) # cap

# BGR colors, one per class (0–13)
CLASS_COLORS = [
    (  0, 204, 255),  #  0 冷山茶王             - 黃
    ( 57, 219,  83),  #  1 茶裏王台式綠茶        - 綠
    ( 34, 139,  34),  #  2 茶裏王日式無糖綠茶    - 深綠
    (180, 180, 180),  #  3 茶裏王白毫烏龍        - 灰
    (  0, 165, 255),  #  4 茶裏王半熟金萱        - 橙
    ( 94, 212,  94),  #  5 原萃台灣青茶          - 青綠
    (139,  69,  19),  #  6 原萃烏龍茶            - 棕
    (148,   0, 211),  #  7 原萃鐵觀音            - 紫
    (255, 255,   0),  #  8 無加糖LP33機能優酪乳  - 青
    (255, 128,   0),  #  9 御茶園特上檸檬茶      - 藍
    ( 50, 205,  50),  # 10 每朝健康双纖綠茶      - 草綠
    (  0,   0, 200),  # 11 每朝健康熟藏紅茶      - 紅
    (255,  20, 147),  # 12 愛之味油切分解茶      - 粉
    ( 20,  20, 220),  # 13 濃韻無糖烏龍茶        - 深紅
]

model = YOLO(MODEL_PATH)
cap_yolo_model = YOLO(CAP_MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=BOTTLE_CONF_THRESHOLD, verbose=False)

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            label = f"{LABEL_NAMES.get(cls_id, str(cls_id))} {conf:.2f}"
            color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            draw_label(frame, label, x1, y1, color)


    cap_results = cap_yolo_model(frame, conf=CAP_CONF_THRESHOLD, verbose=False)
    for result in cap_results:
        for box in result.boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            label = f"Cap {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), CAP_CLASS_COLOR, 2)
            draw_label(frame, label, x1, y1, CAP_CLASS_COLOR)



    cv2.imshow("14 Bottles YOLO", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
