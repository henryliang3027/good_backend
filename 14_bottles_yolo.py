import os
import base64
import io
import time
from collections import Counter
from contextlib import asynccontextmanager
from datetime import datetime


from PIL import Image, ImageDraw, ImageFont


from ultralytics import YOLO




_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
debug_font = ImageFont.truetype(_FONT_PATH, size=18)




# ========== Model & DB Config ==========
YOLO_MODEL_PATH = "14_bottles_yolo/bottle_detector/best7.pt"
CAP_YOLO_MODEL_PATH = "caps_yolo/cap_detector/best_L_421_20260413.pt"
CAP_CONF_THRESHOLD = 0.90
BOTTLE_CONF_THRESHOLD = 0.80


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


CLASS_COLORS = [
   (0, 204, 255),  #  0 冷山茶王
   (57, 219, 83),  #  1 茶裏王台式綠茶
   (34, 139, 34),  #  2 茶裏王日式無糖綠茶
   (180, 180, 180),  #  3 茶裏王白毫烏龍
   (0, 165, 255),  #  4 茶裏王半熟金萱
   (94, 212, 94),  #  5 原萃台灣青茶
   (139, 69, 19),  #  6 原萃烏龍茶
   (148, 0, 211),  #  7 原萃鐵觀音
   (255, 255, 0),  #  8 無加糖LP33機能優酪乳
   (255, 128, 0),  #  9 御茶園特上檸檬茶
   (50, 205, 50),  # 10 每朝健康双纖綠茶
   (0, 0, 200),  # 11 每朝健康熟藏紅茶
   (255, 20, 147),  # 12 愛之味油切分解茶
   (20, 20, 220),  # 13 濃韻無糖烏龍茶
]




# ========== Global Objects ==========
yolo_model = None
cap_yolo_model = None


def detect_and_label(pil_image: Image.Image) -> tuple[list[str], list[tuple]]:
   """用 bottle YOLO 偵測，回傳 (商品名稱列表, [(cls_id, name, conf, (x1,y1,x2,y2)), ...])。"""
   results = yolo_model(pil_image, conf=BOTTLE_CONF_THRESHOLD, verbose=False)
   detected = []
   bottle_bboxes = []


   for result in results:
       for box in result.boxes:
           cls_id = int(box.cls[0])
           conf = float(box.conf[0])
           name = LABEL_NAMES.get(cls_id, f"未知({cls_id})")
           x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
           detected.append(name)
           bottle_bboxes.append((cls_id, name, conf, (x1, y1, x2, y2)))
           print(f"[YOLO bottle] {name} (cls={cls_id}, conf={conf:.2f})")


   return detected, bottle_bboxes




def inventory():
   start_time = time.time()


   pil_image = Image.open(
       "/home/b40351/Documents/Github/good_backend/test_images/input_20260414_113222_030897.jpg"
   ).convert("RGB")


   # 2. YOLO bottle 偵測
   t0 = time.time()
   detected_names, bottle_bboxes = detect_and_label(pil_image)
   print(
       f"[YOLO bottle] detect={round(time.time()-t0, 3)}s, found={len(detected_names)}"
   )
   if not detected_names:
       return {"status": 1, "data": {}}


   # 3. YOLO cap 偵測
   t0 = time.time()
   cap_results = cap_yolo_model(pil_image, conf=CAP_CONF_THRESHOLD, verbose=False)
   cap_bboxes = []
   for result in cap_results:
       for box in result.boxes:
           cap_conf = float(box.conf[0])
           x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
           cap_bboxes.append((cap_conf, (x1, y1, x2, y2)))
   print(f"[YOLO cap] detect={round(time.time()-t0, 3)}s, found={len(cap_bboxes)}")


   # 3-1. 儲存標註圖與 label 檔
   t0 = time.time()
   timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
   


   overview = pil_image.copy()
   draw = ImageDraw.Draw(overview)
   for cls_id, name, conf, (x1, y1, x2, y2) in bottle_bboxes:
       bgr = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
       color = (bgr[2], bgr[1], bgr[0])
       draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
       draw.text(
           (x1, max(0, y1 - 30)), f"{name} {conf:.2f}", fill=color, font=debug_font
       )
   for cap_conf, (x1, y1, x2, y2) in cap_bboxes:
       draw.rectangle([x1, y1, x2, y2], outline="blue", width=2)
       draw.text(
           (x1, max(0, y1 - 30)), f"cap {cap_conf:.2f}", fill="blue", font=debug_font
       )
   overview.show()



   print(f"image saving time={round(time.time()-t0, 3)}s")




if __name__ == "__main__":


   # 1. 載入自訓練 YOLO 偵測模型
   yolo_model = YOLO(YOLO_MODEL_PATH)
   print(f"✅ YOLO 模型載入完成: {YOLO_MODEL_PATH}")


   cap_yolo_model = YOLO(CAP_YOLO_MODEL_PATH)
   print(f"✅ Cap YOLO 模型載入完成: {CAP_YOLO_MODEL_PATH}")


   inventory()



