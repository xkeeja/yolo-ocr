import cv2
import torch
from PIL import Image
import easyocr


cap = cv2.VideoCapture("data/clipped.MP4")
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"frames: {frame_count}")

writer = cv2.VideoWriter("data/output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

yolo_model = torch.hub.load("ultralytics/yolov5", "yolov5x6")
reader = easyocr.Reader(['en'], gpu=True)


n = 1
while True:
    ret, frame = cap.read()
    if ret:
        print(n)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        
        # tank
        cv2.rectangle(img, (100, 420), (550, img.shape[0]), (0, 255, 0), 5)
        cv2.rectangle(img, (575, 420), (980, img.shape[0]), (0, 255, 0), 5)
        
        # human detection
        detection = yolo_model(img_pil)
        coords = detection.xyxy[0]

        if len(coords) > 0:
            for j in coords:
                if j[-1] == 0 and j[-2] > 0.5:
                    l = int(j[0])
                    r = int(j[2])
                    t = int(j[1])
                    b = int(j[3])
                    
                    cv2.rectangle(img, (l, t), (r, b), (255, 0, 0), 5)
        
        # ocr detection
        d = reader.readtext(img, min_size=5, text_threshold=0.3, mag_ratio=2, allowlist='0123456789')
        
        if len(d) > 0:
            for i in d:
                if i[-1] > 0.8:
                    l = int(i[0][0][0]) - 10
                    r = int(i[0][2][0]) + 10
                    t = int(i[0][0][1]) - 10
                    b = int(i[0][2][1]) + 10
                    
                    cv2.rectangle(img, (l, t), (r, b), (0, 0, 255), 2)
                    cv2.putText(img, i[1], (l, t - 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
        
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        writer.write(img)
        n += 1
    else:
        break

writer.release()