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


# check overlap helper function
def check_overlap(l1, r1, l2, r2):     
    # If one rectangle is on left side of other
    if l1[0] > r2[0] or l2[0] > r1[0]:
        return False
 
    # If one rectangle is above other
    if r1[1] < l2[1] or r2[1] < l1[1]:
        return False
 
    return True



n = 0
while True:
    ret, frame = cap.read()
    if ret:
        print(n)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        
        # write fps & frame count
        cv2.putText(img, 'frame: '+str(n+1), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, str(fps)+' fps', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        
        # tank (hardcoded for now)
        tank_lt = (575, 420)
        tank_rb = (980, img.shape[0])
        
        cv2.rectangle(img, (100, 420), (550, img.shape[0]), (0, 255, 0), 5)
        cv2.rectangle(img, tank_lt, tank_rb, (0, 255, 0), 5)
        
        # human detection
        detection = yolo_model(img_pil)
        coords = detection.xyxy[0]

        if len(coords) > 0:
            for j in coords:
                if j[-1] == 0 and j[-2] > 0.5:
                    human_lt = (int(j[0]), int(j[1]))
                    human_rb = (int(j[2]), int(j[3]))
                    
                    cv2.rectangle(img, human_lt, human_rb, (255, 0, 0), 5)
        
        # check overlap & draw arrow & count frame
        counter = 0
        overlap = check_overlap(tank_lt, tank_rb, human_lt, human_rb)
        print(overlap)
        if overlap:
            tank_center = (int((tank_lt[0]+tank_rb[0])/2), int((tank_lt[1]+tank_rb[1])/2))
            human_center = (int((human_lt[0]+human_rb[0])/2), int((human_lt[1]+human_rb[1])/2))
            
            cv2.arrowedLine(img, human_center, tank_center, (255, 165, 0), 5, tipLength=0.2)
            
            # count once per second
            if n // fps == 0:
                counter += 1
                cv2.putText(img, str(counter)+' sec', (human_rb[0]+20, human_lt[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2, cv2.LINE_AA)
                
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