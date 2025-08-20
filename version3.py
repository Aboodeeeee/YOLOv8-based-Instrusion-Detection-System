import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np

model = YOLO('best_old.pt')

def Surveillance(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('Surveillance')
cv2.setMouseCallback('Surveillance', Surveillance)

cap = cv2.VideoCapture('clip3.mp4')  # Adjust the path to your video file

my_file = open("coco1.txt", "r")
data = my_file.read()
class_list = data.split("\n")

# Areas of interest
count = 0
area1 = [(772, 73), (683, 54), (506, 99), (629, 144)]
area2 = [(683, 54), (592, 39), (431, 71), (506, 99)]

while True:    
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    count += 1
    if count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    list1 = []
    list2 = []
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        cx = int(x1 + x2) // 2
        cy = int(y1 + y2) // 2
        w, h = x2 - x1, y2 - y1
        
        # Check if the person is in area1
        result = cv2.pointPolygonTest(np.array(area1, np.int32), (cx, cy), False)
        if result >= 0:
            list1.append((x1, y1, w, h, cx, cy))

        # Check if the person is in area2
        result1 = cv2.pointPolygonTest(np.array(area2, np.int32), (cx, cy), False)
        if result1 >= 0:
            list2.append((x1, y1, w, h, cx, cy))

    cr1 = len(list1)
    cr2 = len(list2)
    
    # Highlight people in area1
    if cr1 >= 2:
        for person in list1:
            x1, y1, w, h, cx, cy = person
            cvzone.cornerRect(frame, (x1, y1, w, h), 3, 2, colorR=(0, 0, 255))  # Red for intruder
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
    else:
        for person in list1:
            x1, y1, w, h, cx, cy = person
            cvzone.cornerRect(frame, (x1, y1, w, h), 3, 2, colorR=(0, 255, 0))  # Green for 1 or less people
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

    # Highlight people in area2
    if cr2 >= 2:
        for person in list2:
            x1, y1, w, h, cx, cy = person
            cvzone.cornerRect(frame, (x1, y1, w, h), 3, 2, colorR=(0, 0, 255))  # Red for intruder
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
    else:
        for person in list2:
            x1, y1, w, h, cx, cy = person
            cvzone.cornerRect(frame, (x1, y1, w, h), 3, 2, colorR=(0, 255, 0))  # Green for 1 or less people
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

    # Highlight people outside both areas in purple
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        cx = int(x1 + x2) // 2
        cy = int(y1 + y2) // 2
        w, h = x2 - x1, y2 - y1
        
        result = cv2.pointPolygonTest(np.array(area1, np.int32), (cx, cy), False)
        result1 = cv2.pointPolygonTest(np.array(area2, np.int32), (cx, cy), False)
        
        if result < 0 and result1 < 0:
            cvzone.cornerRect(frame, (x1, y1, w, h), 3, 2, colorR=(255, 0, 255))  # Purple for outside
            cv2.circle(frame, (cx, cy), 4, (255, 0, 255), -1)

    # Draw areas of interest
    color1 = (0, 0, 255) if cr1 >= 2 else (0, 255, 0)
    color2 = (0, 0, 255) if cr2 >= 2 else (0, 255, 0)
    cv2.polylines(frame, [np.array(area1, np.int32)], True, color1, 2)
    cv2.polylines(frame, [np.array(area2, np.int32)], True, color2, 2)
    
    # Display counter information
   # cvzone.putTextRect(frame, f'counter1: {cr1}', (50, 60), 2, 2)
  #  cvzone.putTextRect(frame, f'counter2: {cr2}', (50, 160), 2, 2)

    cv2.imshow("Surveillance", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
