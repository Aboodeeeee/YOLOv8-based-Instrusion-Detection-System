import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np

# Load the YOLO model
model = YOLO('best.pt')
# model = YOLO('best_re_final.pt')
# model = YOLO('best_old.pt')

# Function to handle mouse events (for debugging coordinates)
def Surveillance(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

# Create window and set mouse callback
cv2.namedWindow('Surveillance')
cv2.setMouseCallback('Surveillance', Surveillance)

# Open video capture
cap = cv2.VideoCapture('clip1.mp4')  # Adjust the path to your video file

# Read class labels
my_file = open("coco1.txt", "r")
data = my_file.read()
class_list = data.split("\n")

# Define areas of interest
area1 = [(772, 73), (683, 54), (506, 99), (629, 144)]
area2 = [(683, 54), (592, 39), (431, 71), (506, 99)]

# Initialize count variable
count = 0

while True:    
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # Process every 3rd frame to reduce computational load
    count += 1
    if count % 3 != 0:
        continue

    # Resize frame
    frame = cv2.resize(frame, (1020, 500))
    
    # Detect objects
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    # Lists to store people in each area
    list1 = []
    list2 = []

    # Process each detected object
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
        
        # Check if the object is in area1
        result = cv2.pointPolygonTest(np.array(area1, np.int32), (cx, cy), False)
        if result >= 0:
            list1.append((x1, y1, w, h, cx, cy))

        # Check if the object is in area2
        result1 = cv2.pointPolygonTest(np.array(area2, np.int32), (cx, cy), False)
        if result1 >= 0:
            list2.append((x1, y1, w, h, cx, cy))

    # Count people in each area
    cr1 = len(list1)
    cr2 = len(list2)
    
    # Process objects in area1
    for person in list1:
        x1, y1, w, h, cx, cy = person
        label = 'Intruder' if cr1 >= 2 else 'Person'
        color = (0, 0, 255) if label == 'Intruder' else (0, 255, 0)
        
        # Draw bounding box with label
        cvzone.cornerRect(frame, (x1, y1, w, h), 3, 2, colorR=color)
        cvzone.putTextRect(frame, label, (x1, y1-10), 
                           scale=1, 
                           thickness=1, 
                           colorR=color, 
                           colorT=(255,255,255))
        cv2.circle(frame, (cx, cy), 4, color, -1)

    # Process objects in area2
    for person in list2:
        x1, y1, w, h, cx, cy = person
        label = 'Intruder' if cr2 >= 2 else 'Person'
        color = (0, 0, 255) if label == 'Intruder' else (0, 255, 0)
        
        # Draw bounding box with label
        cvzone.cornerRect(frame, (x1, y1, w, h), 3, 2, colorR=color)
        cvzone.putTextRect(frame, label, (x1, y1-10), 
                           scale=1, 
                           thickness=1, 
                           colorR=color, 
                           colorT=(255,255,255))
        cv2.circle(frame, (cx, cy), 4, color, -1)

    # Draw areas of interest
    color1 = (0, 0, 255) if cr1 >= 2 else (0, 255, 0)
    color2 = (0, 0, 255) if cr2 >= 2 else (0, 255, 0)
    cv2.polylines(frame, [np.array(area1, np.int32)], True, color1, 2)
    cv2.polylines(frame, [np.array(area2, np.int32)], True, color2, 2)

    # Display the frame
    cv2.imshow("Surveillance", frame)
    
    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Clean up
cap.release()
cv2.destroyAllWindows()