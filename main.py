import math

import pyautogui
import cv2 as cv
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

#Creating detector
base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands = 1
)

detector = vision.HandLandmarker.create_from_options(options)

#Webcam
cap = cv.VideoCapture(0)

pinching = False

screen_width, screen_height = pyautogui.size()

prev_x = 0
prev_y = 0
alpha = 0.4

control_left = 0.2
control_right = 0.8
control_top = 0.2
control_bottom = 0.8

while True:
    ret, frame = cap.read()

    frame_rgb = cv.cvtColor(frame,cv.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )

    result = detector.detect(mp_image)
    #print(result)

    if result.hand_landmarks:
        hand = result.hand_landmarks[0] #hand now has 21 handlandmarks

        index_tip = hand[8]
        thumb_tip = hand[4]

        distance = math.sqrt(
            (thumb_tip.x - index_tip.x)**2 +
            (thumb_tip.y - index_tip.y)**2
        )

        if distance <0.07:
            current_pinch = True
        else:
            current_pinch = False

        if current_pinch and not pinching:
            #print("Pinch Started")
            pyautogui.click()

        pinching = current_pinch

        if not current_pinch:

            #normalizing the coordinates to 60% of the original area they cover
            x = (index_tip.x - control_left) / (control_right - control_left)
            y = (index_tip.y - control_top) / (control_bottom - control_top)

            # if finger goes slightly outside the control area, we restrict the area,
            # x and y can never be less than 0 or greater than 1
            x = max(0, min(1, x))
            y = max(0, min(1, y))

            target_x = int((1 - x) * screen_width)
            target_y = int(y * screen_height)

            smooth_x = prev_x + alpha * (target_x - prev_x)
            smooth_y = prev_y + alpha * (target_y - prev_y)

            pyautogui.moveTo(smooth_x,smooth_y)

            prev_x = smooth_x
            prev_y = smooth_y

        for landmark in hand:

            x = int(landmark.x * frame.shape[1]) #horizontal x width
            y = int(landmark.y * frame.shape[0]) #vertical x height

            cv.circle(frame,(x,y),10,(255,0,0),-1)

            for connection in vision.HandLandmarksConnections.HAND_CONNECTIONS:
                start = hand[connection.start]
                end = hand[connection.end]

                x1 = int(start.x * frame.shape[1])
                y1 = int(start.y * frame.shape[0])

                x2 = int(end.x * frame.shape[1])
                y2 = int(end.y * frame.shape[0])

                cv.line(frame, (x1, y1), (x2, y2), (255,0,0), 2) 

    cv.imshow('camera',frame)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()