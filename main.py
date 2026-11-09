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
    base_options=base_options
)

detector = vision.HandLandmarker.create_from_options(options)

#Webcam
cap = cv.VideoCapture(0)

pinching = False

screen_width, screen_height = pyautogui.size()

prev_x = 0
prev_y = 0
alpha = 0.4

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

        thumb_tip = hand[4]
        index_tip = hand[8]

        distance = math.sqrt(
            (thumb_tip.x - index_tip.x)**2 +
            (thumb_tip.y - index_tip.y)**2
        )

        target_x = int((1 - index_tip.x) * screen_width)
        target_y = int(index_tip.y * screen_height)

        smooth_x = prev_x + alpha * (target_x - prev_x)
        smooth_y = prev_y + alpha * (target_y - prev_y)

        pyautogui.moveTo(smooth_x,smooth_y)

        prev_x = smooth_x
        prev_y = smooth_y

        if distance <0.07:
            current_pinch = True
        else:
            current_pinch = False

        if current_pinch and not pinching:
            print("Pinch Started")
            pyautogui.click()

        pinching = current_pinch

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