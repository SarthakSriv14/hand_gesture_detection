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