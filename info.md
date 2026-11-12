HAND LANDMARK - https://developers.google.com/edge

The Hand Landmarker uses a model bundle with two packaged models: a palm detection model and a hand landmarks detection model. We need a model bundle that contains both these models to run this task.

The palm detection model locates hands within the input image, and the hand landmarks detection model identifies specific hand landmarks on the cropped hand image defined by the palm detection model.

The hand landmark model bundle detects the keypoint localization of 21 hand-knuckle coordinates within the detected hand regions. The model was trained on approximately 30K real-world images, as well as several rendered synthetic hand models imposed over various backgrounds.

Since running the palm detection model is time consuming, Hand Landmarker only re-triggers the palm detection model if the hand landmarks model no longer identifies the presence of hands or fails to track the hands within the frame
 

    hand-landmarker model
            ↓
    create the detector
            ↓
        detector


             ┌-────────────────┐
image ──────→│    detector     │
             └─────────────────┘
                      ↓
                 hand landmarks

BaseOptions → where is the model?
HandLandmarkerOptions → how should we configure the detector?
HandLandmarker.create_from_options() → create the detector

          distance < threshold?
                  │
          ┌───────┴───────┐
          │               │
         YES             NO
          │               │
      PINCHING        NOT PINCHING
          │
          │
 previous = False?
          │
        YES
          │
     PINCH STARTED

-------------------------------------------------------------------------------
PyAutoGUI - https://pyautogui.readthedocs.io/en/latest/

We are not using cv.setMouseCallback() for mouse events. That callback only handles mouse events inside the OpenCV window. If we want to control the actual operating-system cursor,  we'll use PyAutoGUI.

pyautogui.click() is kept inside the if-block of pinch test. If pinch is detected, i makes the cursor click at it's current position.

we need to convert mediapipe coordinates to screen coordinates so: we mutiply screen dimensions with required finger's (here tip of index finger) x and y coordinates, store these mutiplications in two variables and then passing those two varibles to pyautogui.moveTo() function. this makes our cursor follow the fingertip of index finger

But the movement of cursor horizontally is mirrored, so we have to compensate for that in our code

The jitter problem - even a tiny change in mediapipe's coordinates immediately changes the cursor position, to fix this we'll use smoothing.

smooth = previous + α(new - previous), where α ranges from 0 to 1
for example if α=0.2, it means to move 20% of the way toward the new position

Issue encountered - while pinching, cursor still follows the index fingertip. To solve this, we are going to use a gesture state. If pinching is FALSE, move the cursor normally. If pinching is TRUE, don't move the cursor and detect the pinch.

          Detect hand
              ↓
        Calculate pinch
              ↓
       ┌──────┴──────┐
       │             │
    Pinching      Not pinching
       │             │
       |             ↓
       |         Move cursor
       ↓
   Freeze cursor

Another issue is that involvement of other hand into the camera frame confuses the cursor, because we have only told mediapipe to assume the very first hand it detects is the hand we want to use. To solve this, we manually set num_hands = 1. 

We are furthur normalizing the coordinates of mediapipe so that we don't have to move our hands too much, so instead of using the entire camera frame to move the cursor or to control it, we are defining a smaller active region in the middle of it.

added the normalized frame using cv.rectangle

reduced jitter issue furthur by using dead zone concept, we have put the dead zone before the smoothing because it's better for us to tackle the raw movement of cursor first.
