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

