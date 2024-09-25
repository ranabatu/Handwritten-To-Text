import cv2
import mediapipe as mp
import time
import keras_ocr
import numpy as np

# Mediapipe hand tracking module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

canvas = None # for blank canvas

writing = False # activate writing mode
erasing = False  # activate erasing mode

current_mode = "Normal"  # current mod

# for FPS measurement
previous_time = 0
current_time = 0

# previous points
_x, _y = None, None

# using Keras OCR
pipeline = keras_ocr.pipeline.Pipeline()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1) # mirror effect

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert frame to RGB

    results = hands.process(frame_rgb) # detect hands with Mediapipe

    if canvas is None: # create a blank canvas in the first frame
        canvas = frame.copy() * 0

    # if hand is detected process only the first hand
    if results.multi_hand_landmarks:
        # just take the first hand
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # take the tip of the index finger landmark 8
        index_finger_tip = hand_landmarks.landmark[8]
        h, w, _ = frame.shape

        # get fingertip coordinates
        x = int(index_finger_tip.x * w)
        y = int(index_finger_tip.y * h)

        # draw a circle to mark the tip of the finger
        cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)

        # follow the fingertip in drawing mode
        if writing:
            if _x is not None and _y is not None:
                # fill in the gaps by drawing a line between the previous point and the current point
                cv2.line(canvas, (_x, _y), (x, y), (255, 255, 255), 10)
            # save current point as previous point
            _x, _y = x, y
        # if the erase mode is active, erase the specified area with the tip of your finger
        elif erasing:
            cv2.circle(canvas, (x, y), 7, (0, 0, 0), -1)  # erase area
            _x, _y = None, None  # reset previous point in delete mode
        else:
            _x, _y = None, None  # reset previous point if not in write/delete mode

        # draw Mediapipe hand connections
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # add canvas to image
    frame = cv2.add(frame, canvas)

    # FPS measurement
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time

    # show FPS on the screen
    cv2.putText(frame, f'FPS: {int(fps)}', (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    
    # show current mode info
    cv2.putText(frame, f'Mode: {current_mode}', (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # recognize text with OCR when the T key is pressed
    key = cv2.waitKey(1)
    if key == ord('t'):
        canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) # convert canvas black and white
        _, canvas_thresh = cv2.threshold(canvas_gray, 127, 255, cv2.THRESH_BINARY_INV)

        # process canvas with OCR
        canvas_rgb = cv2.cvtColor(canvas_thresh, cv2.COLOR_GRAY2RGB)  # turns to RGB
        ocr_result = pipeline.recognize([canvas_rgb])

        # shows the recognized text on screen
        for text, box in ocr_result[0]:
            print(f"Recognized Text: {text}")
            cv2.putText(frame, f"Recognized Text: {text}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Handwritten text", frame)

    # turn on_/off text when pressing W
    if key == ord('w'):
        writing = not writing
        erasing = False # close the erasing mode
        current_mode = "Writing" if writing else "Normal"
        
    if key == ord('e'):
        erasing = not erasing
        writing = False # close the writing mode
        current_mode = "Erasing" if erasing else "Normal"

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
