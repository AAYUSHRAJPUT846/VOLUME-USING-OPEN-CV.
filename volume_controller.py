import cv2
import mediapipe as mp
import numpy as np
import math
from pynput.keyboard import Key, Controller # <-- New Import

# --- 1. Setup and Initialization ---

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Initialize pynput keyboard controller
keyboard = Controller() # <-- New Controller

# Open the default webcam
cap = cv2.VideoCapture(0)

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# --- 2. Volume Control Variables (No PyCaw required) ---
# Note: Since we use key presses, we track the virtual volume percentage manually for visualization.
volPer = 50 # Start volume at 50%
volBar = np.interp(volPer, [0, 100], [400, 150]) # Visual bar position

# --- 3. Main Loop and Hand Tracking ---

while True:
    success, img = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    lmList = [] 

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
            
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
            # 

    # --- 4. Gesture Detection and Key Press Control ---
    
    if len(lmList) != 0:
        x1, y1 = lmList[4][1], lmList[4][2] # Thumb Tip (ID 4)
        x2, y2 = lmList[8][1], lmList[8][2] # Index Finger Tip (ID 8)
        
        cx, cy = (x1 + x2) // 2, (y1 + y2)
        
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        
        length = math.hypot(x2 - x1, y2 - y1)
        
        # Map the distance length (25-250) to the volume percentage (0-100)
        newVolPer = np.interp(length, [25, 250], [0, 100])
        
        # Check if the new volume percentage is significantly different from the current one
        # This prevents constantly spamming key presses.
        
        if newVolPer > volPer + 5: # Volume needs to go UP by more than 5%
            keyboard.press(Key.media_volume_up)
            keyboard.release(Key.media_volume_up)
            volPer = newVolPer # Update visual percentage
            
        elif newVolPer < volPer - 5: # Volume needs to go DOWN by more than 5%
            keyboard.press(Key.media_volume_down)
            keyboard.release(Key.media_volume_down)
            volPer = newVolPer # Update visual percentage
            
        # Draw volume bar and percentage based on the updated volPer
        volBar = np.interp(volPer, [0, 100], [400, 150])

        if length < 50:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

    # --- 5. Visualization on Screen ---
    
    # Draw Volume Bar (Rectangle)
    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3) # Outline
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED) # Filled bar
    
    # Draw Volume Percentage
    cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)
    
    # Display the final image
    cv2.imshow("Hand Volume Controller", img)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and destroy all windows
cap.release()
cv2.destroyAllWindows()