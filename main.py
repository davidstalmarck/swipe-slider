import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Track the previous position of the hand landmarks and timing
last_swipe_time = 0
previous_state = None
previous_y = None
initial_index_x = None
state_confirmed = False

# Start capturing video
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame and convert to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the x-coordinates and y-coordinates of the index finger tip and thumb tip
            index_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1]
            index_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0]
            thumb_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * frame.shape[1]

            # Initialize the starting position of the index finger
            if initial_index_x is None:
                initial_index_x = index_x

            # Determine current state based on the relative position of index and thumb
            if index_x > thumb_x + 50:
                current_state = "left"
            elif index_x < thumb_x - 50:
                current_state = "right"
            else:
                current_state = None

            # Confirm the state only if it's stable (no immediate change)
            if current_state and current_state != previous_state:
                if not state_confirmed:
                    state_confirmed = True
                    initial_index_x = index_x  # Set initial position for movement check
                else:
                    # Check movement length for swipe
                    movement_length = abs(index_x - initial_index_x)
                    if movement_length > 100:  # Minimum length threshold
                        if current_state == "left":
                            print("Left swipe detected!")
                            pyautogui.press("left")
                        elif current_state == "right":
                            print("Right swipe detected!")
                            pyautogui.press("right")

                        last_swipe_time = time.time()
                        state_confirmed = False  # Reset confirmation after swipe

            # Reset to neutral if the hand moves far enough vertically
            if previous_y is not None and abs(index_y - previous_y) > 100:
                print("State reset to neutral due to vertical movement.")
                current_state = None
                state_confirmed = False
                initial_index_x = None

            # Update previous state and y-coordinate
            previous_state = current_state
            previous_y = index_y

            # Display the current state on the frame
            state_label = current_state if state_confirmed else "neutral"
            cv2.putText(frame, f"State: {state_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the video feed
    cv2.imshow("Hand Tracking", frame)

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
