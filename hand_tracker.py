import cv2
import mediapipe as mp

# Set up MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize the hands detector
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,             
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Set up Camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Get BOTH success flag AND frame
    
    if not ret: 
        print("Did not receive frame")
        break 
    
    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
    results = hands.process(rgb_frame)  # Process the RGB frame
    
    # If hands are detected, draw landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)  # Draw on 'frame'
    
    # Show the processed frame
    cv2.imshow("Hand Tracking", frame)  # Show 'frame', not 'ret'
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
hands.close()
