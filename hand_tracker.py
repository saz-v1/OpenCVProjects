import cv2
import mediapipe as mp

# Set up MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Add this after your MediaPipe setup, before the while loop
class Key:
    def __init__(self, x, y, width, height, text):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text
        self.clicked = False

# Initialize the hands detector
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,             
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Keyboard settings
key_width = 80
key_height = 80
key_gap = 10
start_x = 50
start_y = 400  # Near bottom of screen

# Create keys for top row
keys = []
qwerty_row = "QWERTYUIOP"

for i, char in enumerate(qwerty_row):
    x = start_x + i * (key_width + key_gap)
    y = start_y
    key = Key(x, y, key_width, key_height, char)
    keys.append(key)

def draw_keyboard(frame, keys):
    for key in keys:
        # Draw the key rectangle
        cv2.rectangle(frame, (key.x, key.y), 
                     (key.x + key.width, key.y + key.height), 
                     (200, 200, 200), -1)  # Gray filled rectangle
        
        # Draw the border
        cv2.rectangle(frame, (key.x, key.y), 
                     (key.x + key.width, key.y + key.height), 
                     (0, 0, 0), 2)  # Black border
        
        # Draw the text
        cv2.putText(frame, key.text, 
                   (key.x + 25, key.y + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)


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
    
    draw_keyboard(frame, keys)
    # Show the processed frame
    cv2.imshow("Hand Tracking", frame)  # Show 'frame', not 'ret'
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



# Clean up
cap.release()
cv2.destroyAllWindows()
hands.close()
