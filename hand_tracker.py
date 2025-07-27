import cv2
import mediapipe as mp
import math

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

# NEW Keyboard settings - smaller keys
key_width = 60        # Smaller!
key_height = 60       # Smaller!
key_gap = 5          # Smaller gap
start_y = 250        # Move up to fit more rows

# Define full keyboard layout
keyboard_layout = [
    "QWERTYUIOP",
    "ASDFGHJKL", 
    "ZXCVBNM"
]

# Create all keys
keys = []

for row_index, row in enumerate(keyboard_layout):
    # Calculate starting X to center each row
    row_width = len(row) * key_width + (len(row) - 1) * key_gap
    start_x = (640 - row_width) // 2  # Center the row
    
    for i, char in enumerate(row):
        x = start_x + i * (key_width + key_gap)
        y = start_y + row_index * (key_height + key_gap)
        key = Key(x, y, key_width, key_height, char)
        keys.append(key)

# Add space bar (wider than normal keys)
space_width = 300
space_x = (640 - space_width) // 2
space_y = start_y + 3 * (key_height + key_gap)
space_key = Key(space_x, space_y, space_width, key_height, "SPACE")
keys.append(space_key)

def draw_keyboard(frame, keys, hovered_key=None, clicked_key=None):
    for key in keys:
        # Choose color based on key state
        if key == clicked_key:
            color = (0, 255, 0)  # Green when clicked
        elif key == hovered_key:
            color = (100, 100, 255)  # Light blue when hovered
        else:
            color = (200, 200, 200)  # Default gray
        
        # Draw key background
        cv2.rectangle(frame, (key.x, key.y), 
                     (key.x + key.width, key.y + key.height), 
                     color, -1)
        
        # Draw border
        cv2.rectangle(frame, (key.x, key.y), 
                     (key.x + key.width, key.y + key.height), 
                     (0, 0, 0), 2)
        
        # Center text better
        if key.text == "SPACE":
            text_x = key.x + key.width // 2 - 30
        else:
            text_x = key.x + key.width // 2 - 8
        text_y = key.y + key.height // 2 + 8
        
        cv2.putText(frame, key.text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

def is_point_in_key(finger_x, finger_y, key):
    """Check if a point is inside a key's rectangle"""
    return (key.x <= finger_x <= key.x + key.width and 
            key.y <= finger_y <= key.y + key.height)

def get_finger_positions(hand_landmarks, frame_shape):
    """Extract finger tip positions from hand landmarks"""
    # Index finger tip (landmark 8)
    index_tip = hand_landmarks.landmark[8]
    index_x = int(index_tip.x * frame_shape[1])
    index_y = int(index_tip.y * frame_shape[0])
    
    # Thumb tip (landmark 4) 
    thumb_tip = hand_landmarks.landmark[4]
    thumb_x = int(thumb_tip.x * frame_shape[1])
    thumb_y = int(thumb_tip.y * frame_shape[0])
    
    return index_x, index_y, thumb_x, thumb_y

def calculate_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

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
    
    # Check if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get finger positions
            index_x, index_y, thumb_x, thumb_y = get_finger_positions(hand_landmarks, frame.shape)
            
            # Draw finger tip (for debugging)
            cv2.circle(frame, (index_x, index_y), 10, (255, 0, 0), -1)  # Blue dot
            cv2.circle(frame, (thumb_x, thumb_y), 10, (0, 0, 255), -1)   # Red dot
            
            # Check which key is being hovered
            hovered_key = None
            for key in keys:
                if is_point_in_key(index_x, index_y, key):
                    hovered_key = key
                    break
            
            # Check for pinch gesture (click detection)
            distance = calculate_distance(index_x, index_y, thumb_x, thumb_y)
            clicked_key = None
            
            if distance < 40 and hovered_key:  # Adjust threshold as needed
                clicked_key = hovered_key
                print(f"Key pressed: {clicked_key.text}")  # Debug output
            
            # Draw keyboard with hover/click states
            draw_keyboard(frame, keys, hovered_key, clicked_key)
    else:
        # No hand detected, draw normal keyboard
        draw_keyboard(frame, keys)
    
    # Show the processed frame
    cv2.imshow("Hand Tracking", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
hands.close()