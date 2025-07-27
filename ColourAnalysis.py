import cv2
import numpy as np
import requests
import json
from collections import Counter
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from PIL import Image, ImageTk
import threading
import base64
import io
import os
from dotenv import load_dotenv
import mediapipe as mp
import math

# Load environment variables
load_dotenv()

class FacialColorAnalyzer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Advanced Facial Color Analysis")
        self.root.geometry("1400x800")
        
        # Initialize OpenCV for basic face detection (fallback)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_draw_styles = mp.solutions.drawing_styles
        
        try:
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.use_landmarks = True
            print("✅ MediaPipe Face Mesh initialized successfully!")
        except Exception as e:
            print(f"❌ MediaPipe initialization failed: {e}")
            self.use_landmarks = False
        
        self.cap = None
        self.current_frame = None
        self.color_palette = None
        self.analysis_result = None
        
        # Claude API setup
        self.api_key = os.getenv('CLAUDE_API_KEY')
        self.api_url = "https://api.anthropic.com/v1/messages"
        
        # Check if API key is loaded
        if not self.api_key:
            messagebox.showerror("Configuration Error", 
                               "Claude API key not found. Please create a .env file with CLAUDE_API_KEY=your_key_here")
            self.root.quit()
            return
        
        self.setup_gui()
        
    def setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Advanced Facial Color Analysis", font=("Arial", 18, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=10)
        
        # Camera controls
        camera_frame = ttk.LabelFrame(main_frame, text="Camera Controls", padding="5")
        camera_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        self.start_btn = ttk.Button(camera_frame, text="Start Camera", command=self.start_camera)
        self.start_btn.grid(row=0, column=0, padx=5)
        
        self.capture_btn = ttk.Button(camera_frame, text="Analyze Colors", command=self.capture_and_analyze, state="disabled")
        self.capture_btn.grid(row=0, column=1, padx=5)
        
        self.toggle_overlay_btn = ttk.Button(camera_frame, text="Toggle Overlay", command=self.toggle_overlay, state="disabled")
        self.toggle_overlay_btn.grid(row=0, column=2, padx=5)
        
        self.stop_btn = ttk.Button(camera_frame, text="Stop Camera", command=self.stop_camera, state="disabled")
        self.stop_btn.grid(row=0, column=3, padx=5)
        
        # Show overlay toggle
        self.show_overlay = tk.BooleanVar(value=True)
        self.show_palette = tk.BooleanVar(value=True)
        
        # Create main layout
        content_frame = ttk.Frame(main_frame)
        content_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        
        # Video display (larger)
        self.video_label = ttk.Label(content_frame, text="Camera feed will appear here")
        self.video_label.grid(row=0, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        
        # Status label under video
        self.status_var = tk.StringVar(value="Ready to start")
        self.status_label = ttk.Label(content_frame, textvariable=self.status_var, font=("Arial", 10))
        self.status_label.grid(row=1, column=0, columnspan=2, pady=5)
        
        # Right side results
        results_frame = ttk.LabelFrame(content_frame, text="Color Analysis Results", padding="10")
        results_frame.grid(row=0, column=2, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0))
        
        # Analysis results text
        self.results_text = scrolledtext.ScrolledText(results_frame, width=45, height=35, wrap=tk.WORD, font=("Arial", 9))
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(2, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        content_frame.grid_rowconfigure(0, weight=1)
        content_frame.grid_columnconfigure(0, weight=2)
        content_frame.grid_columnconfigure(2, weight=1)
        results_frame.grid_rowconfigure(0, weight=1)
        results_frame.grid_columnconfigure(0, weight=1)
        
    def start_camera(self):
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open camera")
                return
            
            # Set camera resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            self.start_btn.config(state="disabled")
            self.capture_btn.config(state="normal")
            self.toggle_overlay_btn.config(state="normal")
            self.stop_btn.config(state="normal")
            self.status_var.set("Camera started - Position your face in the frame")
            
            self.update_frame()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera: {str(e)}")
    
    def toggle_overlay(self):
        self.show_overlay.set(not self.show_overlay.get())
    
    def get_face_landmarks(self, frame):
        """Get facial landmarks using MediaPipe"""
        if not self.use_landmarks:
            return None
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None
        
        # Get the first face's landmarks
        face_landmarks = results.multi_face_landmarks[0]
        
        # Convert to numpy array (468 landmarks)
        h, w = frame.shape[:2]
        points = []
        
        # MediaPipe gives normalized coordinates, convert to pixel coordinates
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            points.append((x, y))
        
        return np.array(points)
    
    def draw_face_mesh(self, frame, landmarks):
        """Draw MediaPipe face mesh overlay"""
        if not self.use_landmarks:
            return self.draw_basic_face_detection(frame)
        
        # Process frame with MediaPipe to get the mesh structure
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw the face mesh contours
                self.mp_draw.draw_landmarks(
                    frame,
                    face_landmarks,
                    self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_draw_styles.get_default_face_mesh_contours_style()
                )
                
                # Draw key landmarks as points
                for idx, landmark in enumerate(face_landmarks.landmark):
                    if idx in [10, 151, 234, 454, 234, 10, 151]:  # Key face points
                        h, w = frame.shape[:2]
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
        
        return frame
    
    def draw_basic_face_detection(self, frame):
        """Fallback to basic OpenCV face detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.putText(frame, "Face Detected", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return frame
    
    def extract_detailed_facial_colors(self, frame, landmarks):
        """Extract colors from specific facial regions using MediaPipe landmarks"""
        if landmarks is None:
            return self.extract_basic_facial_colors(frame)
        
        colors = {}
        
        # MediaPipe landmark indices for facial regions
        # Based on the 468-point face mesh model
        regions = {
            'forehead': [9, 10, 151, 337, 299, 333, 298, 301],
            'left_cheek': [116, 117, 118, 119, 120, 121, 126, 142, 36, 205, 206, 207, 213, 192, 147],
            'right_cheek': [345, 346, 347, 348, 349, 350, 355, 371, 266, 425, 426, 427, 436, 416, 376],
            'nose_bridge': [6, 19, 20, 1, 2],
            'nose_tip': [1, 2, 5, 4, 6, 168, 8, 9, 10, 151],
            'chin': [18, 175, 199, 200, 16, 17, 18, 175],
            'left_eye_area': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            'right_eye_area': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            'upper_lip': [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318],
            'lower_lip': [146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318],
            'under_eye_left': [133, 155, 154, 153, 145, 144, 163, 7],
            'under_eye_right': [362, 398, 384, 385, 386, 387, 388, 466]
        }
        
        for region_name, indices in regions.items():
            try:
                # Get points for this region
                region_points = []
                for idx in indices:
                    if idx < len(landmarks):
                        region_points.append(landmarks[idx])
                
                if len(region_points) < 3:  # Need at least 3 points for convex hull
                    continue
                
                region_points = np.array(region_points)
                
                # Create mask for the region
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                hull = cv2.convexHull(region_points)
                cv2.fillPoly(mask, [hull], 255)
                
                # Extract color from masked region
                masked_region = cv2.bitwise_and(frame, frame, mask=mask)
                non_zero_pixels = masked_region[mask > 0]
                
                if len(non_zero_pixels) > 10:  # Ensure we have enough pixels
                    mean_color = np.mean(non_zero_pixels, axis=0)
                    colors[region_name] = [int(mean_color[2]), int(mean_color[1]), int(mean_color[0])]  # BGR to RGB
            except Exception as e:
                print(f"Error processing region {region_name}: {e}")
                continue
        
        # Analyze overall undertone
        if 'left_cheek' in colors and 'right_cheek' in colors:
            avg_cheek = np.mean([colors['left_cheek'], colors['right_cheek']], axis=0)
            colors['overall_undertone'] = self.analyze_undertone_from_rgb(avg_cheek)
        elif 'left_cheek' in colors:
            colors['overall_undertone'] = self.analyze_undertone_from_rgb(colors['left_cheek'])
        elif 'right_cheek' in colors:
            colors['overall_undertone'] = self.analyze_undertone_from_rgb(colors['right_cheek'])
        
        return colors
    
    def extract_basic_facial_colors(self, frame):
        """Fallback method for basic color extraction using OpenCV face detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return {}
        
        # Use the largest detected face
        face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = face
        
        # Ensure face region is within frame bounds
        x = max(0, x)
        y = max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)
        
        face_region = frame[y:y+h, x:x+w]
        
        if face_region.size == 0:
            return {}
        
        colors = {
            'skin_tone': self.get_average_color(face_region[h//4:3*h//4, w//4:3*w//4]),
            'forehead': self.get_average_color(face_region[h//8:h//4, w//4:3*w//4]),
            'left_cheek': self.get_average_color(face_region[h//3:2*h//3, w//8:w//3]),
            'right_cheek': self.get_average_color(face_region[h//3:2*h//3, 2*w//3:7*w//8]),
            'lip_area': self.get_average_color(face_region[2*h//3:4*h//5, w//3:2*w//3]),
        }
        
        # Remove any invalid colors
        colors = {k: v for k, v in colors.items() if v is not None}
        
        return colors
    
    def get_average_color(self, region):
        """Get average color from a region"""
        if region.size == 0:
            return None
        
        try:
            mean_color = np.mean(region.reshape(-1, 3), axis=0)
            return [int(mean_color[2]), int(mean_color[1]), int(mean_color[0])]  # BGR to RGB
        except:
            return None
    
    def analyze_undertone_from_rgb(self, rgb_color):
        """Analyze undertone from RGB values"""
        if len(rgb_color) < 3:
            return {'type': 'neutral', 'confidence': 0.5}
        
        r, g, b = rgb_color[:3]
        
        # More sophisticated undertone analysis
        if r > g and r > b:
            if (r - g) > (r - b):
                return {'type': 'warm', 'confidence': 0.8}
            else:
                return {'type': 'neutral-warm', 'confidence': 0.7}
        elif b > r and b > g:
            return {'type': 'cool', 'confidence': 0.9}
        elif g > r and g > b:
            return {'type': 'neutral', 'confidence': 0.8}
        else:
            # Calculate ratios for more nuanced analysis
            warm_score = (r - b) / (r + b + g) if (r + b + g) > 0 else 0
            if warm_score > 0.1:
                return {'type': 'warm', 'confidence': 0.7}
            elif warm_score < -0.1:
                return {'type': 'cool', 'confidence': 0.7}
            else:
                return {'type': 'neutral', 'confidence': 0.6}
    
    def generate_color_palette(self, colors, analysis_text):
        """Generate a color palette based on analysis"""
        # Define seasonal palettes
        seasonal_palettes = {
            'spring': ['#FFB6C1', '#98FB98', '#87CEEB', '#F0E68C', '#DDA0DD', '#F5DEB3'],
            'summer': ['#B0C4DE', '#DDA0DD', '#F0F8FF', '#E6E6FA', '#FFB6C1', '#98FB98'],
            'autumn': ['#CD853F', '#D2691E', '#A0522D', '#8B4513', '#DAA520', '#B22222'],
            'winter': ['#000080', '#8B0000', '#2F4F4F', '#000000', '#FFFFFF', '#C0C0C0']
        }
        
        # Determine season from analysis
        season = 'spring'  # default
        analysis_lower = analysis_text.lower()
        
        if 'winter' in analysis_lower:
            season = 'winter'
        elif 'autumn' in analysis_lower or 'fall' in analysis_lower:
            season = 'autumn'
        elif 'summer' in analysis_lower:
            season = 'summer'
        
        palette_colors = seasonal_palettes[season]
        
        return palette_colors, season
    
    def draw_color_palette_overlay(self, frame, palette_colors, season):
        """Draw color palette overlay on the frame"""
        if not palette_colors:
            return frame
        
        overlay = frame.copy()
        height, width = frame.shape[:2]
        
        # Draw palette background
        palette_y = height - 120
        cv2.rectangle(overlay, (10, palette_y), (width - 10, height - 10), (0, 0, 0), -1)
        cv2.rectangle(overlay, (10, palette_y), (width - 10, height - 10), (255, 255, 255), 2)
        
        # Draw title
        cv2.putText(overlay, f"Your {season.title()} Color Palette", (20, palette_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw color swatches
        swatch_width = (width - 40) // len(palette_colors)
        for i, hex_color in enumerate(palette_colors):
            try:
                # Convert hex to BGR
                hex_color = hex_color.lstrip('#')
                rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                bgr = (rgb[2], rgb[1], rgb[0])
                
                x1 = 20 + i * swatch_width
                x2 = x1 + swatch_width - 5
                y1 = palette_y + 35
                y2 = palette_y + 75
                
                cv2.rectangle(overlay, (x1, y1), (x2, y2), bgr, -1)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 1)
            except:
                continue
        
        # Blend with original frame
        result = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
        return result
    
    def update_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame.copy()
                display_frame = frame.copy()
                
                # Get landmarks if available
                landmarks = None
                face_detected = False
                
                if self.use_landmarks:
                    landmarks = self.get_face_landmarks(frame)
                    face_detected = landmarks is not None
                else:
                    # Fallback to basic face detection
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                    face_detected = len(faces) > 0
                
                # Draw face mesh overlay
                if self.show_overlay.get():
                    display_frame = self.draw_face_mesh(display_frame, landmarks)
                
                # Draw color palette if available
                if self.color_palette and self.show_palette.get():
                    palette_colors, season = self.color_palette
                    display_frame = self.draw_color_palette_overlay(display_frame, palette_colors, season)
                
                # Convert frame for display
                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                frame_pil = frame_pil.resize((800, 600), Image.Resampling.LANCZOS)
                frame_tk = ImageTk.PhotoImage(frame_pil)
                
                self.video_label.config(image=frame_tk)
                self.video_label.image = frame_tk
                
                # Update status
                if face_detected:
                    self.status_var.set("Face detected! Ready to analyze colors.")
                else:
                    self.status_var.set("Position your face in the frame for detection")
            
            # Schedule next update
            self.root.after(30, self.update_frame)
    
    def stop_camera(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.start_btn.config(state="normal")
        self.capture_btn.config(state="disabled")
        self.toggle_overlay_btn.config(state="disabled")
        self.stop_btn.config(state="disabled")
        self.video_label.config(image="", text="Camera stopped")
        self.video_label.image = None
        self.status_var.set("Camera stopped")
    
    def analyze_with_claude(self, color_data):
        """Send color data to Claude API for analysis"""
        if not self.api_key:
            return "API key not configured."
        
        # Format the color data for analysis
        color_description = "Detailed Facial Color Analysis:\n"
        for region, color in color_data.items():
            if isinstance(color, list) and len(color) == 3:
                color_description += f"- {region.replace('_', ' ').title()}: RGB{color}\n"
            elif isinstance(color, dict):
                color_description += f"- {region.replace('_', ' ').title()}: {color}\n"
        
        prompt = f"""
        I need you to analyze detailed facial colors for comprehensive personalized color analysis. Based on the following extracted facial color data from different regions of the face, please provide detailed recommendations.

        {color_description}

        Please provide a comprehensive analysis including:
        1. **Seasonal Color Analysis** (Spring, Summer, Autumn, Winter) with detailed explanation based on the specific colors extracted
        2. **Best Colors for Clothing** - specific color names with hex codes, organized by category (tops, bottoms, formal wear)
        3. **Colors to Avoid** - specific colors that would clash with this coloring
        4. **Makeup Recommendations**:
           - Foundation undertone and shade recommendations
           - Best lip colors (with specific names/shades)
           - Eye makeup colors (eyeshadows, eyeliner, mascara)
           - Blush colors that complement the natural cheek tones
        5. **Hair Color Suggestions** - colors that would complement this natural coloring
        6. **Accessories** - jewelry metals and colors that work best
        7. **Confidence Level** (1-10) and any limitations of this analysis

        Please format your response with clear sections and be specific with color recommendations including hex codes where possible.
        """
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        
        data = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 3000,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return result['content'][0]['text']
            else:
                return f"API Error {response.status_code}: {response.text}"
                
        except requests.exceptions.RequestException as e:
            return f"Network error: {str(e)}"
        except Exception as e:
            return f"Error analyzing colors: {str(e)}"
    
    def capture_and_analyze(self):
        """Capture current frame and perform comprehensive color analysis"""
        if self.current_frame is None:
            messagebox.showerror("Error", "No frame available for analysis")
            return
        
        self.status_var.set("Analyzing facial colors...")
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "🎨 Analyzing your unique facial colors...\n\nProcessing:\n- Detecting facial features\n- Extracting color data\n- Analyzing undertones\n- Generating recommendations\n\nPlease wait...")
        
        # Run analysis in separate thread to avoid blocking GUI
        thread = threading.Thread(target=self._analyze_thread)
        thread.daemon = True
        thread.start()
    
    def _analyze_thread(self):
        """Thread function for analysis"""
        try:
            # Get landmarks
            landmarks = None
            if self.use_landmarks:
                landmarks = self.get_face_landmarks(self.current_frame)
            
            # Extract colors from current frame
            colors = self.extract_detailed_facial_colors(self.current_frame, landmarks)
            
            if not colors:
                self.root.after(0, lambda: self._update_results("Analysis failed: No face detected. Please ensure good lighting and face the camera directly."))
                return
            
            # Analyze with Claude
            analysis_result = self.analyze_with_claude(colors)
            
            # Generate color palette
            palette_colors, season = self.generate_color_palette(colors, analysis_result)
            self.color_palette = (palette_colors, season)
            
            # Format results with aesthetic presentation
            final_result = self._format_results(colors, analysis_result, season)
            
            self.root.after(0, lambda: self._update_results(final_result))
            
        except Exception as e:
            self.root.after(0, lambda: self._update_results(f"Error during analysis: {str(e)}"))
    
    def _format_results(self, colors, analysis, season):
        """Format results in an aesthetic way"""
        result = "🎨 PERSONALIZED COLOR ANALYSIS RESULTS\n"
        result += "=" * 60 + "\n\n"
        
        result += "👤 YOUR FACIAL COLOR PROFILE:\n"
        result += "-" * 30 + "\n"
        for region, color in colors.items():
            if isinstance(color, list) and len(color) == 3:
                result += f"• {region.replace('_', ' ').title()}: RGB{color}\n"
            elif isinstance(color, dict):
                result += f"• {region.replace('_', ' ').title()}: {color}\n"
        
        result += f"\n🌟 SEASONAL CLASSIFICATION: {season.upper()}\n"
        result += "=" * 60 + "\n\n"
        
        result += "🎯 EXPERT ANALYSIS & RECOMMENDATIONS:\n"
        result += analysis
        
        result += f"\n\n💡 TIP: Your {season} color palette is now displayed on your camera feed!"
        result += "\n    Use the 'Toggle Overlay' button to show/hide the face mesh."
        
        return result
    
    def _update_results(self, text):
        """Update results in main thread"""
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, text)
        self.status_var.set("✨ Analysis complete! Your personalized color palette is ready.")
    
    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def on_closing(self):
        self.stop_camera()
        self.root.destroy()

if __name__ == "__main__":
    print("🎨 Advanced Facial Color Analysis")
    print("=" * 50)
    print("📋 Requirements check:")

    # Check MediaPipe installation
    try:
        import mediapipe as mp
        print("✅ MediaPipe installed")
    except ImportError:
        print("❌ MediaPipe not found. Please install it with: pip install mediapipe")
        exit()

    # Check OpenCV installation
    try:
        import cv2
        print("✅ OpenCV installed")
    except ImportError:
        print("❌ OpenCV not found. Please install it with: pip install opencv-python")
        exit()

    # Check requests
    try:
        import requests
        print("✅ Requests installed")
    except ImportError:
        print("❌ Requests not found. Please install it with: pip install requests")
        exit()

    # Check tkinter
    try:
        import tkinter
        print("✅ Tkinter available")
    except ImportError:
        print("❌ Tkinter not found. Please install it. (Often included with Python)")
        exit()

    # Run the app
    from ColourAnalysis import FacialColorAnalyzer
    app = FacialColorAnalyzer()
    app.run()
