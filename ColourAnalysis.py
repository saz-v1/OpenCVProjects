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

# Load environment variables
load_dotenv()

class FacialColorAnalyzer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Facial Color Analysis App")
        self.root.geometry("1200x700")
        
        # Initialize OpenCV
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.cap = None
        self.current_frame = None
        
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
        title_label = ttk.Label(main_frame, text="Facial Color Analysis", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=10)
        
        # Camera controls
        camera_frame = ttk.LabelFrame(main_frame, text="Camera Controls", padding="5")
        camera_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        self.start_btn = ttk.Button(camera_frame, text="Start Camera", command=self.start_camera)
        self.start_btn.grid(row=0, column=0, padx=5)
        
        self.capture_btn = ttk.Button(camera_frame, text="Capture & Analyze", command=self.capture_and_analyze, state="disabled")
        self.capture_btn.grid(row=0, column=1, padx=5)
        
        self.stop_btn = ttk.Button(camera_frame, text="Stop Camera", command=self.stop_camera, state="disabled")
        self.stop_btn.grid(row=0, column=2, padx=5)
        
        # Create a paned window for layout
        paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned_window.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        
        # Left frame for video
        left_frame = ttk.Frame(paned_window)
        paned_window.add(left_frame, weight=1)
        
        # Video display
        self.video_label = ttk.Label(left_frame, text="Camera feed will appear here")
        self.video_label.grid(row=0, column=0, pady=10)
        
        # Status label under video
        self.status_var = tk.StringVar(value="Ready to start")
        self.status_label = ttk.Label(left_frame, textvariable=self.status_var)
        self.status_label.grid(row=1, column=0, pady=5)
        
        # Right frame for color analysis results
        right_frame = ttk.LabelFrame(paned_window, text="Color Analysis Results", padding="10")
        paned_window.add(right_frame, weight=1)
        
        # Color display canvas
        self.color_canvas = tk.Canvas(right_frame, width=300, height=200, bg="white")
        self.color_canvas.grid(row=0, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        # Analysis results text
        self.results_text = scrolledtext.ScrolledText(right_frame, width=40, height=20, wrap=tk.WORD)
        self.results_text.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Configure grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(3, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        left_frame.grid_rowconfigure(0, weight=1)
        left_frame.grid_columnconfigure(0, weight=1)
        right_frame.grid_rowconfigure(1, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)
        
    def start_camera(self):
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open camera")
                return
            
            self.start_btn.config(state="disabled")
            self.capture_btn.config(state="normal")
            self.stop_btn.config(state="normal")
            self.status_var.set("Camera started - Position your face in the frame")
            
            self.update_frame()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera: {str(e)}")
    
    def update_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame.copy()
                
                # Detect faces
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                
                # Draw rectangles around faces
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(frame, "Face Detected", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                # Convert frame for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                frame_pil = frame_pil.resize((640, 480), Image.Resampling.LANCZOS)
                frame_tk = ImageTk.PhotoImage(frame_pil)
                
                self.video_label.config(image=frame_tk)
                self.video_label.image = frame_tk
                
                # Update status based on face detection
                if len(faces) > 0:
                    self.status_var.set(f"Face detected! Ready to capture. ({len(faces)} face(s) found)")
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
        self.stop_btn.config(state="disabled")
        self.video_label.config(image="", text="Camera stopped")
        self.video_label.image = None
        self.status_var.set("Camera stopped")
    
    def extract_facial_colors(self, frame):
        """Extract dominant colors from facial regions"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return None, "No face detected. Please ensure good lighting and face the camera directly."
        
        # Use the largest detected face
        face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = face
        
        # Extract face region with some padding
        padding = 10
        face_region = frame[max(0, y-padding):min(frame.shape[0], y+h+padding), 
                          max(0, x-padding):min(frame.shape[1], x+w+padding)]
        
        # Check lighting quality
        lighting_quality = self.assess_lighting_quality(face_region)
        if lighting_quality['score'] < 0.5:
            return None, lighting_quality['message']
        
        # Extract colors from different facial areas
        colors = {
            'skin_tone': self.get_skin_tone(face_region),
            'under_eye': self.get_under_eye_color(face_region),
            'lip_area': self.get_lip_color(face_region),
            'overall_undertone': self.analyze_undertone(face_region)
        }
        
        return colors, "Colors extracted successfully"
    
    def assess_lighting_quality(self, face_region):
        """Assess if lighting is adequate for color analysis"""
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        if mean_brightness < 80:
            return {'score': 0.2, 'message': 'Lighting too dark. Please move to better lighting or turn on more lights.'}
        elif mean_brightness > 200:
            return {'score': 0.3, 'message': 'Lighting too bright. Please reduce harsh lighting or move away from direct light.'}
        elif std_brightness < 20:
            return {'score': 0.4, 'message': 'Lighting too flat. Please use natural lighting or add some directional light.'}
        else:
            return {'score': 0.8, 'message': 'Good lighting conditions detected.'}
    
    def get_skin_tone(self, face_region):
        """Extract dominant skin tone from face region"""
        # Focus on central face area (avoid hair and background)
        h, w = face_region.shape[:2]
        central_region = face_region[h//4:3*h//4, w//4:3*w//4]
        
        # Convert to different color spaces for analysis
        hsv = cv2.cvtColor(central_region, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(central_region, cv2.COLOR_BGR2LAB)
        
        # Get mean colors
        mean_bgr = np.mean(central_region.reshape(-1, 3), axis=0)
        mean_hsv = np.mean(hsv.reshape(-1, 3), axis=0)
        mean_lab = np.mean(lab.reshape(-1, 3), axis=0)
        
        return {
            'rgb': [int(mean_bgr[2]), int(mean_bgr[1]), int(mean_bgr[0])],  # BGR to RGB
            'hsv': [int(mean_hsv[0]), int(mean_hsv[1]), int(mean_hsv[2])],
            'lab': [int(mean_lab[0]), int(mean_lab[1]), int(mean_lab[2])]
        }
    
    def get_under_eye_color(self, face_region):
        """Extract under-eye area color for undertone analysis"""
        h, w = face_region.shape[:2]
        under_eye_region = face_region[h//3:h//2, w//4:3*w//4]
        mean_color = np.mean(under_eye_region.reshape(-1, 3), axis=0)
        return [int(mean_color[2]), int(mean_color[1]), int(mean_color[0])]  # BGR to RGB
    
    def get_lip_color(self, face_region):
        """Extract lip area color"""
        h, w = face_region.shape[:2]
        lip_region = face_region[2*h//3:4*h//5, w//3:2*w//3]
        mean_color = np.mean(lip_region.reshape(-1, 3), axis=0)
        return [int(mean_color[2]), int(mean_color[1]), int(mean_color[0])]  # BGR to RGB
    
    def analyze_undertone(self, face_region):
        """Analyze overall undertone (warm/cool/neutral)"""
        lab = cv2.cvtColor(face_region, cv2.COLOR_BGR2LAB)
        a_channel = lab[:, :, 1]  # Green-Red axis
        b_channel = lab[:, :, 2]  # Blue-Yellow axis
        
        mean_a = np.mean(a_channel)
        mean_b = np.mean(b_channel)
        
        # Determine undertone based on LAB values
        if mean_b > 130:  # More yellow
            undertone = "warm"
        elif mean_b < 125:  # More blue
            undertone = "cool"
        else:
            undertone = "neutral"
        
        return {
            'type': undertone,
            'a_value': float(mean_a),
            'b_value': float(mean_b)
        }
    
    def analyze_with_claude(self, color_data):
        """Send color data to Claude API for analysis"""
        if not self.api_key:
            return "API key not configured."
        
        prompt = f"""
        I need you to analyze facial colors for personalized color analysis. Based on the following extracted facial color data, please provide a comprehensive color palette recommendation including seasonal color analysis.

        Facial Color Data:
        - Skin tone RGB: {color_data['skin_tone']['rgb']}
        - Skin tone HSV: {color_data['skin_tone']['hsv']}
        - Skin tone LAB: {color_data['skin_tone']['lab']}
        - Under-eye area RGB: {color_data['under_eye']}
        - Lip area RGB: {color_data['lip_area']}
        - Undertone analysis: {color_data['overall_undertone']['type']} (LAB a*: {color_data['overall_undertone']['a_value']:.1f}, b*: {color_data['overall_undertone']['b_value']:.1f})

        Please provide:
        1. Seasonal color analysis (Spring, Summer, Autumn, Winter) with explanation
        2. Best colors for clothing (specific color names and hex codes if possible)
        3. Colors to avoid
        4. Makeup recommendations (foundation undertone, lip colors, eye colors)
        5. Hair color suggestions that would complement this coloring
        6. Confidence level of your analysis (1-10) and any limitations

        Format your response clearly with sections for each recommendation type.
        """
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        
        data = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 2000,
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
    
    def display_color_analysis(self, colors):
        """Display extracted colors visually on canvas"""
        self.color_canvas.delete("all")
        
        # Display skin tone
        skin_rgb = colors['skin_tone']['rgb']
        skin_hex = f"#{skin_rgb[0]:02x}{skin_rgb[1]:02x}{skin_rgb[2]:02x}"
        self.color_canvas.create_rectangle(10, 10, 80, 60, fill=skin_hex, outline="black")
        self.color_canvas.create_text(45, 70, text="Skin Tone", font=("Arial", 8))
        
        # Display under-eye color
        under_eye_rgb = colors['under_eye']
        under_eye_hex = f"#{under_eye_rgb[0]:02x}{under_eye_rgb[1]:02x}{under_eye_rgb[2]:02x}"
        self.color_canvas.create_rectangle(90, 10, 160, 60, fill=under_eye_hex, outline="black")
        self.color_canvas.create_text(125, 70, text="Under-eye", font=("Arial", 8))
        
        # Display lip color
        lip_rgb = colors['lip_area']
        lip_hex = f"#{lip_rgb[0]:02x}{lip_rgb[1]:02x}{lip_rgb[2]:02x}"
        self.color_canvas.create_rectangle(170, 10, 240, 60, fill=lip_hex, outline="black")
        self.color_canvas.create_text(205, 70, text="Lip Area", font=("Arial", 8))
        
        # Display undertone info
        undertone = colors['overall_undertone']['type']
        undertone_color = {"warm": "#FFD700", "cool": "#87CEEB", "neutral": "#DDA0DD"}
        self.color_canvas.create_rectangle(10, 90, 240, 140, 
                                         fill=undertone_color.get(undertone, "#CCCCCC"), 
                                         outline="black")
        self.color_canvas.create_text(125, 115, text=f"Undertone: {undertone.title()}", 
                                    font=("Arial", 12, "bold"))
        
        # Display RGB values
        self.color_canvas.create_text(125, 160, 
                                    text=f"Skin RGB: {skin_rgb}\nUnder-eye RGB: {under_eye_rgb}\nLip RGB: {lip_rgb}", 
                                    font=("Arial", 9), justify=tk.CENTER)
    
    def capture_and_analyze(self):
        """Capture current frame and perform color analysis"""
        if self.current_frame is None:
            messagebox.showerror("Error", "No frame available for analysis")
            return
        
        self.status_var.set("Analyzing facial colors...")
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Processing...\n")
        
        # Run analysis in separate thread to avoid blocking GUI
        thread = threading.Thread(target=self._analyze_thread)
        thread.daemon = True
        thread.start()
    
    def _analyze_thread(self):
        """Thread function for analysis"""
        try:
            # Extract colors from current frame
            colors, message = self.extract_facial_colors(self.current_frame)
            
            if colors is None:
                self.root.after(0, lambda: self._update_results(f"Analysis failed: {message}"))
                return
            
            # Display colors visually
            self.root.after(0, lambda: self.display_color_analysis(colors))
            
            # Analyze with Claude
            analysis_result = self.analyze_with_claude(colors)
            
            # Format results
            final_result = f"FACIAL COLOR ANALYSIS RESULTS\n{'='*50}\n\n"
            final_result += f"Extracted Colors:\n"
            final_result += f"- Skin Tone RGB: {colors['skin_tone']['rgb']}\n"
            final_result += f"- Under-eye RGB: {colors['under_eye']}\n"
            final_result += f"- Lip Area RGB: {colors['lip_area']}\n"
            final_result += f"- Undertone: {colors['overall_undertone']['type']}\n\n"
            final_result += f"CLAUDE AI ANALYSIS:\n{'='*30}\n\n"
            final_result += analysis_result
            
            self.root.after(0, lambda: self._update_results(final_result))
            
        except Exception as e:
            self.root.after(0, lambda: self._update_results(f"Error during analysis: {str(e)}"))
    
    def _update_results(self, text):
        """Update results in main thread"""
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, text)
        self.status_var.set("Analysis complete")
    
    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def on_closing(self):
        self.stop_camera()
        self.root.destroy()

if __name__ == "__main__":
    app = FacialColorAnalyzer()
    app.run()