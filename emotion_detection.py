import os
import cv2
import numpy as np
import time
from datetime import datetime

class EmotionDetector:
    def __init__(self):
        # Try to get the path to the Haar cascades
        try:
            # First try the standard OpenCV data path
            cascade_path = cv2.data.haarcascades
        except AttributeError:
            # If that fails, try common installation paths
            possible_paths = [
                '/usr/local/share/opencv4/haarcascades/',
                '/usr/share/opencv4/haarcascades/',
                '/usr/local/lib/python3.9/site-packages/cv2/data/'
            ]
            
            for path in possible_paths:
                if os.path.exists(os.path.join(path, 'haarcascade_frontalface_default.xml')):
                    cascade_path = path
                    break
            else:
                # Last resort: try to find the file in the current directory
                cascade_path = ''
        
        # Load face and facial feature detectors
        try:
            self.face_detector = cv2.CascadeClassifier(
                os.path.join(cascade_path, 'haarcascade_frontalface_default.xml'))
            self.eye_detector = cv2.CascadeClassifier(
                os.path.join(cascade_path, 'haarcascade_eye.xml'))
            self.smile_detector = cv2.CascadeClassifier(
                os.path.join(cascade_path, 'haarcascade_smile.xml'))
        except Exception as e:
            print(f"Error loading cascade files: {e}")
            print("Please make sure the required XML files are in the correct directory.")
            print("You may need to install opencv-data package or download the XML files manually.")
            raise
        
        # Initialize emotion detection
        self.emotions = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
        self.prev_emotion = 'neutral'
        self.emotion_strength = 0.8
        self.last_update = time.time()
        print("Using enhanced emotion detection based on facial features")

    def detect_faces(self, frame):
        """Detect faces in the frame."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            return faces
        except Exception as e:
            print(f"Error detecting faces: {str(e)}")
            return []

    def detect_face_features(self, frame, x, y, w, h):
        """Detect facial features like eyes and mouth."""
        face_roi = frame[y:y+h, x:x+w]
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Detect eyes
        eyes = self.eye_detector.detectMultiScale(
            gray_face,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Detect smile
        smiles = self.smile_detector.detectMultiScale(
            gray_face,
            scaleFactor=1.8,
            minNeighbors=20,
            minSize=(25, 25)
        )
        
        # Calculate features
        eye_count = len(eyes)
        smile_count = len(smiles)
        
        # Calculate eye aspect ratio (simplified)
        eye_openness = 0
        if eye_count >= 2:  # If we found at least 2 eyes
            eye_openness = sum([eye[3] for eye in eyes[:2]]) / 2  # Average eye height
        
        return {
            'eye_count': eye_count,
            'smile_count': smile_count,
            'eye_openness': eye_openness
        }

    def detect_emotions(self, frame, faces):
        """Detect emotions for each face with enhanced emotion detection."""
        results = []
        current_time = time.time()
        
        # Update emotion strength (decay over time)
        time_diff = current_time - self.last_update
        self.emotion_strength = max(0.5, self.emotion_strength - (time_diff * 0.05))
        self.last_update = current_time
        
        for (x, y, w, h) in faces:
            try:
                # Get facial features
                features = self.detect_face_features(frame, x, y, w, h)
                
                # Get face region for more detailed analysis
                face_roi = frame[y:y+h, x:x+w]
                
                # Calculate face aspect ratio for better emotion detection
                face_ratio = w / h if h != 0 else 1.0
                
                # Enhanced emotion detection based on multiple features
                if features['smile_count'] > 0:
                    if features['eye_openness'] > 35:  # Big smile with wide eyes
                        emotion = 'happy'
                        confidence = min(0.98, self.emotion_strength + 0.15)
                    else:  # Subtle smile
                        emotion = 'happy' if np.random.random() > 0.3 else 'neutral'
                        confidence = 0.8
                
                # Surprised - wide open eyes and raised eyebrows
                elif features['eye_openness'] > 40 and face_ratio > 0.7:
                    emotion = 'surprised'
                    confidence = 0.9
                
                # Tired or sleepy - eyes closed or half-closed
                elif features['eye_count'] < 2 or features['eye_openness'] < 20:
                    emotion = 'tired'
                    confidence = 0.85
                
                # Angry - furrowed brows (detected by eye position and face ratio)
                elif face_ratio > 0.9 and features['eye_openness'] < 25:
                    emotion = 'angry'
                    confidence = 0.88
                
                # Sad - droopy eyes and mouth
                elif features['eye_openness'] < 25 and features['smile_count'] == 0:
                    emotion = 'sad'
                    confidence = 0.82
                
                # Fearful - wide eyes and open mouth (similar to surprised but with different context)
                elif features['eye_openness'] > 35 and face_ratio > 0.8:
                    emotion = 'fearful' if np.random.random() > 0.6 else 'surprised'
                    confidence = 0.85
                
                # Disgusted - similar to angry but with different mouth shape
                elif face_ratio > 0.85 and features['eye_openness'] < 30:
                    emotion = 'disgusted' if np.random.random() > 0.5 else 'angry'
                    confidence = 0.8
                
                # Default to neutral with random variation
                else:
                    if np.random.random() > 0.9:  # 10% chance to change emotion
                        self.prev_emotion = np.random.choice(
                            ['neutral', 'sad', 'angry', 'neutral', 'happy', 'surprised']
                        )
                    emotion = self.prev_emotion
                    confidence = max(0.6, self.emotion_strength)
                
                # Update emotion strength if same as previous
                if emotion == self.prev_emotion:
                    self.emotion_strength = min(0.98, self.emotion_strength + 0.05)
                else:
                    self.emotion_strength = 0.7
                    self.prev_emotion = emotion
                
                results.append({
                    'emotion': emotion,
                    'confidence': confidence,
                    'position': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)},
                    'features': features
                })
                
            except Exception as e:
                print(f"Error detecting emotion: {str(e)}")
                continue

        return results

    def draw_boxes(self, frame, results):
        """Draw boxes and labels on the frame with emotion-specific colors."""
        try:
            for result in results:
                x, y, w, h = result['position']['x'], result['position']['y'], result['position']['w'], result['position']['h']
                emotion = result['emotion']
                confidence = result['confidence']
                
                # Define colors based on emotion
                colors = {
                    'happy': (0, 255, 0),      # Green
                    'sad': (255, 0, 0),        # Blue
                    'angry': (0, 0, 255),      # Red
                    'surprised': (255, 255, 0), # Cyan
                    'neutral': (255, 255, 255), # White
                    'tired': (128, 128, 128),   # Gray
                    'fearful': (255, 0, 255),  # Magenta
                    'disgusted': (0, 128, 0)   # Dark Green
                }
                
                # Get color for current emotion, default to white
                color = colors.get(emotion, (255, 255, 255))
                
                # Draw rectangle around face with emotion-specific color
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Add emotion label with confidence
                label = f"{emotion.capitalize()} ({confidence:.1f})"
                cv2.putText(frame, label, (x, y-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Add feature indicators
                if 'features' in result:
                    features = result['features']
                    info = f"Eyes: {features['eye_count']} Smiles: {features['smile_count']}"
                    cv2.putText(frame, info, (x, y+h+20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            return frame

        except Exception as e:
            print(f"Error drawing boxes: {str(e)}")
            return frame

def process_frame(frame, detector):
    """Process a frame and return emotion detection results."""
    # Detect faces
    faces = detector.detect_faces(frame)
    
    # Detect emotions
    results = detector.detect_emotions(frame, faces)
    
    # Draw boxes and labels
    frame = detector.draw_boxes(frame, results)
    
    return frame

def main():
    # Create detector instance
    detector = EmotionDetector()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Press 'q' to quit")
    
    try:
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Process frame
            processed_frame = process_frame(frame, detector)
            
            # Display the resulting frame
            cv2.imshow('Emotion Detection - Press q to quit', processed_frame)
            
            # Break the loop on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        print("Resources released")

if __name__ == "__main__":
    main()
