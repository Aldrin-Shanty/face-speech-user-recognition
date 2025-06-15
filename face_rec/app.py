import cv2
import numpy as np
import dlib
import pickle
import os
from collections import defaultdict
import time

class FaceRecognitionSystem:
    def __init__(self, yolo_weights_path, yolo_config_path, known_faces_dir="known_faces"):

        # Initialize YOLO
        print("Loading YOLO model...")
        self.net = cv2.dnn.readNet(yolo_weights_path, yolo_config_path)
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
        # Initialize dlib
        print("Loading dlib models...")
        self.face_detector = dlib.get_frontal_face_detector()
        
        # Load dlib models
        try:
            self.shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
            self.face_encoder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
        except RuntimeError as e:
            print(f"Error loading dlib models: {e}")
            print("Make sure you have downloaded:")
            print("- shape_predictor_68_face_landmarks.dat")
            print("- dlib_face_recognition_resnet_model_v1.dat")
            raise
        
        # Known faces database
        self.known_faces_dir = known_faces_dir
        self.known_face_encodings = []
        self.known_face_names = []
        
        # Load known faces
        self.load_known_faces()
        
    def detect_faces_yolo(self, frame):

        height, width = frame.shape[:2]
    
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        
        boxes = []
        confidences = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if class_id == 0 and confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))

        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            
            face_regions = []
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]

                    face_y = max(0, y)
                    face_h = int(h * 0.4)
                    face_x = max(0, x)
                    face_w = w
                    
                    face_regions.append((face_x, face_y, face_w, face_h))
                    
            return face_regions
        
        return []
    
    def extract_face_encodings_dlib(self, frame, face_regions):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_data = []
        
        for (x, y, w, h) in face_regions:
            # Extract region of interest
            roi_gray = gray[y:y+h, x:x+w]
            
            # Detect faces within the region using dlib
            faces = self.face_detector(roi_gray)
            
            for face in faces:
                # Adjust coordinates to full image
                face_x = face.left() + x
                face_y = face.top() + y
                face_w = face.width()
                face_h = face.height()
                
                # Get facial landmarks
                shape = self.shape_predictor(gray, dlib.rectangle(face_x, face_y, face_x + face_w, face_y + face_h))
                
                # Extract face encoding - dlib needs BGR image, not grayscale
                face_encoding = np.array(self.face_encoder.compute_face_descriptor(frame, shape))
                
                face_data.append((face_encoding, (face_x, face_y, face_w, face_h)))
        
        return face_data
    
    def load_known_faces(self):

        if not os.path.exists(self.known_faces_dir):
            os.makedirs(self.known_faces_dir)
            print(f"Created directory: {self.known_faces_dir}")
            print("Add face images to this directory and restart the program")
            return
        
        print("Loading known faces...")
        
        for filename in os.listdir(self.known_faces_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Load image
                image_path = os.path.join(self.known_faces_dir, filename)
                image = cv2.imread(image_path)
                
                if image is not None:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    
                    # Detect faces
                    faces = self.face_detector(gray)
                    
                    if faces:
                        # Use the first face found
                        face = faces[0]
                        
                        # Get facial landmarks
                        shape = self.shape_predictor(gray, face)
                        
                        # Extract face encoding - dlib needs BGR image, not grayscale
                        face_encoding = np.array(self.face_encoder.compute_face_descriptor(image, shape))
                        
                        # Extract name from filename (remove extension)
                        name = os.path.splitext(filename)[0]
                        
                        self.known_face_encodings.append(face_encoding)
                        self.known_face_names.append(name)
                        
                        print(f"Loaded face for: {name}")
                    else:
                        print(f"No face found in: {filename}")
        
        print(f"Loaded {len(self.known_face_encodings)} known faces")
    
    def euclidean_distance(self, encoding1, encoding2):

        return np.linalg.norm(encoding1 - encoding2)
    
    def recognize_faces(self, face_encodings, threshold=0.6):
        face_names = []
        
        for face_encoding in face_encodings:
            name = "Unknown"
            min_distance = float('inf')
            
            # Compare with all known faces
            for i, known_encoding in enumerate(self.known_face_encodings):
                distance = self.euclidean_distance(face_encoding, known_encoding)
                
                if distance < min_distance:
                    min_distance = distance
                    if distance < threshold:
                        name = self.known_face_names[i]
            
            face_names.append((name, min_distance))
        
        return face_names
    
    def process_frame(self, frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray)
        
        print(f"Direct dlib detected {len(faces)} faces")  
        
        if faces:
            face_data = []
            for face in faces:
                face_x, face_y = face.left(), face.top()
                face_w, face_h = face.width(), face.height()
                
                # Get facial landmarks
                shape = self.shape_predictor(gray, face)
                
                # Extract face encoding
                face_encoding = np.array(self.face_encoder.compute_face_descriptor(frame, shape))
                face_data.append((face_encoding, (face_x, face_y, face_w, face_h)))
            
            # Separate encodings and locations
            face_encodings = [data[0] for data in face_data]
            face_locations = [data[1] for data in face_data]
            
            # Recognize faces
            recognition_results = self.recognize_faces(face_encodings)
            
            # Draw results
            for i, (face_x, face_y, face_w, face_h) in enumerate(face_locations):
                if i < len(recognition_results):
                    name, confidence = recognition_results[i]
                    
                    # Draw bounding box
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (face_x, face_y), (face_x + face_w, face_y + face_h), color, 2)
                    
                    # Draw name and confidence
                    label = f"{name} ({confidence:.2f})"
                    cv2.rectangle(frame, (face_x, face_y - 35), (face_x + face_w, face_y), color, cv2.FILLED)
                    cv2.putText(frame, label, (face_x + 6, face_y - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        else:
            face_regions = self.detect_faces_yolo(frame)
            print(f"YOLO detected {len(face_regions)} person regions")  

            face_data = self.extract_face_encodings_dlib(frame, face_regions)
            print(f"dlib found {len(face_data)} faces in YOLO regions") 
            
            if face_data:

                face_encodings = [data[0] for data in face_data]
                face_locations = [data[1] for data in face_data]

                recognition_results = self.recognize_faces(face_encodings)
                
                for i, (face_x, face_y, face_w, face_h) in enumerate(face_locations):
                    if i < len(recognition_results):
                        name, confidence = recognition_results[i]
                        
                        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                        cv2.rectangle(frame, (face_x, face_y), (face_x + face_w, face_y + face_h), color, 2)
                        
                        label = f"{name} ({confidence:.2f})"
                        cv2.rectangle(frame, (face_x, face_y - 35), (face_x + face_w, face_y), color, cv2.FILLED)
                        cv2.putText(frame, label, (face_x + 6, face_y - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        return frame
    
    def run_video_recognition(self, video_source=0):
       
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print("Error: Could not open video source")
            return
        
        print("Starting face recognition... Press 'q' to quit")
        print("Debug mode enabled - check console for detection info")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            

            processed_frame = self.process_frame(frame)
            
            frame_count += 1
            if frame_count % 30 == 0:
                end_time = time.time()
                fps = 30 / (end_time - start_time)
                print(f"FPS: {fps:.2f}")
                start_time = time.time()
            

            cv2.imshow('Face Recognition - YOLO + dlib (Debug Mode)', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def add_known_face_from_camera(self, name):
       
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print(f"Position your face in the frame and press SPACE to capture for {name}")
        print("Press ESC to cancel")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
          
            cv2.imshow('Capture Face - Press SPACE to capture, ESC to cancel', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 32:  
               
                output_path = os.path.join(self.known_faces_dir, f"{name}.jpg")
                cv2.imwrite(output_path, frame)
                
             
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_detector(gray)
                
                if faces:
                    face = faces[0]
                    shape = self.shape_predictor(gray, face)
                   
                    face_encoding = np.array(self.face_encoder.compute_face_descriptor(frame, shape))
                    
                    self.known_face_encodings.append(face_encoding)
                    self.known_face_names.append(name)
                    
                    print(f"Successfully added {name} to known faces!")
                else:
                    print("No face detected in captured image")
                break
            elif key == 27: 
                print("Capture cancelled")
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    """
    Main function to run the face recognition system
    """
  
    yolo_weights = "yolov3.weights"  
    yolo_config = "yolov3.cfg"      
    
    
    required_files = [
        yolo_weights,
        yolo_config,
        "shape_predictor_68_face_landmarks.dat",
        "dlib_face_recognition_resnet_model_v1.dat"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("Missing required files:")
        for file in missing_files:
            print(f"- {file}")
        print("\nPlease download the required model files:")
        print("1. YOLO: yolov3.weights and yolov3.cfg from https://pjreddie.com/darknet/yolo/")
        print("2. dlib: shape_predictor_68_face_landmarks.dat and dlib_face_recognition_resnet_model_v1.dat")
        print("   from http://dlib.net/files/")
        return
    
    try:
       
        fr_system = FaceRecognitionSystem(yolo_weights, yolo_config)

        while True:
            print("\nFace Recognition System")
            print("1. Run video recognition")
            print("2. Add known face from camera")
            print("3. Exit")
            
            choice = input("Enter your choice (1-3): ").strip()
            
            if choice == '1':
                fr_system.run_video_recognition()
            elif choice == '2':
                name = input("Enter person's name: ").strip()
                if name:
                    fr_system.add_known_face_from_camera(name)
                else:
                    print("Invalid name")
            elif choice == '3':
                break
            else:
                print("Invalid choice")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()