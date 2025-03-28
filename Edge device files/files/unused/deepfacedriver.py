from deepface import DeepFace
import cv2
import time


if __name__ == "__main__":
  models = [
  "VGG-Face", 
  "Facenet", 
  "Facenet512", 
  "OpenFace", 
  "DeepFace", 
  "DeepID", 
  "ArcFace", 
  "Dlib", 
  "SFace",
  "GhostFaceNet",
  "Buffalo_L" 
  ]
  while True:
    cap = cv2.VideoCapture(0) 
    
    
    ret, frame = cap.read()  # Capture frame-by-frame
    if ret:
        # Save the frame as an image file
        cv2.imwrite("authusers/userimage.jpg", frame)
        print('Photo taken')
    else:
        print('Failed to capture image')
    cap.release()
    
    try:
      # Use DeepFace to detect face
      face = DeepFace.detectFace("authusers/userimage.jpg", enforce_detection=True)
      print("Face detected! Proceeding with DeepFace analysis.")
      
      # Run DeepFace verification or other analysis (e.g., comparing images)
      result = DeepFace.verify(img1_path="authusers/imagekev2.jpg", img2_path="authusers/userimage.jpg",model_name = models[0],threshold = 0.68)
      print(result)
      
    except ValueError as e:
      # If no face is detected, catch the error and print a message
      print("No face detected in the image. Skipping DeepFace analysis.")
      

    time.sleep(1)
