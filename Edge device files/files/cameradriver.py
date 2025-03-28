from deepface import DeepFace
import cv2
import time
import numpy as np
import tensorflow as tf
from deepface import DeepFace
from pymongo import MongoClient
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from videostream import VideoStream
from multiprocessing import Queue, Process
from queue import Empty 
import takephoto


# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["project"]
users_collection = db["users"]  # Collection storing embeddings

event_queue = None

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
#(112,112) for mobilefacenet
def preprocess_image(image, target_size=(160, 160)):
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Check if face detection returned a valid face
        if faces is None or len(faces) == 0:
            print("no face detected")
            raise ValueError("No face detected")
        
        
        x, y, w, h = faces[0]
    
        # Crop the face from the image
        face_cropped = image[y:y+h, x:x+w]
        cv2.imwrite('cropped.jpg', face_cropped )
        # Resize face to 112x112
        face_resized = cv2.resize(face_cropped, target_size)
        cv2.imwrite('resized.jpg', face_resized )

        # Normalize image: MobileNetV2 typically expects pixel values in [0, 1]
        face_resized = face_resized.astype(np.float32) / 255.0

        # Convert to batch dimension (add batch size dimension)
        final_img = np.expand_dims(face_resized, axis=0)
        #final_img = np.expand_dims(face_resized.astype(np.float32), axis=0)



        return final_img

    except ValueError as e:
        print(f"Error in preprocess_image: {e}")
        return None  # Return None so calling function knows to retry
        

def average_embeddings(embeddings):
    # Take the average of all embeddings in the list
    return np.mean(embeddings, axis=0)

def compare_embeddings(test_embedding, registered_embedding):
    """Compare a test embedding with a registered embedding using cosine similarity."""
    # Ensure embeddings are not None and are of valid shape
    if test_embedding is None or registered_embedding is None:
        return float("inf")  # Return infinite distance if any embedding is invalid

    # Calculate the cosine similarity between the test image embedding and the registered embedding
    similarity_score = 1 - cosine(test_embedding, registered_embedding)
    return similarity_score
    
def compare_with_minimum_distance(test_embedding, registration_embeddings):
    # Find the minimum cosine distance between the test embedding and each registration embedding
    distances = [1 - cosine(test_embedding, reg_emb) for reg_emb in registration_embeddings]
    min_distance = min(distances)
    return min_distance


def get_embedding(image,interpreter):
    
    
    cv2.imwrite('embed.jpg', image)
    # Get input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    """ Extract face embeddings using MobileFaceNet """
    try:
        img = preprocess_image(image)
        if img is None:
            return None
    except Exception as e:
        raise ValueError(f"Error in preprocess_image: {str(e)}")
    
    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], img)

    # Run inference
    interpreter.invoke()

    # Get the embedding from the output tensor
    embedding = interpreter.get_tensor(output_details[0]['index'])

    # Flatten the embedding to 1D array
    return embedding.flatten()


def verify_faces(embedding1, embedding2, threshold=0.5):
    """ Compare two face embeddings using cosine similarity """
    
    # Use DeepFace's cosine similarity function
    similarity_score = 1 - cosine(emb1, stored_emb)
    
    is_match = similarity_score > threshold  # Lower score means more similar
    return {"verified": is_match, "distance": similarity_score}
    
    
def recognise_face(video_stream,username="Kevin",threshold=0.8,queue=None):
    global event_queue
    
    #video_stream = VideoStream()
    
    event_queue = queue
    
    
    #cap = cv2.VideoCapture(0) 
    #cap.set(cv2.CAP_PROP_FPS, 21)  # Lower FPS (e.g., 10 instead of 30)

    failcount=0
    
    #interpreter = tf.lite.Interpreter(model_path="mobilefacenet.tflite")
    interpreter = tf.lite.Interpreter(model_path="facenet.tflite")
    #interpreter.set_num_threads(4)
    interpreter.allocate_tensors()
    print(f"press s to capture image")
    while True:
        
        if not event_queue.empty():
            msg = event_queue.get(timeout=2)
            if msg == "accelerometer_movement" or msg == "reset_signal":
                print("Intrusion detected! Locking the system.")
                return False
            elif msg == "wake_signal":
                pass

       
        
        if failcount>5:
            #cap.release()
            video_stream.stop()
            return False
        #ret, frame = cap.read()  # Capture frame-by-frame
        ret, frame = video_stream.read()
        
        if ret:
            displayframe = cv2.resize(frame, (320, 240))
            cv2.imshow("Live Camera Feed", displayframe)
            try:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    embedding = get_embedding(frame,interpreter)
                    if embedding is None:
                        print("please retake photo, no face detected")
                        continue
                    result = verify_user(embedding,username,threshold)

                    print(result)
                    if bool(result['verified']) is True:
                        print(result)
                        print(f"user {result['match']} verified")
                        #cap.release()
                        #video_stream.stop()
                        cv2.destroyAllWindows()
                        return True
                    else:
                        failcount+=1
        
        
            except ValueError as e:
                raise ValueError(f"Error in get_embedding: {str(e)}")
        else:
            print('Failed to capture image')
            
        
        #time.sleep(1)
        


def search_face_in_db(img_path, threshold=0.5):
    """Searches for the closest match in MongoDB"""
    query_emb = np.array(get_embedding(img_path))

    best_match = None
    best_score = float("inf")  # Cosine distance (lower = more similar)

    # Fetch all user embeddings from MongoDB
    users = users_collection.find({})
    
    for user in users:
        username = user["username"]
        stored_embeddings = user["embeddings"]  # List of stored embeddings per user

        for stored_emb in stored_embeddings:
            stored_emb = np.array(stored_emb)
            similarity_score = DeepFace.findCosineDistance(query_emb, stored_emb)

            if similarity_score < best_score:
                best_score = similarity_score
                best_match = username

    is_match = best_score < threshold
    return {"verified": is_match, "match": best_match if is_match else None, "distance": best_score}
        
def verify_user(emb1,username, threshold=0.8):


    best_match = None
    best_score = float("-inf")  # Cosine distance (lower = more similar)

    # Fetch all user embeddings from MongoDB
    user_cursor = users_collection.find({"username": username}).limit(1)

    # Get the first user (or None if no user is found)
    user = next(user_cursor, None)
    
    if user:
        stored_embeddings = user["image_embeddings"]  # List of stored embeddings per user
        stored_embeddings = [np.array(emb) for emb in stored_embeddings]

        #average_embedding = average_embeddings(stored_embeddings)
        #similarity = compare_embeddings(emb1, average_embedding)
        
        # Compare the incoming embedding (emb1) against each stored embedding
        for stored_emb in stored_embeddings:
            similarity_score = compare_embeddings(emb1, stored_emb)
            print(f"Comparing with stored embedding: similarity_score = {similarity_score}")

            if similarity_score > best_score:
                best_score = similarity_score

    
        
        # Check if the best match meets the threshold
        is_match = best_score > threshold
        if is_match:
            return {"verified": is_match, "match": username, "similarity": best_score}
        else:
            return {"verified": is_match, "match": None, "similarity": best_score}


        




if __name__ == "__main__":
    video_stream=VideoStream()
    recognise_face(video_stream)
    video_stream.stop()
