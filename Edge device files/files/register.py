from pymongo import MongoClient
import takephoto
import cv2
import tensorflow as tf
from cameradriver import get_embedding 
import os
from videostream import VideoStream
#import audiodrivermelspec as audiodriver
import audiodriver



def register_new_user(usn):
    # Connect to MongoDB (local instance)
    client = MongoClient('mongodb://localhost:27017/')

    # Access the 'test' database
    db = client['project']
    #cap = cv2.VideoCapture(0)
    video_stream=VideoStream()
    image_embeddings = []
    audio_embeddings =[]
    # Load the TFLite model interpreter
    interpreter = tf.lite.Interpreter(model_path="facenet.tflite")
    #interpreter.set_num_threads(4)
    interpreter.allocate_tensors()
    
    if usn == "":
        registered_username = input("Enter username:")
    else:
        registered_username = usn
    
    print("start voiceprint registration: say something")
    audio_embeddings = audiodriver.embed_voice(3)
    
    audiodriver.stop_listening()
    
    print("start face registration:")
    folder_path = f"authusers/{registered_username}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for i in range(1, 4):
        print(f"press s to capture image {i}")
        while True:
            
            # Capture frame-by-frame
            #ret, frame = cap.read()
            ret, frame = video_stream.read()
            
            if not ret:
                print("Failed to grab frame")
                break
                
            displayframe = cv2.resize(frame, (320, 240))

            # Display the resulting frame in a window
            cv2.imshow("Live Camera Feed", displayframe)
            
            # Wait for key press: 's' to capture an image, 'q' to quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                embedding = get_embedding(frame,interpreter)  # Assuming this function processes the image for embeddings
                if embedding is None:
                    print("please retake photo, no face detected")
                    continue
                image_embeddings.append(embedding)
                
                # Save the frame to the specified folder
                image_path = os.path.join(folder_path, f"image{i}.jpg")
                cv2.imwrite(image_path, frame)  # Save the image
                print(f"Image {i} saved to {image_path}")
                break
           
            


    
    #cap.release()
    video_stream.stop()
    cv2.destroyAllWindows()
    
    image_embeddings = [emb.tolist() for emb in image_embeddings]
    audio_embeddings = [emb.tolist() for emb in audio_embeddings]
  

    print("Images taken, inserting...")

    # Insert a document
    users_collection = db['users']
    users_collection.replace_one(
        {"username": registered_username},  # Filter for the username
        {"username": registered_username , "audio_embeddings": audio_embeddings ,"image_embeddings": image_embeddings},# New document to insert or replace
        upsert=True               # If no document matches, insert a new one
    )

    print("Data inserted!")

    # Query the collection
    user = users_collection.find_one({"username": registered_username})
    print(user)
    print(len(audio_embeddings))
    client.close()

def register_new_user_old(usn=""):
    # Connect to MongoDB (local instance)
    client = MongoClient('mongodb://localhost:27017/')

    # Access the 'test' database
    db = client['project']
    #cap = cv2.VideoCapture(0)
    video_stream=VideoStream()
    image_embeddings = []
    audio_embeddings =[]
    # Load the TFLite model interpreter
    interpreter = tf.lite.Interpreter(model_path="facenet.tflite")
    #interpreter.set_num_threads(4)
    interpreter.allocate_tensors()
    
    if usn == "":
        username = input("Enter username:")
    else:
        username = usn
    
    print("start voiceprint registration: say something")
    audio_embeddings = audiodriver.embed_voice(3)
    
    audiodriver.stop_listening()
    
    print("start face registration:")
    folder_path = f"authusers/{username}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for i in range(1, 4):
        print(f"press s to capture image {i}")
        while True:
            
            # Capture frame-by-frame
            #ret, frame = cap.read()
            ret, frame = video_stream.read()
            
            if not ret:
                print("Failed to grab frame")
                break
                
            displayframe = cv2.resize(frame, (320, 240))

            # Display the resulting frame in a window
            cv2.imshow("Live Camera Feed", displayframe)
            
            # Wait for key press: 's' to capture an image, 'q' to quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                embedding = get_embedding(frame,interpreter)  # Assuming this function processes the image for embeddings
                if embedding is None:
                    print("please retake photo, no face detected")
                    continue
                image_embeddings.append(embedding)
                
                # Save the frame to the specified folder
                image_path = os.path.join(folder_path, f"image{i}.jpg")
                cv2.imwrite(image_path, frame)  # Save the image
                print(f"Image {i} saved to {image_path}")
                break
           
            


    
    #cap.release()
    video_stream.stop()
    cv2.destroyAllWindows()
    
    image_embeddings = [emb.tolist() for emb in image_embeddings]
    audio_embeddings = [emb.tolist() for emb in audio_embeddings]
  

    print("Images taken, inserting...")

    # Insert a document
    users_collection = db['users']
    users_collection.replace_one(
        {"username": username},  # Filter for the username
        {"username": username , "audio_embeddings": audio_embeddings ,"image_embeddings": image_embeddings},# New document to insert or replace
        upsert=True               # If no document matches, insert a new one
    )

    print("Data inserted!")

    # Query the collection
    user = users_collection.find_one({"username": username})
    print(user)
    print(len(audio_embeddings))
    client.close()


if __name__ == "__main__":
    register_new_user_old()
