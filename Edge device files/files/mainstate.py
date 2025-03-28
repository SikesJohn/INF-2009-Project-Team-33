import time
import random
from multiprocessing import Queue, Process, Value , Manager
#import audiodrivermelspec as audiodriver
import audiodriver as audiodriver
import radardriver
import cameradriver
import takephoto
from videostream import VideoStream
import acceldriver
import tensorflow as tf
import cv2
import register
import mqtt_publisher


class AuthState:
    IDLE = 'IDLE'
    RADAR_DETECTION = 'RADAR_DETECTION'
    VOICE_AUTH = 'VOICE_AUTH'
    FACE_AUTH = 'FACE_AUTH'
    AUTHENTICATED = 'AUTHENTICATED'


#current_state = AuthState.RADAR_DETECTION
current_state = AuthState.RADAR_DETECTION
username=""
registered_username = ""
video_stream = None

# Global variable for radar process
radar_process = None  # Define radar_process globally, outside of state machine
accel_proc = None
signal = ""

publisher = mqtt_publisher.mqtt_pub()

# Start the background processes for radar and accelerometer







# def radar_detection_process(person_in_view: Value):
#     while True:
#         # Simulate radar logic here
#         time.sleep(1)
#         movement_detected = random.choice([True, False])  # Random detection simulation
#         if movement_detected:
#             person_in_view.value = 1  # Simulate person entering the field of view
#             queue.put("radar_movement")  # Put event in the queue
#             print("Radar: Movement detected.")
#         else:
#             if person_in_view.value == 1:
#                 person_in_view.value = 0  # Simulate person leaving the field of view
#                 queue.put("radar_leave")  # Put event in the queue
#                 print("Radar: Person left the field of view.")

# Transition to RADAR_DETECTION state (just wait for radar events now)

def set_lock_state(new_state):
    """Updates the lock state (True = Locked, False = Unlocked)."""
    with lock_state.get_lock():  # Ensure safe modification
        lock_state.value = new_state
    print(f"Lock state set to: {'Locked' if new_state else 'Unlocked'}")


def transition_to(state):
    global current_state,username,video_stream,lock_state


    print(f"Transitioning to: {state}")
    current_state = state
    
    if state == AuthState.IDLE:
        pass
    elif state == AuthState.RADAR_DETECTION:
        # No need to start radar process here if it was already started
        print("Waiting for radar events...")
        time.sleep(5)
        print("Starting detection.")
        #transition_to(AuthState.VOICE_AUTH)
        
        
    elif state == AuthState.VOICE_AUTH:
        if video_stream is None:
            video_stream=VideoStream()
        
    elif state == AuthState.FACE_AUTH:
       print("transitioning to face authentication")

    elif state == AuthState.AUTHENTICATED:
        print("User authenticated successfully!")
        
        time.sleep(5)
        transition_to(AuthState.RADAR_DETECTION)
        
        
# State machine logic
def state_machine(queue):
    global current_state,video_stream,signal,registered_username

    # Shared variable to track if the person is in the radar's field of view
    person_in_view = Value('i', 0)  # 0 means not in view, 1 means in view
    


    while True:

        print("in loop")


        # Handle state transitions
        
            
        if current_state == AuthState.IDLE:
            #check mqtt topic code here
            
            
            print("idling")
            
            signal = input("type a to arm the lock or r to register a new user: ")
            registered_username = "" # to add from mqtt message
            
            if signal == "a":
                transition_to(AuthState.RADAR_DETECTION)
            
            elif signal == "r":
                #if video_stream is None:
                #    video_stream=VideoStream()
                print(f"registering user {registered_username}: ")
                register.register_new_user(registered_username)
                #video_stream.stop()
                #video_stream=None
       

        elif current_state == AuthState.RADAR_DETECTION:
            print("Running radar detection.")
            if not radar_proc.is_alive():
                radar_proc.start()
            if not accel_proc.is_alive():
                accel_proc.start()

            while True:
                
                #check mqtt
                
                if signal == "d":
                    print("disarmed")
                    if radar_proc.is_alive():
                        radar_proc.join()
                    if accel_proc.is_alive():
                        accel_proc.join()
                    
                    transition_to(AuthState.IDLE)
                    
                
                
                if not queue.empty():
                    msg = queue.get(timeout=2)
                    print(msg)
                    if msg == "accelerometer_movement":
                        print("Intrusion detected! Locking the system.")
                        if video_stream is None:
                            video_stream = VideoStream()
                        takephoto.take_photo(video_stream , "accel_trigger.jpg")
                        video_stream.stop()
                        video_stream = None
                        
                        publisher.send_image("accel_trigger.jpg")
                        
                        transition_to(AuthState.RADAR_DETECTION)
                        break

                    elif msg == "wake_signal":
                        print("waking up...")
                        transition_to(AuthState.VOICE_AUTH)
                        break
                    elif msg == "reset_signal":
                        pass
                time.sleep(1)


        elif current_state == AuthState.VOICE_AUTH:
            success = audiodriver.recognise_voice(3,0.80,queue)
            print(success)
            if success["verified"]:
                username = success["match"]
                transition_to(AuthState.FACE_AUTH)
            else:
                print("Voice authentication failed. Returning to IDLE.")
                takephoto.take_photo(video_stream , "voice_fail.jpg")
                video_stream.stop()
                video_stream = None
                publisher.send_image("voice_fail.jpg")
                transition_to(AuthState.RADAR_DETECTION)
            success=None
            print("Running voice authentication.")

        elif current_state == AuthState.FACE_AUTH:
            # Face authentication handled by function
            print("Running face authentication.")
           # Run the face authentication function
            success = cameradriver.recognise_face(video_stream , username,0.8, queue)
            if success:
                publisher.send_text(f"{username} successfully authenticated.")
                video_stream.stop()
                video_stream = None
                cv2.destroyAllWindows()
                transition_to(AuthState.AUTHENTICATED)
            else:
                if video_stream is None:
                    video_stream = VideoStream()
                takephoto.take_photo(video_stream, "face_fail.jpg")
                print("Face authentication failed. Returning to IDLE.")
                
                video_stream.stop()
                video_stream = None
                
                publisher.send_image("face_fail.jpg")
                
                cv2.destroyAllWindows()
                transition_to(AuthState.RADAR_DETECTION)
            success = None
                    

        time.sleep(1)  # Avoid high CPU usage
        
def radar_process(queue, lock_state):
    radardriver.proxy_check_2(queue)
        
        
# Function to start the accelerometer as a separate process
def accelerometer_process(queue, lock_state):
    acceldriver.calibration()  # Calibrate before starting detection
    while True:
        result = acceldriver.detect_doorshake_loopable2()
        #with lock_state.get_lock():  # Ensure safe access
        #    if not lock_state.value:
        #        continue  # Skip if system is unlocked
        if result:
            print("movement - adding to queue")
            queue.put("accelerometer_movement")
            
        #print(queue.get())
        time.sleep(2)

if __name__ == "__main__":
    
  
    queue = Queue(maxsize = 10)
    
    
    
    #queue = Queue()  # Shared message queue
    
    lock_state = Value('b', True)
    
    radar_proc = Process(target=radar_process, args=(queue, lock_state,),)
    accel_proc = Process(target=accelerometer_process, args=(queue, lock_state,),)


    

    # Start the state machine
    try:
        state_machine(queue)
    
    except KeyboardInterrupt:
        video_stream.stop()
        print("loop stopped")
        
    if radar_proc.is_alive():
        radar_proc.join()
    if accel_proc.is_alive():
        accel_proc.join()

    

    # Ensure processes are terminated properly


