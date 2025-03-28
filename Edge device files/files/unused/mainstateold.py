import time
import random
from multiprocessing import Queue, Process, Value
#import audiodrivermelspec as audiodriver
import audiodriver as audiodriver
import cameradriver
import takephoto
from videostream import VideoStream
import acceldriver


class AuthState:
    IDLE = 'IDLE'
    RADAR_DETECTION = 'RADAR_DETECTION'
    VOICE_AUTH = 'VOICE_AUTH'
    FACE_AUTH = 'FACE_AUTH'
    AUTHENTICATED = 'AUTHENTICATED'


current_state = AuthState.IDLE
username=""
video_stream = None

# Global variable for radar process
radar_process = None  # Define radar_process globally, outside of state machine
accel_proc = None

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

    if state == AuthState.RADAR_DETECTION:
        # No need to start radar process here if it was already started
        print("Waiting for radar events...")
        time.sleep(5)
        print("Person Detected. simulated")
        transition_to(AuthState.VOICE_AUTH)
        
        
    elif state == AuthState.VOICE_AUTH:
        video_stream=VideoStream()
        # Run the voice authentication function
        success = audiodriver.recognise_voice(3,0.80)
        print(success)
        if success["verified"]:
            username = success["match"]
            transition_to(AuthState.FACE_AUTH)
        else:
            print("Voice authentication failed. Returning to IDLE.")
            takephoto.take_photo(video_stream)
            video_stream.stop()
            transition_to(AuthState.RADAR_DETECTION)

    elif state == AuthState.FACE_AUTH:
        print("Transition to faceauth!")

    elif state == AuthState.AUTHENTICATED:
        print("User authenticated successfully!")
        
        time.sleep(5)
        transition_to(AuthState.RADAR_DETECTION)

# State machine logic
def state_machine(queue):
    global current_state,video_stream

    # Shared variable to track if the person is in the radar's field of view
    person_in_view = Value('i', 0)  # 0 means not in view, 1 means in view
    

    while True:

        print("Checking queue...")
        try:
            print("Checking queue...")
            event = queue.get(timeout=5)  # Use timeout to prevent blocking forever
            print(f"Event received: {event}")

            if event == "accelerometer_movement":
                print("Intrusion detected! Locking the system.")
                video_stream = VideoStream()
                takephoto.take_photo(video_stream)
                video_stream.stop()
                set_lock_state(True)
                transition_to(AuthState.IDLE)

            elif event == "radar_movement":
                if current_state == AuthState.IDLE:
                    set_lock_state(True)
                    print("Movement detected! Transitioning to radar detection.")
                    transition_to(AuthState.RADAR_DETECTION)

        except Exception as e:
            print(f"Queue empty or error: {e}")

        # Handle state transitions
        if current_state == AuthState.IDLE:
            print("System is idle.")
            time.sleep(60)
            print("simulate radar detection")
            transition_to(AuthState.RADAR_DETECTION)

        elif current_state == AuthState.RADAR_DETECTION:
            print("Running radar detection.")


        elif current_state == AuthState.VOICE_AUTH:
            # Voice authentication handled by function
            print("Running voice authentication.")
            # Run the face authentication function
            success = cameradriver.recognise_face(video_stream , username)
            if success:
                video_stream.stop()
                transition_to(AuthState.AUTHENTICATED)
            else:
                video_stream = VideoStream()
                takephoto.take_photo(video_stream)
                print("Face authentication failed. Returning to IDLE.")
                
                video_stream.stop()
                cv2.destroyAllWindows()
                transition_to(AuthState.RADAR_DETECTION)

        elif current_state == AuthState.FACE_AUTH:
            # Face authentication handled by function
            print("Running face authentication.")
            transition_to(AuthState.AUTHENTICATED)

        elif current_state == AuthState.AUTHENTICATED:
            set_lock_state(False)
            print("User session active. Resetting after 10 seconds...")
            time.sleep(10)
            transition_to(AuthState.IDLE)

        time.sleep(1)  # Avoid high CPU usage
        
        
# Function to start the accelerometer as a separate process
def accelerometer_process(queue, lock_state):
    acceldriver.calibration()  # Calibrate before starting detection
    while True:
        result = acceldriver.detect_doorshake_loopable2()
        with lock_state.get_lock():  # Ensure safe access
            if not lock_state.value:
                continue  # Skip if system is unlocked
        if result:
            print("movement - adding to queue")
            queue.put("accelerometer_movement")
            
        #print(queue.get())
        time.sleep(2)
            


if __name__ == "__main__":
    
    queue = Queue()  # Shared message queue
    
    lock_state = Value('b', True)

    # radar_process = Process(target=radar_detection_process, args=(Value('i', 0),))
    accel_proc = Process(target=accelerometer_process, args=(queue, lock_state,),)

    # radar_process.start()
    accel_proc.start()

    # Start the state machine
    try:
        state_machine(queue)
    
    except KeyboardInterrupt:
        video_stream.stop()
        print("loop stopped")
        
    

    # Ensure processes are terminated properly

    # radar_process.join()
    # accelerometer_process.join()
