import serial
import time
from collections import deque
from videostream import VideoStream
from datetime import datetime, timedelta
import takephoto
import statistics

# Configure the serial connection
ser = serial.Serial("/dev/ttyS0", baudrate=115200, timeout=1)

presence_status = "OFF"  # Default state
range_value = None  # Store last known range value

presence_queue = deque(maxlen=5)  # Store last 5 presence readings
loiter_queue = deque(maxlen=5)  # Store last 5 presence readings
DISTANCE_THRESHOLD = 40  # Only detect presence if object is closer than 100 cm
LOITER_THRESHOLD = 100


auth_detected = False

def adaptive_average(buffer, tolerance=30):
    if not buffer:
        return None
    med = statistics.median(buffer)
    filtered_values = [x for x in buffer if abs(x - med) <= tolerance]  # Ignore extreme outliers
    return sum(filtered_values) / len(filtered_values) if filtered_values else med
        
def proxy_check_2(queue):
    global ser,presence_status,range_value,presence_queue,DISTANCE_THRESHOLD,auth_detected,LOITER_THRESHOLD
    print("start mm")
    revolving_list = deque(maxlen=30)
    loiter_count =0
    loiter_time = None
    while True:
        #ser.flushInput()
        #range_value = 0
        if ser.in_waiting > 0:
            data = ser.readline().decode('utf-8').strip()
            #print("Raw Data:", repr(data))  # Debug output

            # Check for range data
            if data.startswith("Range"):
                _, value = data.split()  # Extract number
                range_value = int(value)
                revolving_list.append(range_value)

                # Apply adaptive filtering
                smoothed_distance = adaptive_average(revolving_list)

                # Update range_value with the filtered result
                if smoothed_distance is not None:
                    range_value = smoothed_distance
                
                print(f"Distance: {range_value} cm, Presence: {auth_detected}")
                

            if range_value is not None and range_value <= DISTANCE_THRESHOLD:
                presence_queue.append(True)
                
                
                if list(presence_queue)[-3:].count(True) >= 3 and auth_detected == False:  # If at least 3 out of 5 readings are ON
                    presence_status = "ON"
                    auth_detected = True
                    
                    if queue.qsize() < queue._maxsize:  # Avoid blocking
                        print(f"sending wake signal Range:{range_value}, authstatus {auth_detected}")
                        queue.put("wake_signal")
                    else:
                        print("Queue is full, skipping wake_signal.")


            if range_value is not None and range_value <= LOITER_THRESHOLD:
                
                if auth_detected and range_value > DISTANCE_THRESHOLD:
                    print(f"Person moved to loiter range. Resetting auth.")
                    auth_detected = False
                    if queue.qsize() < queue._maxsize:  # Avoid blocking and reset if auth was previously detected
                        auth_detected = False
                        print(f"sending reset signal Range:{range_value}, authstatus {auth_detected}")
                        queue.put("reset_signal")
                    else:
                        print("Queue is full, skipping reset_signal.")
                if loiter_time is not None and datetime.now() - loiter_time >= timedelta(seconds = 45):
                    print("loiter_detected")
                    loiter_time = None
                    if queue.qsize() < queue._maxsize:  # Avoid blocking and reset if auth was previously detected
                        auth_detected = False
                        print(f"sending reset signal Range:{range_value}, authstatus {auth_detected}")
                        queue.put("reset_signal")
                    else:
                        print("Queue is full, skipping reset_signal.")
                elif loiter_time is None:
                    loiter_time = datetime.now()
                    
            else:
                presence_status = "OFF"  # Ignore "ON" if distance is too far
                
                presence_queue.clear()
                if auth_detected or loiter_time is not None: # if person was either trying to authenticate or loiter
                    loiter_count = 0
                    loiter_time = None
                    if queue.qsize() < queue._maxsize:  # Avoid blocking and reset if auth was previously detected
                        auth_detected = False
                        print(f"sending reset signal Range:{range_value}, authstatus {auth_detected}")
                        queue.put("reset_signal")
                    else:
                        print("Queue is full, skipping reset_signal.")



