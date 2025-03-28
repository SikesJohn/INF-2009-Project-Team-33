import serial
print(serial.__file__)
import time
from collections import deque

# Configure the serial connection
ser = serial.Serial("/dev/ttyS0", baudrate=115200, timeout=1)

presence_status = "OFF"  # Default state
range_value = None  # Store last known range value

presence_queue = deque(maxlen=5)  # Store last 5 presence readings
DISTANCE_THRESHOLD = 80  # Only detect presence if object is closer than 100 cm

while True:
    if ser.in_waiting > 0:
        data = ser.readline().decode('utf-8').strip()
        print("Raw Data:", repr(data))  # Debug output

        # Check for range data
        if data.startswith("Range"):
            _, value = data.split()  # Extract number
            range_value = int(value)
            print(f"Distance: {range_value} cm, Presence: {presence_status}")

        # Check for presence status
        elif data == "ON":
            presence_queue.append("ON")

        elif data == "OFF":
            presence_queue.append("OFF")

        # Determine actual presence based on recent readings & distance
        if range_value is not None and range_value <= DISTANCE_THRESHOLD:
            if presence_queue.count("ON") >= 3:  # If at least 3 out of 5 readings are ON
                presence_status = "ON"
            else:
                presence_status = "OFF"
        else:
            presence_status = "OFF"  # Ignore "ON" if distance is too far

        # Print final result
        if presence_status == "ON" and range_value is not None:
            print(f"Stable Presence Detected! Distance: {range_value} cm")
        elif presence_status == "OFF":
            print("No Presence Detected.")
