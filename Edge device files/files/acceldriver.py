import smbus2
import time
from multiprocessing import Queue, Process

ACCEL_ADDRESS = 0x19
MAG_ADDRESS = 0x1E
CTRL_REG1_A = 0x20  # Power on the accelerometer
OUT_X_L_A = 0x28    # Accelerometer data
CALIBRATION_COUNT = 10
THRESHOLD = 3000

resting_x = 0
resting_y = 0
resting_z = 0

# Initialize I2C bus
bus = smbus2.SMBus(1)

# Enable accelerometer
bus.write_byte_data(ACCEL_ADDRESS, CTRL_REG1_A, 0x57)


def read_accel():
    """Read accelerometer data and return x, y, z data"""
    data = bus.read_i2c_block_data(ACCEL_ADDRESS, OUT_X_L_A | 0x80, 6)
    
    x = (data[1] << 8) | data[0]
    y = (data[3] << 8) | data[2]
    z = (data[5] << 8) | data[4]

    # Convert to signed values
    x = x if x < 32768 else x - 65536
    y = y if y < 32768 else y - 65536
    z = z if z < 32768 else z - 65536

    return x, y, z


def calibration():
    """Calibrate the resting state"""
    global resting_x, resting_y, resting_z

    for i in range(CALIBRATION_COUNT):
        ax, ay, az = read_accel()
        resting_x += ax
        resting_y += ay
        resting_z += az

        print(f"Calibration: {ax}, Y: {ay}, Z: {az}")
        time.sleep(0.2)

    resting_x = resting_x / CALIBRATION_COUNT
    resting_y = resting_y / CALIBRATION_COUNT
    resting_z = resting_z / CALIBRATION_COUNT

    print(f"Final Calibration X: {resting_x}, Y: {resting_y}, Z: {resting_z}")
    return resting_x


def detect_doorshake():
    """Get accelerometer data and check if door is being shaken"""
    x_low_threshold = resting_x - THRESHOLD
    x_high_threshold = resting_x + THRESHOLD

    while True:
        ax, ay, az = read_accel()
        print(f"Accel X: {ax}, Y: {ay}, Z: {az}")

        if x_low_threshold < ax < x_high_threshold:
            print("Kalm")
        else:
            print("Panik")
            return "Intrusion"

        time.sleep(2)
        
def detect_doorshake_loopable():
    """Continuously detect movement and send alerts."""
    """Get accelerometer data and check if door is being shaken"""
    x_low_threshold = resting_x - THRESHOLD
    x_high_threshold = resting_x + THRESHOLD
    
    while True:
        ax, ay, az = read_accel()
        print(f"Accel Data -> X: {ax}, Y: {ay}, Z: {az}")

        if not lock_state.value:
            continue  # Ignore movements if the system is unlocked
        
        if x_low_threshold < ax < x_high_threshold:
            #print("Kalm")
            pass
        else:
            print("Panik")
            
            queue.put("accelerometer_movement")
            print("Adding to queue: accelerometer_movement")

        time.sleep(2)  # Avoid CPU overuse
        
def detect_doorshake_loopable2():
    """Continuously detect movement and send alerts."""
    """Get accelerometer data and check if door is being shaken"""
    x_low_threshold = resting_x - THRESHOLD
    x_high_threshold = resting_x + THRESHOLD
    
   
    ax, ay, az = read_accel()
    #print(f"Accel Data -> X: {ax}, Y: {ay}, Z: {az}")
        
        
    if x_low_threshold < ax < x_high_threshold:
        #print("Kalm")
        return False
    else:
        #print("Panik")
        return True
        

def main():
    calibration()
    detect_doorshake()


if __name__ == "__main__":
    main()
