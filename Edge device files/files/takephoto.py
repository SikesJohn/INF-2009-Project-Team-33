import cv2
import videostream

def take_photo(video_stream, output_file = 'surveilance.jpg'):
    
    ret, frame = video_stream.read()  # Capture frame-by-frame
        
    if ret:
        # Save the frame as an image file
        cv2.imwrite( output_file , frame)
        print('Photo taken')
        
        # Read the image back in and publish
#        with open("surveilance.jpg", "rb") as f:
#            file_content = f.read()
#            byte_arr = bytearray(file_content)
    else:
        print('Failed to capture image')
#    return byte_arr
