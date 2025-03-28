import threading
import cv2

class VideoStream:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        threading.Thread(target=self.update, args=(), daemon=True).start()
    
    def update(self):
        while not self.stopped:
            self.ret, self.frame = self.cap.read()
    
    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

if __name__ == "__main__":
    video_stream = VideoStream()
    
    while True:
        ret, frame = video_stream.read()
        frame = cv2.resize(frame, (320, 240))
        if not ret or frame is None:
            print("Failed to capture frame")
            continue

        cv2.imshow("Live Camera Feed", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_stream.stop()
    cv2.destroyAllWindows()
        
