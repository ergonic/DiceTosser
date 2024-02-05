import cv2
import threading
import queue

class CameraCapture:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.q = queue.Queue()
        self.running = True
        self.thread = threading.Thread(target=self._reader)
        self.thread.start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()   # Discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()

    def get_w(self):
        return self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    def get_h(self):
        return self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

# # Usage
# camera = CameraCapture()
#
# # Main loop
# while True:
#     frame = camera.read()
#     # Process the frame
#     cv2.imshow('Frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# camera.stop()
# cv2.destroyAllWindows()
