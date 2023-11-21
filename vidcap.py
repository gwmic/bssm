import cv2
import time
from collections import deque

class VideoBuffer:
    def __init__(self, buffer_time, fps, frame_size):
        self.buffer = deque(maxlen=int(buffer_time * fps))
        self.fps = fps
        self.frame_size = frame_size

    def add_frame(self, frame):
        self.buffer.append(frame)

    def get_buffer(self):
        return list(self.buffer)

def captureBuffer(flag):
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get the FPS of the webcam
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # Initialize the buffer
    video_buffer = VideoBuffer(3, fps, frame_size)  # 2 seconds buffer

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    recording = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        video_buffer.add_frame(frame)

        if flag() and not recording:
            recording = True
            out = cv2.VideoWriter('output.mp4', fourcc, fps, frame_size)
            for f in video_buffer.get_buffer():  # Write the last 2 seconds
                out.write(f)

        if recording:
            out.write(frame)

        if not flag() and recording:
            break

    # Release everything when done
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

'''
# Example flag function
flag_state = False
def flag():
    global flag_state
    return flag_state

# Example usage with the flag
# The flag function can be replaced with any condition you want to use
import threading

def simulate_flag_change():
    global flag_state
    time.sleep(5)  # Wait for 5 seconds then start recording
    print("started")
    flag_state = True
    time.sleep(5)  # Record for 5 seconds
    flag_state = False

# Start the flag simulation in another thread
threading.Thread(target=simulate_flag_change).start()

# Start video capture
captureBuffer(flag)
'''