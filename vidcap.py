import cv2
import numpy as np
from collections import deque

class VideoBuffer:
    def __init__(self, fps, frame_size):
        self.buffer = deque(maxlen=fps * 3)  # Store last 3 seconds
        self.frame_size = frame_size
        self.fps = fps

    def add_frame(self, frame):
        self.buffer.append(frame)

    def get_last_seconds(self, seconds):
        # Get last 'seconds' seconds of frames
        num_frames = seconds * self.fps
        return list(self.buffer)[-num_frames:]

def capture_video(start_flag, stop_flag):
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get the FPS of the webcam
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    video_buffer = VideoBuffer(int(fps), frame_size)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    recording = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        video_buffer.add_frame(frame)

        if start_flag() and not recording:
            recording = True
            out = cv2.VideoWriter('output.mp4', fourcc, fps, frame_size)
            for f in video_buffer.get_last_seconds(2):  # Get the last 2 seconds
                out.write(f)

        if recording:
            out.write(frame)

        if stop_flag() and recording:
            break

    # Release everything when done
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

# Example usage with flags
start_flag = lambda: input("Press Enter to start recording (or type 'start')...") == 'start'
stop_flag = lambda: input("Press Enter to stop recording (or type 'stop')...") == 'stop'

capture_video(start_flag, stop_flag)