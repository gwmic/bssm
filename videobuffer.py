import cv2
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

def captureBuffer(data):

    # Initialize the webcam
    cap = cv2.VideoCapture(data.source)
    if data.fps == 0:
        data.fps = cap.get(cv2.CAP_PROP_FPS)  # Get the FPS of the webcam
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # Initialize the buffer
    video_buffer = VideoBuffer(2, data.fps, frame_size)  #2 seconds buffer

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    recording = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if not(recording):
            video_buffer.add_frame(frame)

        if data.vidFlag and not(recording):
            recording = True
            out = cv2.VideoWriter('output.mp4', fourcc, data.fps, frame_size)

        if recording:
            out.write(frame)

        if not data.vidFlag and recording:
            for f in video_buffer.get_buffer():  # Write the last 2 seconds
                out.write(f)
            break

    # Release everything when done
    cap.release()
    if out:
        out.release()
    data.done = True