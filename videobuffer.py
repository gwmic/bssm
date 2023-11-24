import cv2
from collections import deque
import numpy as np
from datetime import datetime
import os

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
            out = cv2.VideoWriter('.temp.mp4', fourcc, data.fps, frame_size)

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

    dateTime = datetime.now()
    dateStr = dateTime.strftime("%Y_%m_%d")
    if data.timeStr == "null":
        timeStr = dateTime.strftime("%H_%M")
    shotNum = np.size(data.shotArr) + 1
    directory = f"shots/{dateStr}/{timeStr}"
    fileName = f"{directory}/shot{shotNum}.mp4"

    # Check if the directory exists, and if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    processVideo(".temp.mp4", fileName)
    cropVid("output.mp4", "output_cropped.mp4", 40, 25, data)
    data.done = True

def cropVid(inputPath, outputPath, startRemove, endRemove, data):
    # Open the input video
    cap = cv2.VideoCapture(inputPath)
    border = (cap.get(cv2.CAP_PROP_FRAME_HEIGHT))/30

    xValues = [data.x1, data.x2, data.x3, data.x4]
    yValues = [data.y1, data.y2, data.y3, data.y4]

    xmin = int(min(xValues) - border)
    xmax = int(max(xValues) + border)
    ymin = int(min(yValues) - border) - 30
    ymax = int(max(yValues) + border)

    xTrans = [(x - xmin) for x in xValues]
    yTrans = [(y - ymin) for y in yValues]

    quadArr = [(xTrans[0], yTrans[0]), (xTrans[1], yTrans[1]), 
                           (xTrans[3], yTrans[3]), (xTrans[2], yTrans[2])]
    
    data.croppedLaneArr = np.array([[xTrans[0], yTrans[0]], [xTrans[1], yTrans[1]], 
                                         [xTrans[3], yTrans[3]], [xTrans[2], yTrans[2]], 
                                         [xTrans[0], yTrans[0]]], np.int32)
    
    data.croppedPoly = np.float32(quadArr)

    # Get video properties
    width = int(xmax - xmin)
    height = int(ymax - ymin)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(outputPath, fourcc, data.fps, (width, height))

    totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if totalFrames <= startRemove + endRemove:
        print("The video is too short to remove the specified number of frames.")

    cap.set(cv2.CAP_PROP_POS_FRAMES, startRemove)

    for i in range(startRemove, totalFrames - endRemove):
        ret, frame = cap.read()
        if ret:
            # Crop the frame
            croppedFrame = frame[ymin:ymax, xmin:xmax]

            # Write the cropped
            out.write(croppedFrame)

        else:
            print("Error reading frame.")
            break

    # Release everything when job is finished
    cap.release()
    out.release()

def processVideo(input_file, output_file):
    # Open the video file
    cap = cv2.VideoCapture(input_file)

    # Get total number of frames and frame rate
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Check if there are enough frames in the video
    if total_frames < 60:
        print("The video is too short.")
        return

    # Read the last 60 frames
    last_60_frames = []
    for i in range(total_frames - 60, total_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            last_60_frames.append(frame)
        else:
            print("Error reading frame.")
            return

    # Create a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, frame_rate, (frame_width, frame_height))

    # Write the last 60 frames to the start of the new video
    for frame in last_60_frames:
        out.write(frame)

    # Reset to the start of the video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Write the rest of the video
    for i in range(total_frames - 60):
        ret, frame = cap.read()
        if ret:
            out.write(frame)
        else:
            print("Error reading frame.")
            break

    # Release everything
    cap.release()
    out.release()