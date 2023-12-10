import cv2
from collections import deque
import numpy as np
from datetime import datetime
import os
import modules as mod
import mastergui as gui

def captureBuffer(data):

    # Initialize the webcam
    cap = cv2.VideoCapture(data.source)
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # Initialize the buffer
    #video_buffer = VideoBuffer(2, data.fps, frame_size)  # 2 seconds buffer

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    recording = False
    while data.running:
        if data.saveShots:
            dateTime = datetime.now()
            dateStr = dateTime.strftime("%Y_%m_%d")
            if data.timeStr == "null":
                data.timeStr = dateTime.strftime("%H_%M")
            shotNum = np.size(data.shotArr) + 1
            directory = f"shots/{dateStr}/{data.timeStr}"
            fileName = f"{directory}/shot{shotNum}.mp4"

            # Check if the directory exists, and if not, create it
            if not os.path.exists(directory):
                os.makedirs(directory)
        else:
            fileName = "/Volumes/RAMdisk/output.mp4"

        while data.running:
            ret, frame = cap.read()
            if not ret:
                cap = None
                cap = cv2.VideoCapture(data.source)
            else:
                if data.vidFlag and not recording:
                    recording = True
                    out = cv2.VideoWriter(fileName, fourcc, data.fps, frame_size)

                if recording:
                    out.write(frame)

                if not data.vidFlag and recording:
                    break
        # Release everything when done
        cap.release()
        if out:
            out.release()

        #fileName = "/Volumes/Macintosh HD/Users/gmicc/Downloads/temp.mp4"
        cropVid(fileName, "/Volumes/RAMdisk/output_cropped.mp4", 0, 1, data)
        data.predictions = np.array([])
        data.done.set()
        recording = False
        out = None

def cropVid(inputPath, outputPath, startRemove, endRemove, data):
    # Open the input video
    cap = cv2.VideoCapture(inputPath)
    border = (cap.get(cv2.CAP_PROP_FRAME_HEIGHT))/30

    xValues = [data.x1, data.x2, data.x3, data.x4]
    yValues = [data.y1, data.y2, data.y3, data.y4]

    xmin = int(min(xValues) - border)
    xmax = int(max(xValues) + border)
    ymin = int(min(yValues) - 3*border) #give room for rack in fov
    ymax = int(max(yValues) + border)

    xTrans = [(x - xmin) for x in xValues]
    yTrans = [(y - ymin) for y in yValues]

    quadArr = [(xTrans[0], yTrans[0]), (xTrans[1], yTrans[1]),
               (xTrans[3], yTrans[3]), (xTrans[2], yTrans[2])]

    data.croppedLaneArr = np.array([[xTrans[0], yTrans[0]], [xTrans[1], yTrans[1]],
                                    [xTrans[3], yTrans[3]], [xTrans[2], yTrans[2]],
                                    [xTrans[0], yTrans[0]]], np.int32)

    data.croppedPoly = np.float32(quadArr)

    #create bounding box for rack
    y_box = min(yTrans[0], yTrans[1])
    small_border = int(border/3)
    rackArrTemp = [(xTrans[0]-small_border, y_box-small_border), (xTrans[1]+small_border, y_box-small_border),
               (xTrans[1]+small_border, y_box+(border*2)), (xTrans[0]-small_border, y_box+(border*2))]
    
    data.rackArr = np.float32(rackArrTemp)

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
        if (i % data.scanRate) == 0:
            if ret:
                # Crop the frame
                croppedFrame = frame[ymin:ymax, xmin:xmax]

                # Write the cropped
                out.write(croppedFrame)

                # Calculate and update progress
                progress = ((i - startRemove) / (totalFrames -
                            startRemove - endRemove))
                mod.cliProgress(progress, "Video Preprocess", data)

            else:
                print("Error reading frame.")
                break

    # Release everything when job is finished
    mod.cliProgress(1, "Video Preprocess", data)
    cap.release()
    out.release()


def processVideo(input_file, output_file, data):
    # Open the video file
    cap = cv2.VideoCapture(input_file)

    # Get total number of frames and frame rate
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = 30
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Check if there are enough frames in the video
    if total_frames < 60:
        print("The video is too short.")
        return

    # Read the last 60 frames
    last_60_frames = []
    for i in range(total_frames - 60, total_frames):
        if True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                last_60_frames.append(frame)
            else:
                print("Error reading frame.")
                return

    # Create a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, frame_rate,
                          (frame_width, frame_height))

    # Write the last 60 frames to the start of the new video
    for frame_index, frame in enumerate(last_60_frames):
        out.write(frame)
        mod.cliProgress(((frame_index + 1) / total_frames)
                        * (4/7), "Video Preprocess", data)

    # Reset to the start of the video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Write the rest of the video
    for i in range(total_frames - 60):
        if True:
            ret, frame = cap.read()
            if ret:
                out.write(frame)
                mod.cliProgress(((i + 61) / total_frames)
                                * (4/7), "Video Preprocess", data)
            else:
                print("Error reading frame.")
                break

    # Final progress update to ensure 100% is shown at the end
    mod.cliProgress(4/7, "Video Preprocess", data)

    # Release everything
    cap.release()
    out.release()