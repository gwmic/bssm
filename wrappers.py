from inference.core.interfaces.stream.inference_pipeline import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame
from typing import Optional
import modules as mod
import numpy as np
import cv2
import supervision as sv
import shot as st
import callables

import videobuffer as vb

def renderWrapper(data):
    def render(
            predictions: dict,
            video_frame: VideoFrame,
            fps_monitor: Optional[sv.FPSMonitor] = sv.FPSMonitor(),
        ) -> None:
        
        fps_value = None
        if fps_monitor is not None:
            fps_monitor.tick()
        fps_value = fps_monitor()
        data.fps = fps_value

        # Extract class IDs and positions from predictions
        image = video_frame.image
        detections = sv.Detections.from_roboflow(predictions)
        idArr = detections.class_id
        frame = video_frame.frame_id

        # Process only if lane is set
        if data.laneSet and not data.processing:
            # Process only if a ball is detected
            if np.size(detections.xyxy) > 0:
                ballPosArr = detections.xyxy[np.where(idArr == 0)]
                frameBallArr = [st.Ball(element, frame)
                                for element in ballPosArr]
                frameBallArr = [ball for ball in frameBallArr if mod.inside(
                    data.poly, ball.x, ball.y)]

                # Handle ball count and set conditions
                ballCount = np.size(frameBallArr)
                if ballCount >= 1:
                    data.frameLimit = frame
                    data.vidFlag = True
                    data.ballCount += ballCount
                elif ballCount > 1:
                    print("Error: more than one ball detected on the lane")

            # Check for time elapsed without ball detection
            if data.vidFlag and (frame - data.frameLimit) > data.fps*3 and data.ballCount >= 3:
                print("flag set")
                data.vidFlag = False
                data.processing = True
        
        else:
            cv2.setMouseCallback("BSSM", callables.clickEvent, data)
            data.frameRender = frame
            
        if data.selection == 1:
            # Annotate and display image
            image = data.annotator.annotate(scene=image, detections=detections)

            data.window["Camera"] = image

            if data.count == 4:
                image = cv2.polylines(
                    image, [data.laneArr], False, (0, 165, 255), 3)
        
        if data.done.is_set():
            data.frameCount = 0
            data.ballCount = -1
            data.percentage = 0.0
            data.scan = False
            
            master = InferencePipeline.init(
                model_id="bssm-small/2",
                video_reference="/Volumes/RAMdisk/output_cropped.mp4",
                confidence=data.confidenceLive,
                iou_threshold=0.01,
                on_prediction=masterWrapper(data),
                api_key="IQvYLHUhWtgWqompoERt"
            )
            master.start(use_main_thread=False)
            data.done.clear()
    return render

def masterWrapper(data):
    def master(
            predictions: dict,
            video_frame: VideoFrame,
            fps_monitor: Optional[sv.FPSMonitor] = sv.FPSMonitor(),
        ) -> None:

        image = video_frame.image
        fps_value = None
        if fps_monitor is not None:
            fps_monitor.tick()
        fps_value = fps_monitor()
        data.fps = fps_value

        #print(predictions)
        # Extract class IDs and positions from predictions
        detections = sv.Detections.from_roboflow(predictions)
        idArr = detections.class_id
        frame = video_frame.frame_id

        #data.predictions = np.append(data.predictions, predictions)

        # Initialize or update progress bar based on frame count
        if data.frameCount == 0:
            cap = cv2.VideoCapture("/Volumes/RAMdisk/output_cropped.mp4")
            data.frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            data.frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)/2)
            print("here")
            data.predictions = np.append(data.predictions, predictions)
        elif not data.scan:
            data.percentage = frame / (data.frameCount - 2)

        # Detect balls and store their positions
        # checks if a ball has been detected in a given frame
        if np.size(detections.xyxy) > 0 and not data.scan:
            ballPosArr = detections.xyxy[np.where(idArr == 0)]
            frameBallArr = [st.Ball(element, frame) for element in ballPosArr]
            # create a ball object for each element in posArr — passing in current frame_id
            frameBallArr = [ball for ball in frameBallArr 
                if mod.inside(data.croppedPoly, 
                    ball.x, ball.y
                )
            ]
            data.ballArr = np.append(data.ballArr, frameBallArr)

        # Handle creation and storage of shot data
        if frame > 5 and data.percentage >= 1.0 and not data.scan:
            data.predictions = np.append(data.predictions, predictions)
            shot = st.Shot(data.ballArr, data)
            data.shotArr = np.append(data.shotArr, shot)


            data.size = np.size(data.ballArr)
            data.ballArr = np.array([])
            data.selection = 2
            data.ballCount = 0
            data.scan = True
            data.processing = False
            #mod.cliProgress(1, "Video Segmentation", data)

        # Shows annotations if SHOWPROCESS is true
        if data.showProcess:
            image = cv2.polylines(
                image, [data.croppedLaneArr], False, (0, 165, 255), 2)
            image = data.annotator.annotate(scene=image, detections=detections)

            data.window["Process Region"] = image

        elif not data.scan:
            mod.cliProgress(data.percentage, "Video Segmentation", data)
    return master