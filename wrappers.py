import modules as mod
import numpy as np
import cv2
import inference
import threading
import videobuffer as vb
import time
import supervision as sv
import shot as st
import callables

def renderWrapper(data):
    def render(predictions, image):
        # Extract class IDs and positions from predictions
        detections = sv.Detections.from_roboflow(predictions)
        idArr = detections.class_id
        frame = mod.extractframe(predictions)

        # Process only if lane is set
        if data.laneSet:
            # Process only if a ball is detected
            if "ball" in str(predictions):
                ballPosArr = detections.xyxy[np.where(idArr == 0)]
                frameBallArr = [st.Ball(element, frame) for element in ballPosArr]
                frameBallArr = [ball for ball in frameBallArr if mod.inside(data.poly, ball.x, ball.y)]

                # Handle ball count and set conditions
                ball_count = np.size(frameBallArr)
                if ball_count == 1:
                    data.frameLimit = frame
                    data.vidFlag = True
                    data.ballCount += ball_count
                elif ball_count > 1:
                    print("Error: more than one ball detected on the lane")

            # Check for time elapsed without ball detection
            if (frame - data.frameLimit) > data.fps * 2 and data.ballCount >= 2:
                data.done = False
                data.vidFlag = False

                while not data.done:
                    time.sleep(0.1)

                data.frameCount = 0

                customMaster = masterWrapper(data)
                inference.Stream(
                    source=".output_cropped.mp4",
                    model="bowling-model/6",
                    confidence=data.confidenceProcess,
                    iou_threshold=0.01,
                    output_channel_order="BGR",
                    use_main_thread=True,
                    on_prediction=customMaster,
                    enforce_fps=True
                )

                data.ballCount = 0
                data.scan = False
                thread = threading.Thread(target=vb.captureBuffer, args=(data,))
                thread.start()

                print("\nREADY")

            # Draw lane and set callback if lane not defined
            image = cv2.polylines(image, [data.laneArr], False, (0, 165, 255), 3)
        else:
            cv2.setMouseCallback("Camera", callables.clickEvent, data)
            data.frameRender = frame

            # Annotate and display image
        image = data.annotator.annotate(scene=image, detections=detections)
        cv2.imshow("Camera", image)
        cv2.waitKey(1)

        # Process saved shots
        size = np.size(data.shotArr)
        if size >= 1 and size != data.shotLimit:
            callables.bssm(data)
            data.shotLimit = size
    return render

def masterWrapper(data):
    def master(predictions, image):
        # Extract class IDs and positions from predictions
        detections = sv.Detections.from_roboflow(predictions)
        idArr = detections.class_id
        frame = mod.extractframe(predictions)

        # Initialize or update progress bar based on frame count
        if data.frameCount == 0:
            cap = cv2.VideoCapture(".output_cropped.mp4")
            data.frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            data.frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)/2)
            print("\n")
        elif not data.scan:
            data.percentage = min(frame / (data.frameCount - 2), 1.0)

        # Detect balls and store their positions
        if "ball" in str(predictions) and not data.scan:  # checks if a ball has been detected in a given frame 
            ballPosArr = detections.xyxy[np.where(idArr == 0)]
            frameBallArr = [st.Ball(element, frame) for element in ballPosArr]
            frameBallArr = [ball for ball in frameBallArr if mod.inside(data.croppedPoly, ball.x, ball.y)]  # create a ball object for each element in posArr — passing in current frame_id
            data.ballArr = np.append(data.ballArr, frameBallArr)

        # Handle creation and storage of shot data
        if frame > 5 and data.percentage >= 1.0 and not data.scan:
            shot = st.Shot(data.ballArr, data)
            data.shotArr = np.append(data.shotArr, shot)
            newline = "\n" if data.showProcess else "\n\n"
            print(f"{newline}Shot # {np.size(data.shotArr)} Saved With {np.size(data.ballArr)} Cords")
            data.ballArr = np.array([])
            data.scan = True
        
        # Shows annotations if SHOWPROCESS is true
        if data.showProcess:
            image = cv2.polylines(image, [data.croppedLaneArr], False, (0, 165, 255), 2)
            mod.drawProgressBar(image, data.percentage, 30, (255, 255, 255), (0, 0, 0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            image = data.annotator.annotate(scene=image, detections=detections)
            cv2.imshow("Process Region", image)
            cv2.waitKey(1)
        elif not data.scan:
            mod.cliProgress(data.percentage, "Video Segmentation")
    return master