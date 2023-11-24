import cv2
import inference
import supervision as sv
import numpy as np
import threading
import time
import modules as mod
import videobuffer as vb
import shot as st

SOURCE = 1 # Source num of webcam
SHOWPROCESS = True # Toggle the processing annotations; Turning off saves 30% time vs. on

# manages all global data as data object
class DataMan:
    def __init__(self, source, showProcess):
        self.init_arrays()
        self.init_flags()
        self.init_counters()
        self.annotator = sv.BoxAnnotator()
        self.currentY = 173
        self.source = source
        self.showProcess = showProcess
        self.timeStr = "null"
        self.clickArr = ["Top Left", "Top Right", "Bottom Left", "Bottom Right"]

    def init_arrays(self):
        # Initialize all array attributes to empty numpy arrays
        self.ballArr = self.shotArr = self.laneArr = self.laneDimArr = self.bssmDimArr = self.poly = self.croppedPoly = self.croppedLaneArr = np.array([])

    def init_flags(self):
        # Initialize all boolean flags
        self.laneSet = self.vidFlag = self.done = self.scan = False

    def init_counters(self):
        # Initialize all counter attributes to zero
        self.count = self.x1 = self.x2 = self.x3 = self.x4 = 0
        self.percentage = self.y1 = self.y2 = self.y3 = self.y4 = self.frameWidth = 0
        self.shotLimit = self.frameLimit = self.frameCount = self.frameRender = self.ballCount = self.fps = 0

data = DataMan(SOURCE, SHOWPROCESS)

# Called when a button is cliked by setMouseCallback in render function
def click_event(event, x, y, flags, data):
    if event == cv2.EVENT_LBUTTONDOWN:  # Checking for left mouse clicks 
        # Collecting points for lane class
        if data.count < 4:
            setattr(data, f'x{data.count+1}', x)
            setattr(data, f'y{data.count+1}', y)
            print(f"{data.clickArr[data.count]}: ({x}, {y})")
            data.count += 1

            if data.count == 4:
                # Once all points are collected, perform further processing
                data.laneArr = np.array([[data.x1, data.y1], [data.x2, data.y2], 
                                         [data.x4, data.y4], [data.x3, data.y3], 
                                         [data.x1, data.y1]], np.int32)

                quadArr = [(data.x1, data.y1), (data.x2, data.y2), 
                           (data.x4, data.y4), (data.x3, data.y3)]
                data.bssmDimArr = np.float32([[0, 0], [183, 0], [183, 2457], [0, 2457]])  
                data.laneDimArr = np.float32([[0, 0], [39, 0], [39, 671], [0, 671]])  
               
                # Convert the quadrilateral and point into a format suitable for cv2 functions
                data.poly = np.float32(quadArr)
                data.laneSet = True

                data.frameLimit = data.frameRender
                '''
                data.fps = 30
                vb.processVideo('output_test.mp4', 'output.mp4')
                vb.cropVid("output.mp4", "output_cropped.mp4", 40, 25, data)
                customMaster = masterWrapper(data)
                inference.Stream(
                    source="output_cropped.mp4",
                    model="bowling-model/6",
                    confidence=0.1,
                    iou_threshold=0.01,
                    output_channel_order="BGR",
                    use_main_thread=True,
                    on_prediction=customMaster,
                    enforce_fps=True
                )
                '''
                print("\nREADY")

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
                    source="output_cropped.mp4",
                    model="bowling-model/6",
                    confidence=0.1,
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
            cv2.setMouseCallback("Camera", click_event, data)
            data.frameRender = frame

            # Annotate and display image
        image = data.annotator.annotate(scene=image, detections=detections)
        cv2.imshow("Camera", image)
        cv2.waitKey(1)

        # Process saved shots
        size = np.size(data.shotArr)
        if size >= 1 and size != data.shotLimit:
            bssm(data)
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
            cap = cv2.VideoCapture("output_cropped.mp4")
            data.frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            data.frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)/2)
            print("\n")
        elif not data.scan:
            data.percentage = min(frame / (data.frameCount - 1), 1.0)
            if not data.showProcess:
                filled_length = int(50 * data.percentage)
                bar = '█' * filled_length + '-' * (50 - filled_length)
                print(f"\rProgress: |{bar}| {(data.percentage*100):.2f}%", end='\r') # save the progress bar with percentage

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
            cv2.imshow("image", image)
            cv2.waitKey(1)


    return master
           
def bssm(data):
    image = cv2.imread("background.png")
    recentShot = data.shotArr[-1]
    polyBssm = recentShot.polyBssm

    # Generate curve points more efficiently
    curve = np.array([], dtype=np.int32)
    for x in range(0, 2457):
        y = int(polyBssm(x))
        curve = np.append(curve, [x, 970 + y])

    curveReshaped = curve.reshape(-1, 2)
    curve = np.array([curveReshaped], dtype=np.int32)

    # Draw the curve on the image
    image = cv2.polylines(image, [curve], False, (252, 255, 63), 2)

    # Draw information for previous shots
    for idx, shot in enumerate(data.shotArr[:-1]):
        mod.drawShotInfo(image, shot, 173 + 86 * (idx + 1), np.size(data.shotArr) - idx - 1)

    # Draw the most recent shot's information
    mod.drawShotInfo(image, recentShot, 173, np.size(data.shotArr))

    # Update the y-coordinate for the next shot
    data.currentY += 86

    # Display the updated image
    cv2.imshow("BSSM", image)
    cv2.waitKey(1)

thread = threading.Thread(target=vb.captureBuffer, args=(data,))
thread.start()

# run render function for each frame
customRender = renderWrapper(data)
inference.Stream(
    source=data.source,
    model="bowling-model/6",
    confidence=0.1,
    iou_threshold=0.01,
    output_channel_order="BGR",
    use_main_thread=True,
    on_prediction=customRender
)