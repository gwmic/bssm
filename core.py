import cv2
import inference
import supervision as sv
import numpy as np
import threading
import time
import modules as mod
import videobuffer as vb
import shot as st

source = 1 #Source num of webcam

# manages all global data as data object
class DataMan:
    def __init__(self, source):
        self.init_arrays()
        self.init_flags()
        self.init_counters()
        self.annotator = sv.BoxAnnotator()
        self.currentY = 173
        self.source = source
        self.clickArr = ["Top Left", "Top Right", "Bottom Left", "Bottom Right"]

    def init_arrays(self):
        # Initialize all array attributes to empty numpy arrays
        self.ballArr = self.shotArr = self.laneArr = self.matrix = self.realMatrix = self.poly = np.array([])

    def init_flags(self):
        # Initialize all boolean flags
        self.laneSet = self.vidFlag = self.done = self.scan = False

    def init_counters(self):
        # Initialize all counter attributes to zero
        self.count = self.x1 = self.x2 = self.x3 = self.x4 = 0
        self.percentage = self.y1 = self.y2 = self.y3 = self.y4 = 0
        self.shotLimit = self.frameLimit = self.frameCount = self.frameRender = self.ballCount = 0

data = DataMan(source)

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
                bssmDimArr = np.float32([[0, 0], [183, 0], [183, 2457], [0, 2457]])  
                laneDimArr = np.float32([[0, 0], [39, 0], [39, 671], [0, 671]])  
               
                # Convert the quadrilateral and point into a format suitable for cv2 functions
                data.poly = np.float32(quadArr)
                data.matrix = cv2.getPerspectiveTransform(data.poly, bssmDimArr)
                data.realMatrix = cv2.getPerspectiveTransform(data.poly, laneDimArr)
                data.laneSet = True

                data.frameLimit = data.frameRender

                print("\nREADY")

def renderWrapper(data):
    def render(predictions, image):

        idArr = sv.Detections.from_roboflow(predictions).class_id  # creates an array of all class_id of detections in a given frame
        frame = mod.extractframe(predictions)  # define current frame

        if data.laneSet:
            if "ball" in str(predictions):  # checks if a ball has been detected in a given frame
                ballPosArr = sv.Detections.from_roboflow(predictions).xyxy[np.where(idArr == 0)]  # creates an array of min/max values for the bounding boxes of each ball in a given frame

                frameBallArr = [st.Ball(element, frame) for element in ballPosArr]  # create a ball object for each element in posArr — passing in current frame_id 

                frameBallArr =  [ball for ball in frameBallArr if mod.inside(data.poly, ball.x, ball.y)]  # removes balls from frameBallArr if they are not on the lane 

                if np.size(frameBallArr) == 1:  # checks if ball is on lane, then sets frameLimit = frame
                    data.frameLimit = frame
                    data.vidFlag = True 
                    data.ballCount += np.size(frameBallArr)
                
                elif np.size(frameBallArr) > 1:
                    print("Error: more than one ball detected on the lane")

            if (frame - data.frameLimit) > 60:  # if 60 frames (2 sec) has elapsed without a ball detection on the lane, the shot will be exported to
                if data.ballCount >= 2:
                    data.done = False
                    data.vidFlag = False
                    
                    while not data.done:
                        time.sleep(0.1)  # To prevent a tight loop, we sleep for a short duration

                    time.sleep(0.5)
                    data.frameCount = 0

                    customMaster = masterWrapper(data)

                    inference.Stream(
                        source="output.mp4",
                        model="bowling-model/6",
                        confidence=0.1,
                        iou_threshold=0.01,
                        output_channel_order="BGR",
                        use_main_thread=True,
                        on_prediction=customMaster
                    )

                    data.ballCount = 0
                    thread = threading.Thread(target=vb.captureBuffer, args=(data,))
                    thread.start()

                    print("\nREADY")

            image = cv2.polylines(image, [data.laneArr], False, (0, 165, 255), 3)  # draws orange lane

        else:  # if lane not defined then call click_event
            cv2.setMouseCallback("Camera", click_event, data) 
            data.frameRender = frame

        image = data.annotator.annotate(
            scene=image, 
            detections=sv.Detections.from_roboflow(predictions)
        )

        cv2.imshow("Camera", image)  # update Camera window
        cv2.waitKey(1)

        size = np.size(data.shotArr)  # calls bssm function if a shot has been saved
        if size >= 1:  
            if not(size == data.shotLimit):
                bssm(data)
                data.shotLimit = size
    return render

def masterWrapper(data):
    def master(predictions, image):

        idArr = sv.Detections.from_roboflow(predictions).class_id  # creates an array of all class_id of detections in a given frame
        frame = mod.extractframe(predictions)  # define current frame

        if data.frameCount == 0:
            cap = cv2.VideoCapture("output.mp4")
            data.frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print("\n")

        else:
            data.percentage = frame/(data.frameCount - 10)
            if data.percentage > 1.0:
                data.percentage = 1.0
            if data.percentage < 1.0:
                filled_length = int(50 * data.percentage)  # Calculate the number of 'filled' characters in the bar
                bar = '█' * filled_length + '-' * (50 - filled_length)  # Create the bar string
                print(f"\rProgress: |{bar}| {data.percentage*100:.2f}%", end='\r')  # Print the progress bar with percentage

        if "ball" in str(predictions) and not data.scan:  # checks if a ball has been detected in a given frame 
            ballPosArr = sv.Detections.from_roboflow(predictions).xyxy[np.where(idArr == 0)]  # creates an array of min/max values for the bounding boxes of each ball in a given frame
            frameBallArr = [st.Ball(element, frame) for element in ballPosArr]  # create a ball object for each element in posArr — passing in current frame_id 
            frameBallArr =  [ball for ball in frameBallArr if mod.inside(data.poly, ball.x, ball.y)]  # removes balls from frameBallArr if they are not on the lane 
            data.ballArr = np.append(data.ballArr, frameBallArr)  

        if frame > 5 and data.percentage >= 1.0 and not data.scan:
            shot = st.Shot(data.ballArr, data)
            data.shotArr = np.append(data.shotArr, shot)
            print("\n\nShot #", np.size(data.shotArr), " Saved With ", np.size(data.ballArr), "Cords")
            data.ballArr = np.array([])
            data.scan = True
    return master
           
def bssm(data): 
    image = cv2.imread("background.png")
    recentShot = data.shotArr[-1]
    polyBssm = recentShot.polyBssm

    # Generate curve points
    curve = np.array([], dtype=np.int32)
    for x in range(0, 2457):
        y = int(polyBssm(x))
        curve = np.append(curve, [x, 1150 - y])

    curveReshaped = curve.reshape(-1, 2)
    curve = np.array([curveReshaped], dtype=np.int32)

    image = cv2.polylines(image, [curve], False, (252, 255, 63), 2)

    # Shift the previous shots' information down
    for idx, shot in enumerate(data.shotArr[:-1]):
        mod.drawShotInfo(image, shot, 173 + 86 * (idx + 1), np.size(data.shotArr) - idx - 1)

    # Draw the most recent shot's information at y = 173
    mod.drawShotInfo(image, recentShot, 173, np.size(data.shotArr))

    # Update the y-coordinate for the next shot
    data.currentY += 86

    cv2.imshow("BSSM", image)  # update BSSM window
    cv2.waitKey(1)

thread = threading.Thread(target=vb.captureBuffer, args=(data,))
thread.start()

customRender = renderWrapper(data)

# run render function for each frame
inference.Stream(
    source=data.source,
    model="bowling-model/6",
    confidence=0.01,
    iou_threshold=0.01,
    output_channel_order="BGR",
    use_main_thread=True,
    on_prediction=customRender
)