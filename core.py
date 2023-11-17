import cv2
import inference
import supervision as sv
import numpy as np
import re

annotator = sv.BoxAnnotator()

# Define global variables 
count = 0
x1=0
x2=0
x3=0
x4=0
y1=0
y2=0
y3=0
y4=0
global lane
global laneArr
global ballArr
global frameLimit
global quadArr
global poly
global laneSet
global shotArr
global matrix
ballArr = np.array([])
shotArr = np.array([])
laneSet = False
frameLimit = 0

class LaneCord:
  def __init__(self, ball):
    global matrix
    pt = np.float32([[ball.x, ball.y]])  # Changed to a 2D point
    pt = np.array([pt])  # Reshaping to (1, N, 2) format
    transPt = cv2.perspectiveTransform(pt, matrix)
    
    self.x = transPt[0][0][0]
    self.y = transPt[0][0][1]
    self.frame = ball.frame

# Extracts frame number from prediction object
def extractframe(predictions):
    match = re.search(r"'frame_id': (\d+)", str(predictions))  # Regular expression to find "frame:" followed by an integer

    if match:
        return int(match.group(1))  # Extract and return the integer
    else:
        return -1  # Return -1 if no frame is found

# checks if x, y are inside a poly (in this case, the lane)
def inside(x, y):
    result = cv2.pointPolygonTest(poly, (x, y), False)  # Use cv2.pointPolygonTest to check if the point is inside the polygon

    return result >= 0 # result > 0: inside, result = 0: on the edge, result < 0: outside
    
class Shot:
  def __init__(self, arr):
    shotPts = [LaneCord(ball) for ball in arr]

    self.pts = shotPts

# Ball class given min/max arry, and the corresponding frame 
class Ball:
  def __init__(self, arr, frame):
    self.y = arr[3] # arr[3] is ymax for bounding box (bottom of bowling ball)
    self.x = (arr[0] + arr[2])/2 #arr[0] and arr[2] are xmin and xmax, respectivley, for bounding box
    self.frame = frame
  
  def __str__ (self):
     return "(" + str(self.x) + "," + str(self.y) + ":" + str(self.frame) + ")"

# Lane stores the four points for the lane trapizoid
class Lane:
  def __init__(self, x1, y1, x2, y2, x3, y3, x4, y4):
    self.x1 = x1
    self.y1 = y1
    self.x2 = x2
    self.y2 = y2
    self.x3 = x3
    self.y3 = y3
    self.x4 = x4
    self.y4 = y4

# Called when a button is cliked by setMouseCallback in render function
def click_event(event, x, y, flags, params): 
    global count 
    global x1
    global x2
    global x3
    global x4
    global y1
    global y2
    global y3
    global y4
    global laneArr
    if event == cv2.EVENT_LBUTTONDOWN: # Checking for left mouse clicks 
        if count >= 4:  # Collecting points for lane class
            x4 = x
            y4 = y
            print("Lane Already Selcted") 
            print(laneArr)
        elif count == 0:
            x1 = x
            y1 = y
            print("Top Left: (",x,",",y,")")
            count += 1
        elif count == 1:
            x2 = x
            y2 = y
            print("Top Right: (",x,",",y,")")
            count += 1
        elif count == 2:
            x3 = x
            y3 = y
            print("Bottom Left: (",x,",",y,")")
            count += 1
        elif count == 3:
            x4 = x
            y4 = y
            print("Bottom Right: (",x,",",y,")")
            count += 1

            # creating lane class, then assigning values to laneArr to be used in cv2.polylines    
            global lane
            global matrix
            global quadArr
            global poly
            global laneSet
            global ballArr
            global shotArr

            lane = Lane(x1, y1, x2, y2, x3, y3, x4, y4)
            laneArr = np.array([[lane.x1, lane.y1], [lane.x2, lane.y2], 
                [lane.x4, lane.y4], [lane.x3, lane.y3], [lane.x1, lane.y1]],
               np.int32)
            laneDimArr = np.float32([[0, 0], [39, 0], [39, 773], [0, 773]])
            quadArr = [(lane.x1, lane.y1), (lane.x2, lane.y2), (lane.x4, lane.y4), (lane.x3, lane.y3)]
           
            # Convert the quadrilateral and point into a format suitable for cv2 functions
            poly = np.float32(quadArr)
            matrix = cv2.getPerspectiveTransform(poly, laneDimArr)
            laneSet = True
          
def render(predictions, image):
    global ballArr
    global frameLimit
    global laneSet
    global shotArr

    idArr = sv.Detections.from_roboflow(predictions).class_id  # creates an array of all class_id of detections in a given frame
    frame = extractframe(predictions)  # define current frame

    if laneSet:
        if "ball" in str(predictions):  # checks if a ball has been detected in a given frame
            ballPosArr = sv.Detections.from_roboflow(predictions).xyxy[np.where(idArr == 0)]  # creates an array of min/max values for the bounding boxes of each ball in a given frame

            frameBallArr = [Ball(element, frame) for element in ballPosArr]  # create a ball object for each element in posArr — passing in current frame_id 

            frameBallArr =  [ball for ball in frameBallArr if inside(ball.x, ball.y)]  # removes balls from frameBallArr if they are not on the lane 

            if np.size(frameBallArr) >= 1:  # checks if ball is on lane, then sets frameLimit = frame
                frameLimit = frame 

            if np.size(ballArr) == 0:  # appends this new array to the global array ballArr
                ballArr = frameBallArr
            else:
                ballArr = np.append(ballArr, frameBallArr)

        if (frame - frameLimit) > 30:  # if 30 frames (1 sec) has elapsed without a ball detection on the lane, the shot will be exported
            frameLimit = frame 

            if np.size(shotArr) == 0:
                shotArr = np.array([Shot(ballArr)]) 
            else:
                shotArr = np.append(shotArr, Shot(ballArr))
            
            print("Shot #", np.size(shotArr), " Saved")
            ballArr = np.array([])
        
        image = cv2.polylines(image, [laneArr], False, (0, 165, 255), 3)  # draws orange lane

    else:  # if lane not defined then call click_event
        cv2.setMouseCallback("BSSM", click_event) 

    image = annotator.annotate(
        scene=image, 
        detections=sv.Detections.from_roboflow(predictions)
    )

    cv2.imshow("BSSM", image) 
    cv2.waitKey(1)

# run render function for each frame
print("Please select lane coordinates")
inference.Stream(
    source=0,
    model="bowling-model/5",
    confidence=0.01,
    iou_threshold=0.01,
    output_channel_order="BGR",
    use_main_thread=True,
    on_prediction=render
)