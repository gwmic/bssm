import cv2
import inference
import supervision as sv
import numpy as np
import re
import warnings
from collections import deque
import threading
import time
from scipy.optimize import minimize_scalar
from scipy.integrate import quad

annotator = sv.BoxAnnotator()

# Define global variables 
global laneArr, ballArr, frameLimit, quadArr, poly, laneSet, shotArr, matrix, realMatrix, shotLimit, vidFlag, frameCount, done, flag, frameRender, ballCount, currentY
ballArr = shotArr = np.array([])
laneSet = vidFlag = done = False
count = x1 = x2 = x3 = x4 = y1 = y2 = y3 = y4 = shotLimit = frameLimit = frameCount = frameRender = ballCount = 0
currentY = 173

def flag():
    global vidFlag
    return vidFlag

def putCentered(img, text, center, font, font_scale, color, thickness):
    # Get the text size
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)

    # Calculate the bottom-left corner of the text
    text_x = center[0] - text_size[0] // 2
    text_y = center[1] + text_size[1] // 2

    # Put the text on the image
    cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness)


def arcLength(poly, a, b):
    # Derivative of the polynomial
    dpoly = np.polyder(poly)

    # Function under the square root
    def integrand(x):
        return np.sqrt(1 + dpoly(x)**2)

    # Compute the integral
    length, _ = quad(integrand, a, b)
    return length

def findClosest(arr, target):
    closest_index = 0
    min_diff = float('inf')
    
    for i, value in enumerate(arr):
        diff = abs(value - target)
        if diff < min_diff:
            min_diff = diff
            closest_index = i

    return closest_index

def findLocalMin(poly, range_min, range_max):
    # Objective function
    def objective_function(x):
        return poly(x)

    # Find local minimum
    result = minimize_scalar(objective_function, bounds=(range_min, range_max), method='bounded')

    if result.success:
        return result.x, result.fun
    else:
        return None, None

def tan(poly, x_point):
    # Derivative of the polynomial
    derivative = np.polyder(poly)

    # Slope of the tangent at x_point
    slope = derivative(x_point)

    # Angle of the tangent from the horizontal line
    angle_radians = np.arctan(slope)
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees

class VideoBuffer:
    def __init__(self, buffer_time, fps, frame_size):
        self.buffer = deque(maxlen=int(buffer_time * fps))
        self.fps = fps
        self.frame_size = frame_size

    def add_frame(self, frame):
        self.buffer.append(frame)

    def get_buffer(self):
        return list(self.buffer)

def captureBuffer():
    global done, flag

    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get the FPS of the webcam
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # Initialize the buffer
    video_buffer = VideoBuffer(2, fps, frame_size)  # 2 seconds buffer

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    recording = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        video_buffer.add_frame(frame)

        if flag() and not(recording):
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
    print("Saved mp4")
    done = True

class LaneCord:
  def __init__(self, ball):
    global matrix
    global realMatrix

    pt = np.float32([[ball.x, ball.y]])  # Changed to a 2D point
    pt = np.array([pt])  # Reshaping to (1, N, 2) format
    transPt = cv2.perspectiveTransform(pt, matrix)
    
    realTransPt = cv2.perspectiveTransform(pt, realMatrix)
    
    self.bssmx = transPt[0][0][1] # bssmx and bssmy are the x and y cords scaled to the lane at the bottom of the bssm window (unit is pixles)
    self.bssmy = transPt[0][0][0]
    self.realx = realTransPt[0][0][1] # realx and realy are the x and y cords scaled to the dims of a bowling lane (unit is boards)
    self.realy = realTransPt[0][0][0]
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
    pts = [LaneCord(ball) for ball in arr]  # creates an array of translated cords 

    xBssm = [cord.bssmx for cord in pts]
    yBssm = [cord.bssmy for cord in pts]
    xReal = [cord.realx for cord in pts]
    yReal = [cord.realy for cord in pts]
    '''
    xBssm = [0, 819, 1638, 2457]
    yBssm = [173, 61, 33, 89]
    xReal = [0, 224, 447, 671]
    yReal = [37, 13, 7, 19]
    
    xBssm =  [temp for temp in xBssm if temp >= 0]
    yBssm =  [temp for temp in yBssm if temp >= 0]
    xReal =  [temp for temp in xReal if temp >= 0]
    yReal =  [temp for temp in yReal if temp >= 0]
    '''
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', np.RankWarning)
        coefficientsBssm = np.polyfit(xBssm, yBssm, 4)
        polynomialBssm = np.poly1d(coefficientsBssm)
    
        coefficientsReal = np.polyfit(xReal, yReal, 4)
        polynomialReal = np.poly1d(coefficientsReal)

    intersectionPoly = polynomialReal - np.poly1d([0.736, -103.04])
    roots = np.roots(intersectionPoly)
    rootsRight = [root.real for root in roots if np.isreal(root)]
    intersectionPoly = polynomialReal - np.poly1d([-0.736, 142.04])
    roots = np.roots(intersectionPoly)
    rootsLeft = [root.real for root in roots if np.isreal(root)]
    roots = np.append(rootsRight, rootsLeft)
    roots = [pt for pt in roots if 139.96 <= pt <= 166.45]
    arrows = '%.1f'%(polynomialReal(roots[0])) # find board where ball crosses arrows

    xmin, fmin = findLocalMin(polynomialReal, 0, 671)
    breakPtBoard = '%.1f'%(fmin)
    breakPtDis = '%.1f'%(xmin * (60/671))

    foulLine = '%.1f'%(polynomialReal(0))
    entryBoard = '%.1f'%(polynomialReal(671))

    launchAng = '%.1f'%(-tan(polynomialReal, 0))
    impactAng = '%.1f'%(tan(polynomialReal, 671))
    '''
    indexi = findClosest(xReal, 30)
    indexf = findClosest(xReal, 190)
    xi = xReal[indexi]
    xf = xReal[indexf]
    hours = (pts[indexf].frame - pts[indexi].frame)*120.0
    miles = (arcLength(polynomialReal, xi, xf))/59001.0
    launchSpeed = '%.1f'%(miles/hours)

    indexi = findClosest(xReal, 510)
    indexf = findClosest(xReal, 670)
    xi = xReal[indexi]
    xf = xReal[indexf]
    hours = (pts[indexf].frame - pts[indexi].frame)*120.0
    miles = (arcLength(polynomialReal, xi, xf))/59001.0
    entrySpeed = '%.1f'%(miles/hours)
    '''
    self.foulLine = foulLine
    self.arrows = arrows 
    self.breakPtBoard = breakPtBoard
    self.breakPtDis = breakPtDis
    self.entryBoard = entryBoard
    self.launchAng = launchAng
    self.impactAng = impactAng
    self.polyBssm = polynomialBssm
    self.launchSpeed = launchSpeed = 0
    self.entrySpeed = entrySpeed = 0
    self.pts = pts

# Ball class given min/max arry, and the corresponding frame 
class Ball:
  def __init__(self, arr, frame):
    self.y = arr[3] # arr[3] is ymax for bounding box (bottom of bowling ball)
    self.x = (arr[0] + arr[2])/2 #arr[0] and arr[2] are xmin and xmax, respectivley, for bounding box
    self.frame = frame
  
  def __str__ (self):
     return "(" + str(self.x) + "," + str(self.y) + ":" + str(self.frame) + ")"

# Called when a button is cliked by setMouseCallback in render function
def click_event(event, x, y, flags, params): 
    global count, x1, x2, x3, x4, y1, y2, y3, y4, laneArr, frameLimit

    if event == cv2.EVENT_LBUTTONDOWN: # Checking for left mouse clicks 
        # Collecting points for lane class
        if count == 0:
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
            global matrix, quadArr, poly, laneSet, shotArr, realMatrix
    
            laneArr = np.array([[x1, y1], [x2, y2], 
                [x4, y4], [x3, y3], [x1, y1]],
               np.int32)
            
            quadArr = [(x1, y1), (x2, y2), (x4, y4), (x3, y3)]
            bssmDimArr = np.float32([[0, 0], [183, 0], [183, 2457], [0, 2457]])  # dim of lane on bssm window (unit px)
            laneDimArr = np.float32([[0, 0], [39, 0], [39, 671], [0, 671]])  # dim of real bowling lane (unit boards = 1.04 in)
           
            # Convert the quadrilateral and point into a format suitable for cv2 functions
            poly = np.float32(quadArr)
            matrix = cv2.getPerspectiveTransform(poly, bssmDimArr)
            realMatrix = cv2.getPerspectiveTransform(poly, laneDimArr)
            laneSet = True

            frameLimit = frameRender
          
def render(predictions, image):
    global frameLimit, laneSet, shotLimit, vidFlag, done, frameRender, ballCount, frameCount

    idArr = sv.Detections.from_roboflow(predictions).class_id  # creates an array of all class_id of detections in a given frame
    frame = extractframe(predictions)  # define current frame

    if laneSet:
        if "ball" in str(predictions):  # checks if a ball has been detected in a given frame
            ballPosArr = sv.Detections.from_roboflow(predictions).xyxy[np.where(idArr == 0)]  # creates an array of min/max values for the bounding boxes of each ball in a given frame

            frameBallArr = [Ball(element, frame) for element in ballPosArr]  # create a ball object for each element in posArr — passing in current frame_id 

            frameBallArr =  [ball for ball in frameBallArr if inside(ball.x, ball.y)]  # removes balls from frameBallArr if they are not on the lane 

            if np.size(frameBallArr) == 1:  # checks if ball is on lane, then sets frameLimit = frame
                frameLimit = frame
                vidFlag = True 
                ballCount += np.size(frameBallArr)
            
            elif np.size(frameBallArr) > 1:
                print("Error: more than one ball detected on the lane")

        if (frame - frameLimit) > 60:  # if 60 frames (2 sec) has elapsed without a ball detection on the lane, the shot will be exported to
            if ballCount >= 2:
                done = False
                vidFlag = False
                
                while not done:
                    time.sleep(0.1)  # To prevent a tight loop, we sleep for a short duration

                time.sleep(0.5)
                frameCount = 0

                inference.Stream(
                    source="output.mp4",
                    model="bowling-model/6",
                    confidence=0.1,
                    iou_threshold=0.01,
                    output_channel_order="BGR",
                    use_main_thread=True,
                    on_prediction=master
                )

                ballCount = 0
                thread = threading.Thread(target=captureBuffer)
                thread.start()

        image = cv2.polylines(image, [laneArr], False, (0, 165, 255), 3)  # draws orange lane

    else:  # if lane not defined then call click_event
        cv2.setMouseCallback("Camera", click_event) 
        frameRender = frame

    image = annotator.annotate(
        scene=image, 
        detections=sv.Detections.from_roboflow(predictions)
    )

    cv2.imshow("Camera", image)  # update Camera window
    cv2.waitKey(1)

    size = np.size(shotArr)  # calls bssm function if a shot has been saved
    if size >= 1:  
        if not(size == shotLimit):
            bssm()
            shotLimit = size

def master(predictions, image):
    global ballArr, shotArr, frameCount, percentage

    idArr = sv.Detections.from_roboflow(predictions).class_id  # creates an array of all class_id of detections in a given frame
    frame = extractframe(predictions)  # define current frame

    if frameCount == 0:
        cap = cv2.VideoCapture("output.mp4")
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    else:
        percentage = frame/(frameCount -5)
        if percentage > 1.0:
            percentage = 1.0
        if percentage < 1.0 and frameCount > 0:
            filled_length = int(50 * percentage)  # Calculate the number of 'filled' characters in the bar
            bar = '█' * filled_length + '-' * (50 - filled_length)  # Create the bar string
            print(f"\rProgress: |{bar}| {percentage*100:.2f}%", end='\r')  # Print the progress bar with percentage

    if "ball" in str(predictions) and frameCount > 0:  # checks if a ball has been detected in a given frame 
        ballPosArr = sv.Detections.from_roboflow(predictions).xyxy[np.where(idArr == 0)]  # creates an array of min/max values for the bounding boxes of each ball in a given frame
        frameBallArr = [Ball(element, frame) for element in ballPosArr]  # create a ball object for each element in posArr — passing in current frame_id 
        frameBallArr =  [ball for ball in frameBallArr if inside(ball.x, ball.y)]  # removes balls from frameBallArr if they are not on the lane 
        ballArr = np.append(ballArr, frameBallArr)  

    if frame > 5 and percentage >= 1.0 and frameCount > 0:
        shot = Shot(ballArr)
        shotArr = np.append(shotArr, shot)
        print("\n\nShot #", np.size(shotArr), " Saved With ", np.size(ballArr), "Cords")
        ballArr = np.array([])
        frameCount = -1
           
def bssm(): 
    global shotArr, currentY

    image = cv2.imread("background.png")
    recentShot = shotArr[-1]
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
    for idx, shot in enumerate(shotArr[:-1]):
        drawShotInfo(image, shot, 173 + 86 * (idx + 1), np.size(shotArr) - idx - 1)

    # Draw the most recent shot's information at y = 173
    drawShotInfo(image, recentShot, 173, np.size(shotArr))

    # Update the y-coordinate for the next shot
    currentY += 86

    cv2.imshow("BSSM", image)  # update BSSM window
    cv2.waitKey(1)

def drawShotInfo(image, shot, ycord, num):
    if ycord < 900:
        attributes = [
            shot.foulLine, shot.arrows, shot.breakPtBoard,
            shot.breakPtDis, shot.entryBoard, shot.launchAng,
            shot.impactAng, shot.launchSpeed, shot.entrySpeed
        ]
        if ycord > 200:
            color = (255, 255, 255)
        else:
            color = (0, 0, 0)

        putCentered(image, str(num), (126, ycord), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 255, 255), 4)

        # Iterate over the attributes and display them
        for i, attribute in enumerate(attributes):
            xcord = 382 + i * 256
            putCentered(image, str(attribute), (xcord, ycord), cv2.FONT_HERSHEY_SIMPLEX, 1.7, color, 3)


thread = threading.Thread(target=captureBuffer)
thread.start()

# run render function for each frame
inference.Stream(
    source=0,
    model="bowling-model/6",
    confidence=0.1,
    iou_threshold=0.01,
    output_channel_order="BGR",
    use_main_thread=True,
    on_prediction=render
)