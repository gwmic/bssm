import cv2
import inference
import supervision as sv
import numpy as np
import re
import warnings

annotator = sv.BoxAnnotator()

# Define global variables 
global laneArr, ballArr, frameLimit, quadArr, poly, laneSet, shotArr, matrix, realMatrix, shotLimit
ballArr = shotArr = np.array([])
laneSet = False
count = x1 = x2 = x3 = x4 = y1 = y2 = y3 = y4 = shotLimit = frameLimit = 0

class LaneCord:
  def __init__(self, ball):
    global matrix
    global realMatrix

    pt = np.float32([[ball.x, ball.y]])  # Changed to a 2D point
    pt = np.array([pt])  # Reshaping to (1, N, 2) format
    transPt = cv2.perspectiveTransform(pt, matrix)
    
    realTransPt = cv2.perspectiveTransform(pt, realMatrix)
    
    self.bssmx = 2457 - transPt[0][0][1] # bssmx and bssmy are the x and y cords scaled to the lane at the bottom of the bssm window (unit is pixles)
    self.bssmy = transPt[0][0][0]
    self.realx = 773 - realTransPt[0][0][1] # realx and realy are the x and y cords scaled to the dims of a bowling lane (unit is boards)
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

    #xBssm = [0, 1876, 938, 2415]
    #yBssm = [117, 42, 61, 120]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', np.RankWarning)
        coefficientsBssm = np.polyfit(xBssm, yBssm, 3)
        polynomialBssm = np.poly1d(coefficientsBssm)
    
        coefficientsReal = np.polyfit(xReal, yReal, 3)
        polynomialReal = np.poly1d(coefficientsReal)

    '''
    intersectionPoly = np.poly1d([*polynomialReal.c[:-1] + 0.534, polynomialReal.c[-1] - 446])  # Rearrange to form a polynomial equation: polynomial(x) - line_slope * x - line_intercept = 0
    intersectionPtsX = np.roots(intersectionPoly)  # Find the roots of this polynomial, which are the x-coordinates of the intersection points
    intersectionPtsX = [pt for pt in intersectionPtsX if 490 <= pt <= 690]  # remove x-intercepts if they do not fall within the x range of the arrows 
    arrows = polynomialReal(intersectionPtsX[0]) # find board where ball crosses arrows
    '''
    
    self.foulLine = round(polynomialReal(0))
    self.arrows = round(polynomialReal(590))
    self.polyBssm = polynomialBssm
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
    global count, x1, x2, x3, x4, y1, y2, y3, y4, laneArr

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
            laneDimArr = np.float32([[0, 0], [39, 0], [39, 773], [0, 773]])  # dim of real bowling lane (unit boards = 1.04 in)
           
            # Convert the quadrilateral and point into a format suitable for cv2 functions
            poly = np.float32(quadArr)
            matrix = cv2.getPerspectiveTransform(poly, bssmDimArr)
            realMatrix = cv2.getPerspectiveTransform(poly, laneDimArr)
            laneSet = True
          
def render(predictions, image):
    global ballArr
    global frameLimit
    global laneSet
    global shotArr
    global shotLimit

    idArr = sv.Detections.from_roboflow(predictions).class_id  # creates an array of all class_id of detections in a given frame
    frame = extractframe(predictions)  # define current frame

    if laneSet:
        if "ball" in str(predictions):  # checks if a ball has been detected in a given frame
            ballPosArr = sv.Detections.from_roboflow(predictions).xyxy[np.where(idArr == 0)]  # creates an array of min/max values for the bounding boxes of each ball in a given frame

            frameBallArr = [Ball(element, frame) for element in ballPosArr]  # create a ball object for each element in posArr — passing in current frame_id 

            frameBallArr =  [ball for ball in frameBallArr if inside(ball.x, ball.y)]  # removes balls from frameBallArr if they are not on the lane 

            if np.size(frameBallArr) >= 1:  # checks if ball is on lane, then sets frameLimit = frame
                frameLimit = frame 

            ballArr = np.append(ballArr, frameBallArr)

        if (frame - frameLimit) > 60:  # if 60 frames (2 sec) has elapsed without a ball detection on the lane, the shot will be exported

            ballSize = np.size(ballArr)

            if ballSize >= 4: # checks if shot contains four or more balls 
                
                shot = Shot(ballArr)
                print(shot.foulLine)

                shotArr = np.append(shotArr, shot)
                
                print("Shot #", np.size(shotArr), " Saved")
                ballArr = np.array([])

                '''# debug
                print(np.size(shot.pts))
                for pt in shot.pts:
                    print(pt.x,"  ",pt.y)
                    print("printed")'''

            elif ballSize >= 2: # if less than four then the shot is delted, if ball is an arrray then errror is thrown
                frameLimit = frame 
                print("Shot not saved; only ", ballSize, " balls in class")

        image = cv2.polylines(image, [laneArr], False, (0, 165, 255), 3)  # draws orange lane

    else:  # if lane not defined then call click_event
        cv2.setMouseCallback("Camera", click_event) 

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

def bssm(): 
    global shotArr

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

    cv2.imshow("BSSM", image)  # update BSSM window
    cv2.waitKey(1)

# run render function for each frame
print("Please select lane coordinates")
inference.Stream(
    source=0,
    model="bowling-model/5",
    confidence=0.2,
    iou_threshold=0.01,
    output_channel_order="BGR",
    use_main_thread=True,
    on_prediction=render
)