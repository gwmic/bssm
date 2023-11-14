import cv2
import inference
import supervision as sv
import numpy as np

annotator = sv.BoxAnnotator()

# Define global variables for lane class
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

''' class Bowler:
  def __init__(self, str):
    self.x = 
    self.y = 
    self.length = 
    self.width =

class Ball:
  def __init__(self, str):
    self.x = 
    self.y = 
    self.length = 
    self.width =
'''

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

    # Checking for left mouse clicks 
    if event == cv2.EVENT_LBUTTONDOWN: 

        # Collecting points for lane class
        if count >= 4:
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

           # creating lane class, then assigning values to laneArr to be used in cv2.polylines    
           global lane
           lane = Lane(x1, y1, x2, y2, x3, y3, x4, y4)

           laneArr = np.array([[lane.x1, lane.y1], [lane.x2, lane.y2], 
                [lane.x4, lane.y4], [lane.x3, lane.y3], [lane.x1, lane.y1]],
               np.int32)
           
           print("Bottom Right: (",x,",",y,")")
           count += 1

def render(predictions, image):
    # print(predictions)
    
    image = annotator.annotate(
        scene=image, 
        detections=sv.Detections.from_roboflow(predictions)
    )
    # If all the lane class is defined then cv2 will draw its trapizoid
    if count >= 4:
        image = cv2.polylines(image, [laneArr], False, (0, 165, 255), 3)

    cv2.imshow("BSSM", image) 
    cv2.waitKey(1)
    cv2.setMouseCallback("BSSM", click_event) 
    

inference.Stream(
    source=0,
    model="bowling-model/3",
    confidence=0.2,
    output_channel_order="BGR",
    use_main_thread=True,
    on_prediction=render
)