import numpy as np
import cv2
import modules as mod

# Called when a button is cliked by setMouseCallback in render function
def clickEvent(event, x, y, flags, data):
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

class Curve:
  def __init__(self, arr):
    self.arr = arr

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
    curveObj = Curve(curve)

    data.curveArr = np.append(data.curveArr, curveObj)

    # Draw the curve on the image
    for i in range(np.size(data.curveArr)):
        if i == np.size(data.curveArr) - 1:
            image = cv2.polylines(image, data.curveArr[i].arr, False, (252, 255, 63), 2)
        else:
            image = cv2.polylines(image, data.curveArr[i].arr, False, (255, 255, 255), 2)

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