import numpy as np
import cv2
import modules as mod
import oilheatmap as ohm
import mastergui as gui


# Called when a button is cliked by setMouseCallback in render function
def clickEvent(event, x, y, flags, data):
    if event == cv2.EVENT_LBUTTONDOWN:  # Checking for left mouse clicks camera window is selected
        if data.selection == 1:
            # Collecting points for lane class
            if data.count < 4:
                setattr(data, f'x{data.count+1}', x)
                setattr(data, f'y{data.count+1}', y)
                message = " - Click a Corner to Edit or Hold '0' to Confirm" if data.count == 3 else f" - Select {data.clickArr[data.count+1][4:]} Next"
                gui.printGui(f"{data.clickArr[data.count]}: ({x}, {y}){message}", data)
                data.count += 1

                if data.count == 4:
                    # Once all points are collected, perform further processing
                    data.laneArr = np.array([[data.x1, data.y1], [data.x2, data.y2],
                                            [data.x4, data.y4], [data.x3, data.y3],
                                            [data.x1, data.y1]], np.int32)

                    quadArr = [(data.x1, data.y1), (data.x2, data.y2),
                            (data.x4, data.y4), (data.x3, data.y3)]
                    data.bssmDimArr = np.float32(
                        [[0, 0], [183, 0], [183, 2457], [0, 2457]])
                    data.laneDimArr = np.float32(
                        [[0, 0], [39, 0], [39, 671], [0, 671]])

                    # Convert the quadrilateral and point into a format suitable for cv2 functions
                    data.poly = np.float32(quadArr)

                    data.frameLimit = data.frameRender
            else:
                for i, point in enumerate(data.poly):
                    if abs(x - point[0]) < 100 and abs(y - point[1]) < 100:
                        data.dragging = i  # Start dragging this point
                        if data.dragging == 2 or data.dragging == 3:
                            num = 3 if data.dragging == 2 else 2
                        else:
                            num = int(data.dragging)
                        print(num)
                        gui.printGui(f"{(data.clickArr[num])[4:]}; Use Arrow Keys to Move - Press '0' to Confirm, or Use Mouse to Select a Diffrent Corner", data)

        elif data.selection == 2:
            if event == cv2.EVENT_LBUTTONDOWN:
                for i, (bx, by, bw, bh) in enumerate(data.buttons):
                    if bx <= x <= bx + bw and by <= y <= by + bh:
                        if data.activeBowler == -1:
                            data.processing = False
                            data.upNext = i
                            gui.printGui(f"Bowler {data.list[i]} Selected - READY", data)

                        if data.processing:
                            gui.printGui("Cannot Change Bowler While Processing", data)

                        if flags & cv2.EVENT_FLAG_CTRLKEY:
                            data.upNext = i 
                        else:
                            data.activeBowler = i
                        break
             
class Curve:
    def __init__(self, arr, color):
        self.arr = arr
        self.color = color


def bssm(data, flag):
    image = np.copy(data.window["Background"])

    if np.size(data.shotArr) >= 1:
        recentShot = data.shotArr[-1]

        if flag:
            polyBssm = recentShot.polyBssm

            # Generate curve points more efficiently
            curve = np.array([], dtype=np.int32)
            for x in range(0, 2457):
                y = int(polyBssm(x))
                curve = np.append(curve, [x, 970 + y])

            curveReshaped = curve.reshape(-1, 2)
            curve = np.array([curveReshaped], dtype=np.int32)
            if recentShot.strike:
                color = (255, 255, 255)#white for strikes
            else:
                color = (0, 0, 0)#black for non strikes
            curveObj = Curve(curve, color)

            data.curveArr = np.append(data.curveArr, curveObj)

        # Draw the curves on the image
        for i in range(np.size(data.curveArr)):
            # Determine the color based on the condition
            if i == np.size(data.curveArr) - 1:
                if recentShot.spare:
                    c = (0, 0, 255)  # Red for the last element if a spare
                else:
                    c = (252, 255, 63)  # Blue for the recent shot
            else:
                c = data.curveArr[i].color

            # Draw the polyline with the determined color
            image = cv2.polylines(image, data.curveArr[i].arr, False, c, 2)

        # Create a filtered list of shots that are not spares
        nonSpares = [shot for shot in data.shotArr[:-1] if shot.strike]
        l = len(nonSpares)

        # Draw information for non-spare shots
        for idx, shot in enumerate(nonSpares):
            mod.drawShotInfo(image, shot, 173 + 86 * (idx + 1), l - idx)

        # Draw the most recent shot's information
        if recentShot.spare:
            num = "SPARE"
        else:
            num = l+1
        mod.drawShotInfo(image, recentShot, 173, num)

        # After displaying the spare's stats — remove it, so it won't be displayed on the chart
        if recentShot.spare:
            data.curveArr = data.curveArr[:-1]

        # Update the y-coordinate for the next shot
        data.currentY += 86

        if np.size(data.shotArr) >= 50:
            ohm.heatMap(data)
            gui.printGui("Oil Heatmap Ready; Hold '3'", data)
        else:
            gui.printGui(f"Shot # {np.size(data.shotArr)} Saved With {data.size} Coords - READY", data)
    
    # Display the updated image
    data.window["BSSM"] = image