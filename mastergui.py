import cv2
import numpy as np
import callables


def printGui(str, data):
    data.printText = str

def overlay(topImg, bottomImg):
    # Get dimensions of the smaller image
    height, width = topImg.shape[:2]

    # Overlay the smaller image onto the larger image at the top left corner
    bottomImg[0:height, 0:width] = topImg

    return bottomImg

# Once the dictionary is fully defined it will have three definitions at max:
# {"BSSM":image, "Camera":image, "Process Region":image, "Heatmap":image}
def processSelection(data):
    image = np.zeros((1298, 2560, 3), np.uint8)

    if data.selection == 1:
        if "Camera" in data.window:
            cameraImg = data.window["Camera"]
            return overlay(cameraImg, image)
        else:
            return image 

    elif data.selection == 2:
        if "BSSM" in data.window:
            bssmImg = data.window["BSSM"]
        else:    
            bssmImg = data.window["Background"]
        image = overlay(bssmImg, image)

        # Draw the buttons
        for i, (x, y, w, h) in enumerate(data.buttons):
            if i == data.activeBowler:
                colorRect = (158, 245, 22)
                colorText = (0, 0, 0)
                size = 3
                cv2.rectangle(image, (x, y), (x + w, y + h), colorRect, -1)
                cv2.rectangle(image, (x, y), (x + w, y + h), colorRect, 3)
            else:
                colorRect =  colorText = (252, 255, 63)
                size = 2
                cv2.rectangle(image, (x, y), (x + w, y + h), colorRect, 3)

            if i < len(data.bowlerArr):
                cv2.putText(image, data.list[i], (x + 10, y + h - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.1, colorText, size)
            if i == 0 and data.upNext != -1:
                cv2.putText(image, f"Up Next: {data.list[data.upNext]}", (x-200, 1259), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        return image

    elif data.selection == 3:
        if  "Heatmap" in data.window:
            oilImg = data.window["Heatmap"]
            return overlay(oilImg, image)
        else:
            data.selection = 2
            #printGui(f"({np.size(data.shotArr)}/5) Oil Heatmap Requires Five Strike Shots", data)

    return image

def renderGui(data):
    WINDOW_WIDTH, WINDOW_HEIGHT = 2560, 1280
    UI_WIDTH, UI_HEIGHT = int((500/3) * data.bowlerCount + 10), 75
    BORDER_BUFFER = 15
    BUTTON_WIDTH, BUTTON_HEIGHT = 150, 60

    if not data.running:
        data.start_x = WINDOW_WIDTH - UI_WIDTH + BORDER_BUFFER
        data.start_y = WINDOW_HEIGHT - UI_HEIGHT + BORDER_BUFFER
        for i in range(data.bowlerCount):
            x = data.start_x + i * (BUTTON_WIDTH + BORDER_BUFFER)
            y = data.start_y
            data.buttons.append((x, y, BUTTON_WIDTH, BUTTON_HEIGHT))
        data.running = True

    key = cv2.waitKey(1) & 0xFF

    #print(f"laneSet:{data.laneSet} window:{data.selection} dragging: {data.dragging}")
    if not data.laneSet and data.selection == 1 and data.dragging != -1:
        x, y = data.poly[data.dragging]
        if key == 2:  # Left arrow
            x -= 5
        elif key == 0:  # Up arrow
            y -= 5
        elif key == 3:  # Right arrow
            x += 5
        elif key == 1:  # Down arrow
            y += 5

        data.poly[data.dragging] = (x, y)
        if data.dragging == 0:
            data.laneArr[0] = data.laneArr[4] = [x, y]
        else:
            data.laneArr[data.dragging] = [x, y]

    # Update selection based on key press
    if key in [ord(str(i)) for i in range(1, 4)]:
        if key - ord('0') == 2:
            if data.laneSet:
                data.selection = 2
            else:
                printGui("Cannot Go to BSSM Window Until Lane Set", data)
        else:
            data.selection = key - ord('0')
    
    if key == ord('0'):
        if data.count == 4:
            data.laneSet = True
            printGui("Lane Set - Select Active Bowler", data)
            data.selection = 2
        else:
            printGui(f"Select The {data.clickArr[data.count][4:]} Corner Before Saving", data)

    if key == ord('q'):
        data.running = False

    if data.showProcess and data.ballCount == -1:
        data.selection = 2
        if "Process Region" in data.window:
            processImg = data.window["Process Region"]
            image = overlay(processImg, image)
        else:
            image = np.zeros((1298, 2560, 3), np.uint8)

    else:
        image = processSelection(data)

    cv2.putText(image, f"FPS: {'%.1f' % data.fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 0), 6)
    cv2.putText(image, f"FPS: {'%.1f' % data.fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 3)
    
    cv2.putText(image, data.printText, (10, 1259), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("BSSM", image)
    cv2.waitKey(1)

    # Process if saved shots
    size = np.size(data.shotArr)
    if size >= 1 and size != data.shotLimit[data.upNext]:
        data.shotLimit[data.upNext] = size
        print("running bssm")
        recentShot = data.shotArr[-1]
        data.oldBowler = data.upNext
        data.activeBowler = data.upNext
        callables.bssm(data, True)

        if recentShot.spare or recentShot.strike:
            print("next bowler")
            data.upNext +=1
            if data.upNext == len(data.list):
                data.upNext = 0

    # call bssm if bowler is switched
    if data.activeBowler != data.oldBowler:
        callables.bssm(data, False)
        data.oldBowler = data.activeBowler