import cv2
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
import re
import mastergui as gui


def putCentered(img, text, center, font, fontScale, color, thickness):
    # Get the text size
    textSize, _ = cv2.getTextSize(text, font, fontScale, thickness)
    # Calculate the bottom-left corner of the text
    textx = center[0] - textSize[0] // 2
    texty = center[1] + textSize[1] // 2

    # Put the text on the image
    cv2.putText(img, text, (textx, texty), font, fontScale, color, thickness)


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

        putCentered(image, str(num), (126, ycord),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 255, 255), 4)

        # Iterate over the attributes and display them
        for i, attribute in enumerate(attributes):
            xcord = 382 + i * 256
            putCentered(image, str(attribute), (xcord, ycord),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.7, color, 3)


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
    closestIndex = 0
    minDiff = float('inf')

    for i, value in enumerate(arr):
        diff = abs(value - target)
        if diff < minDiff:
            minDiff = diff
            closestIndex = i

    return closestIndex


def findLocalMin(poly, rangeMin, rangeMax):
    # Objective function
    def objective_function(x):
        return -1 * poly(x)

    # Find local minimum
    result = minimize_scalar(objective_function, bounds=(
        rangeMin, rangeMax), method='bounded')

    if result.success:
        return result.x, result.fun
    else:
        return None, None


def tan(poly, xPoint):
    # Derivative of the polynomial
    derivative = np.polyder(poly)

    # Slope of the tangent at xPoint
    slope = derivative(xPoint)

    # Angle of the tangent from the horizontal line
    angleRadians = np.arctan(slope)
    angleDegrees = np.degrees(angleRadians)

    return angleDegrees

def inside(poly, x, y):
    # Use cv2.pointPolygonTest to check if the point is inside the polygon
    result = cv2.pointPolygonTest(poly, (x, y), False)

    return result >= 0  # result > 0: inside, result = 0: on the edge, result < 0: outside


def drawProgressBar(img, progress, barHeight, barColor, textColor, font, fontScale, thickness):
    # Draw the progress bar
    _, imgWidth = img.shape[:2]
    barWidth = int(imgWidth * progress)
    cv2.rectangle(img, (0, 0), (barWidth, barHeight), barColor, -1)

    # Put the progress text
    text = f"{int(progress * 100)}%"
    textSize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    textx = (imgWidth - textSize[0]) // 2
    texty = (barHeight + textSize[1]) // 2
    cv2.putText(img, text, (textx, texty), font,
                fontScale, barColor, thickness + 3)
    cv2.putText(img, text, (textx, texty), font,
                fontScale, textColor, thickness)


def cliProgress(percentage, string, data):
    filledLength = int(30 * percentage)
    bar = '+' * filledLength + '-' * (30 - filledLength)

    # save the progress bar with percentage
    gui.printGui(f"{string}: |{bar}| {(percentage*100):.2f}%", data)  


def findArea(poly, x1, x2):
    # Calculate the area under the polynomial curve
    areaCurve, _ = quad(poly, x1, x2)

    # Calculate the area of the trapezoid
    y1, y2 = poly(x1), poly(x2)
    areaTrapezoid = 0.5 * (y1 + y2) * (x2 - x1)

    # The enclosed area is the difference between the two
    enclosedArea = areaCurve - areaTrapezoid

    return enclosedArea

def extractFrame(input_file, output_file, frame_number):
    # Capture the video from the input file
    cap = cv2.VideoCapture(input_file)

    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: Unable to open the input file.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID' if mp4v doesn't work
    out = cv2.VideoWriter(output_file, fourcc, frame_rate, (frame_width, frame_height))

    # Set the frame position and read the frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read the frame.")
        return

    # Write the frame to the output file
    out.write(frame)

    # Release everything
    cap.release()
    out.release()