import cv2
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
import re

def putCentered(img, text, center, font, font_scale, color, thickness):
    # Get the text size
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    # Calculate the bottom-left corner of the text
    text_x = center[0] - text_size[0] // 2
    text_y = center[1] + text_size[1] // 2

    # Put the text on the image
    cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness)

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
        return -1 * poly(x)

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

def extractframe(predictions):
    match = re.search(r"'frame_id': (\d+)", str(predictions))  # Regular expression to find "frame:" followed by an integer

    if match:
        return int(match.group(1))  # Extract and return the integer
    else:
        return -1  # Return -1 if no frame is found
    
def inside(poly, x, y):
    result = cv2.pointPolygonTest(poly, (x, y), False)  # Use cv2.pointPolygonTest to check if the point is inside the polygon

    return result >= 0 # result > 0: inside, result = 0: on the edge, result < 0: outside

def drawProgressBar(img, progress, bar_height, bar_color, text_color, font, font_scale, thickness):
    # Draw the progress bar
    img_height, img_width = img.shape[:2]
    bar_width = int(img_width * progress)
    cv2.rectangle(img, (0, 0), (bar_width, bar_height), bar_color, -1)

    # Put the progress text
    text = f"{int(progress * 100)}%"
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (img_width - text_size[0]) // 2
    text_y = (bar_height + text_size[1]) // 2
    cv2.putText(img, text, (text_x, text_y +1), font, font_scale, bar_color, thickness + 3)
    cv2.putText(img, text, (text_x, text_y +1), font, font_scale, text_color, thickness)

def cliProgress(percentage, string):
    filled_length = int(50 * percentage)
    bar = '█' * filled_length + '-' * (50 - filled_length)
    print(f"\r{string}: |{bar}| {(percentage*100):.2f}%", end='\r') # save the progress bar with percentage