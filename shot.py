import warnings 
import numpy as np
import modules as mod
import cv2

class Ball:
  def __init__(self, arr, frame):
    self.y = arr[3] # arr[3] is ymax for bounding box (bottom of bowling ball)
    self.x = (arr[0] + arr[2])/2 #arr[0] and arr[2] are xmin and xmax, respectivley, for bounding box
    self.frame = frame

class LaneCord:
    def __init__(self, ball, data):
        matrix = cv2.getPerspectiveTransform(data.croppedPoly, data.bssmDimArr)
        realMatrix = cv2.getPerspectiveTransform(data.croppedPoly, data.laneDimArr)
        point = np.float32([[ball.x, ball.y]])  # 2D point
        point = point.reshape(1, -1, 2)  # Reshaping to (1, N, 2) format for perspectiveTransform

        transformedPoint = cv2.perspectiveTransform(point, matrix)
        realTransformedPoint = cv2.perspectiveTransform(point, realMatrix)
        
        # Assigning transformed coordinates directly
        self.bssmx, self.bssmy = 2457 - transformedPoint[0][0][1], transformedPoint[0][0][0]
        self.realx, self.realy = 671 - realTransformedPoint[0][0][1], realTransformedPoint[0][0][0]
        self.frame = ball.frame

class Shot:
    def __init__(self, ballArr, data):

        pts = [LaneCord(ball, data) for ball in ballArr]
        filteredpts = [(cord.bssmx, cord.bssmy, cord.realx, cord.realy) 
                        for cord in pts if cord.realx >= 25 and cord.realy >= 0] #filters points to only be past the dots on the lane
        if not filteredpts:
            self.polyBssm = self.polyReal = self.arrows = self.breakPtBoard = self.breakPtDis = self.foulLine = self.entryBoard = self.launchAng = self.impactAng = self.launchSpeed = self.entrySpeed = "ERR0"
        else:
            xBssm, yBssm, xReal, yReal = zip(*filteredpts)

            self.polyBssm = self.calculatePoly(xBssm, yBssm)
            self.polyReal = self.calculatePoly(xReal, yReal)

            self.arrows = self.calculateArrows(self.polyReal)
            self.breakPtBoard, self.breakPtDis = self.calculateBreakPt(self.polyReal)
            self.foulLine = self.calculateBoard(self.polyReal, 0)
            self.entryBoard = self.calculateBoard(self.polyReal, 671)
            self.launchAng = self.calculateAng(self.polyReal, 0, 1)
            self.impactAng = self.calculateAng(self.polyReal, 671, -1)
            self.launchSpeed = self.calculateSpeed(pts, self.polyReal, xReal, 30, 670, 0.73, data)
            self.entrySpeed = self.calculateSpeed(pts, self.polyReal, xReal, 30, 670, 0.71, data)
            self.pts = pts  # Storing the processed points for potential future use

    @staticmethod
    def calculatePoly(x, y):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', np.RankWarning)
            coefficients = np.polyfit(x, y, 4)
            return np.poly1d(coefficients)

    @staticmethod
    def calculateArrows(poly):
        def find_roots(poly, coefficients):
            intersectionPoly = poly - np.poly1d(coefficients)
            roots = np.roots(intersectionPoly)
            return [root.real for root in roots if np.isreal(root) and 139.96 <= root.real <= 166.45]

        rootsRight = find_roots(poly, [0.736, -103.04])
        rootsLeft = find_roots(poly, [-0.736, 142.04])
        roots = rootsRight + rootsLeft

        if not roots:
            # Handle the case where no valid roots are found
            return "ERR1"

        return '%.1f' % (39 - poly(roots[0]))  # Return the first valid root

    @staticmethod
    def calculateBreakPt(poly):
        LANELENGTH = 671
        SCALEFACTOR = 60 / LANELENGTH

        # Find local minimum
        minx, miny = mod.findLocalMin(poly, 0, LANELENGTH)

        if minx is None:
            # Handle the case where no local minimum is found
            return "ERR1"
        
        # Calculate and return the break point board and distance
        break_pt_board = '%.1f' % (39 + miny)
        break_pt_distance = '%.1f' % (minx * SCALEFACTOR)
        return break_pt_board, break_pt_distance

    @staticmethod
    def calculateBoard(poly, distance):
        return '%.1f'%(39 - poly(distance))

    @staticmethod
    def calculateAng(poly, distance, multiplier):
        return '%.1f'%(multiplier * mod.tan(poly, distance))

    @staticmethod
    def calculateSpeed(pts, poly, xReal, start, end, multiplier, data):
        # Define constants for clarity
        FRAMESTOHOURS = (1/data.fps)/3600  # Frames per second to hours conversion factor
        BOARDSTOMILES = (60 / 671) / 5280  # Convert bowling lane boards to miles
        #                ^^^^^^^^ 60 feet are in a 671 boards (length of lane)

        startIndex = mod.findClosest(xReal, start)
        endIndex = mod.findClosest(xReal, end)

        if startIndex == endIndex:
            return "ERR1"

        startx = xReal[startIndex]
        endx = xReal[endIndex]

        if endIndex > np.size(pts):
            return "ERR2"
        
        timehours = (pts[endIndex].frame - pts[startIndex].frame) * FRAMESTOHOURS
        if timehours < 0:
            timehours = (pts[endIndex].frame + (data.frameCount - pts[startIndex].frame)) * FRAMESTOHOURS
        distancemiles = mod.arcLength(poly, startx, endx) * BOARDSTOMILES

        if timehours == 0:
            return "ERR3"

        return '%.1f' % ((distancemiles / timehours)*multiplier)