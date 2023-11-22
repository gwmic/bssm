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
        point = np.float32([[ball.x, ball.y]])  # 2D point
        point = point.reshape(1, -1, 2)  # Reshaping to (1, N, 2) format for perspectiveTransform

        transformedPoint = cv2.perspectiveTransform(point, data.matrix)
        realTransformedPoint = cv2.perspectiveTransform(point, data.realMatrix)
        
        # Assigning transformed coordinates directly
        self.bssmx, self.bssmy = transformedPoint[0][0][1], transformedPoint[0][0][0]
        self.realx, self.realy = realTransformedPoint[0][0][1], realTransformedPoint[0][0][0]
        self.frame = ball.frame

class Shot:
    def __init__(self, ballArr, data):
        pts = [LaneCord(ball, data) for ball in ballArr]
        filteredpts = [(cord.bssmx, cord.bssmy, cord.realx, cord.realy) 
                        for cord in pts if cord.bssmx >= 0 and cord.bssmy >= 0]

        xBssm, yBssm, xReal, yReal = zip(*filteredpts)

        xBssm = [0, 819, 1638, 2457]
        yBssm = [173, 61, 33, 89]
        xReal = [0, 224, 447, 671]
        yReal = [37, 13, 7, 19]

        self.polyBssm = self.calculatePoly(xBssm, yBssm)
        self.polyReal = self.calculatePoly(xReal, yReal)

        self.arrows = self.calculateArrows(self.polyReal)
        self.breakPtBoard, self.breakPtDis = self.calculateBreakPt(self.polyReal)
        self.foulLine = self.calculateBoard(self.polyReal, 0)
        self.entryBoard = self.calculateBoard(self.polyReal, 671)
        self.launchAng = self.calculateAng(self.polyReal, 0, -1)
        self.impactAng = self.calculateAng(self.polyReal, 671, 1)
        self.launchSpeed = self.calculateSpeed(pts, self.polyReal, xReal, 30, 190, data)
        self.entrySpeed = self.calculateSpeed(pts, self.polyReal, xReal, 510, 670, data)
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
            raise ValueError("No valid roots found within the specified range.")

        return '%.1f' % poly(roots[0])  # Return the first valid root

    @staticmethod
    def calculateBreakPt(poly):
        LANELENGTH = 671
        SCALEFACTOR = 60 / LANELENGTH

        # Find local minimum
        minx, miny = mod.findLocalMin(poly, 0, LANELENGTH)

        if minx is None:
            # Handle the case where no local minimum is found
            raise ValueError("No valid local minimum found within the specified range.")

        # Calculate and return the break point board and distance
        break_pt_board = '%.1f' % miny
        break_pt_distance = '%.1f' % (minx * SCALEFACTOR)
        return break_pt_board, break_pt_distance

    @staticmethod
    def calculateBoard(poly, distance):
        return '%.1f'%(poly(distance))

    @staticmethod
    def calculateAng(poly, distance, multiplier):
        return '%.1f'%(multiplier * mod.tan(poly, distance))

    @staticmethod
    def calculateSpeed(pts, poly, xReal, start, end, data):
        # Define constants for clarity
        FPSTOHOURS = 3600/data.fps  # Frames per second to hours conversion factor
        BOARDSTOMILES = (60 / 671) / 5280  # Convert bowling lane boards to miles
        #                ^^^^^^^^ 60 feet are in a 671 boards (length of lane)

        startIndex = mod.findClosest(xReal, start)
        endIndex = mod.findClosest(xReal, end)

        if startIndex == endIndex:
            raise ValueError("Start and end points are too close or identical.")

        start_x = xReal[startIndex]
        end_x = xReal[endIndex]
        time_hours = (pts[endIndex].frame - pts[startIndex].frame) * FPSTOHOURS
        distance_miles = mod.arcLength(poly, start_x, end_x) * BOARDSTOMILES

        return '%.1f' % (distance_miles / time_hours)