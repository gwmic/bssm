from inference.core.interfaces.stream.inference_pipeline import InferencePipeline
import supervision as sv
import numpy as np
import threading
import videobuffer as vb
import wrappers
import cv2
import mastergui as gui
import caffeine
import sys
import subprocess


# Source num of webcam
SOURCE = 0
# Toggle the processing annotations; turning off saves 30% time vs. on
SHOW_PROCESS = False
# Confidence for infernce on live vid feed to detect shot; value ranges from 0 to 1.0
CONFIDENCE_LIVE = 0.3
# Confidence for infernce on cropped shots to be processed into data; value ranges from 0 to 1.0
CONFIDENCE_PROCESS = 0.1
# Maximum Percent of lane enclosed by the the shot curve of a spare; value ranges from 0 to 1.0
SPARE_DETECTION_FACTOR = 0.10
# Rate of frames that will be scanned; if n is selected then every n frames will be scanned (must be a natural number)
SCAN_RATE = 2
# Toggle saving each shot as an mp4 to /shot; Turning off saves storage
SAVE_SHOTS = False

# manages all global data as data object
class DataManager:
    def __init__(self, source, showProcess, confidenceLive, confidenceProcess, spareFactor, scanRate, saveShots):
        self.annotator = sv.BoxAnnotator() 
        self.spareArea = spareFactor * 39 * 671
        self.currentY = 173
        self.source = source
        self.showProcess = showProcess
        self.scanRate = scanRate
        self.saveShots = saveShots
        self.timeStr = "null"
        self.fps = 30
        self.list = self.predict = None
        self.processing = self.running = True
        self.buttons = []
        self.window = {}
        self.predictions = np.array([])
        self.printText = "Welcome to BSSM - Click The Top Left Corner of The Lane to Begin"
        self.selection = 1
        self.dragging = self.activeBowler = self.oldBowler = self.upNext = -1 
        self.done = threading.Event()
        
        self.clickArr = ["1/4 Top Left", "2/4 Top Right",
                         "3/4 Bottom Left", "4/4 Bottom Right"]
        self.confidenceLive, self.confidenceProcess = confidenceLive, confidenceProcess

        # Initialize all array attributes to empty numpy arrays
        self.ballArr = self.laneArr = self.bowlerArr = self.shotLimit = self.output_cropped = np.array([])
        self.laneDimArr = self.bssmDimArr = self.poly = np.array([])
        self.croppedPoly = self.croppedLaneArr = self._curveArr = self.rackArr = np.array([])

        # Initialize all boolean flags
        self.laneSet = self.vidFlag = self.scan = self.displayOil = self.running = False

        # Initialize all counter attributes to zero
        self.count = self.x1 = self.x2 = self.x3 = self.x4 = self.start_x = self.start_y = self.fps = 0
        self.percentage = self.y1 = self.y2 = self.y3 = self.y4 = self.frameWidth = self.bowlerCount = 0
        self.frameLimit = self.frameCount = self.frameRender = self.ballCount = self.size = 0

    @property
    def shotArr(self):
        if self.list is None or self.upNext is None:
            return np.array([])
        return self.bowlerArr[str(self.list[self.upNext])]
    
    @shotArr.setter
    def shotArr(self, arr):
        self.bowlerArr[str(self.list[self.upNext])] = arr
    
    @property
    def curveArr(self):
        if self.activeBowler == -1:
            return np.array([np.array([])])
        return self._curveArr[self.activeBowler]
    
    @curveArr.setter
    def curveArr(self, arr):
        self._curveArr[self.activeBowler] = arr

# initilize data with constants
data = DataManager(SOURCE, SHOW_PROCESS, CONFIDENCE_LIVE,
                   CONFIDENCE_PROCESS, SPARE_DETECTION_FACTOR,
                   SCAN_RATE, SAVE_SHOTS)

#Check sys.argv 
correctFormat = True
l = len(sys.argv)
if l == 1:
    print("Specify Bowler Name(s): python bssm.py <bowler1> <bowler2> ...")
    sys.exit()
if l > 7:
    print("Error: six bowler maximum")
    sys.exit()
else:
    for i in range(1, l):
        if not len(sys.argv[i]) == 2:
            print(f"Error: index {i} is not two chracters  ")
            correctFormat = False

    if correctFormat:
        data.bowlerArr = {bowler: np.array([]) for bowler in [s.upper() for s in sys.argv][1:]}
        data.list = list(data.bowlerArr.keys())
        data.bowlerCount = l - 1
        data.shotLimit = np.full(data.bowlerCount, 0)
        data._curveArr = [np.array([]) for _ in range(data.bowlerCount)]
    else:
        sys.exit() 


# load background image
data.window["Background"] = cv2.imread("background.png")

subprocess.run(
    "diskutil erasevolume HFS+ 'RAMdisk' `hdiutil attach -nomount ram://100000`", 
    shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
)

# start video buffer thread
thread = threading.Thread(target=vb.captureBuffer, args=(data,))
thread.start()

# keep laptop from sleeping
caffeine.on(display=True)

# run render function for each frame
customRender = wrappers.renderWrapper(data)
data.predict = InferencePipeline.init(
    model_id="bssm-small/2",
    video_reference=data.source,
    confidence=data.confidenceLive,
    iou_threshold=0.01,
    on_prediction=wrappers.renderWrapper(data),
    api_key="IQvYLHUhWtgWqompoERt"
)
data.predict.start(use_main_thread=False)

# set gui to update until program is stopped
while True:
   gui.renderGui(data)
   if data.running == False:
       break
data.predict.terminate()
subprocess.run(
    "diskutil eject RAMdisk", 
    shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
)