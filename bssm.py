import inference
import supervision as sv
import numpy as np
import threading
import videobuffer as vb
import wrappers

SOURCE = 1 # Source num of webcam
SHOW_PROCESS = True # Toggle the processing annotations; turning off saves 30% time vs. on
CONFIDENCE_LIVE = 0.15 # Confidence for infernce on live vid feed to detect shot; value ranges from 0 to 1.0
CONFIDENCE_PROCESS = 0.45 # Confidence for infernce on cropped shots to be processed into data; value ranges from 0 to 1.0
SPARE_DETECTION_FACTOR = 0.1 # Maximum Percent of lane enclosed by the the shot curve of a spare; value ranges from 0 to 1.0

# manages all global data as data object
class DataMan:
    def __init__(self, source, showProcess, confidenceLive, confidenceProcess, spareFactor):
        self.initArrays()
        self.initFlags()
        self.initCounters()
        self.annotator = sv.BoxAnnotator()
        self.currentY = 173
        self.source = source
        self.showProcess = showProcess
        self.timeStr = "null"
        self.fps = 30
        self.spareArea = spareFactor * 39 * 671
        self.clickArr = ["Top Left", "Top Right", "Bottom Left", "Bottom Right"]
        self.confidenceLive, self.confidenceProcess = confidenceLive, confidenceProcess

    def initArrays(self):
        # Initialize all array attributes to empty numpy arrays
        self.ballArr = self.shotArr = self.laneArr = self.laneDimArr = self.bssmDimArr = self.poly = self.croppedPoly = self.croppedLaneArr = self.curveArr = np.array([])

    def initFlags(self):
        # Initialize all boolean flags
        self.laneSet = self.vidFlag = self.done = self.scan = self.displayOil = False

    def initCounters(self):
        # Initialize all counter attributes to zero
        self.count = self.x1 = self.x2 = self.x3 = self.x4 = 0
        self.percentage = self.y1 = self.y2 = self.y3 = self.y4 = self.frameWidth = 0
        self.shotLimit = self.frameLimit = self.frameCount = self.frameRender = self.ballCount = 0

data = DataMan(SOURCE, SHOW_PROCESS, CONFIDENCE_LIVE, CONFIDENCE_PROCESS, SPARE_DETECTION_FACTOR)

thread = threading.Thread(target=vb.captureBuffer, args=(data,))
thread.start()

# run render function for each frame
customRender = wrappers.renderWrapper(data)
inference.Stream(
    source=data.source,
    model="bowling-model/6",
    confidence=data.confidenceLive,
    iou_threshold=0.01,
    output_channel_order="BGR",
    use_main_thread=True,
    on_prediction=customRender
)