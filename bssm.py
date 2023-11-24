import inference
import supervision as sv
import numpy as np
import threading
import videobuffer as vb
import wrappers

SOURCE = 0 # Source num of webcam
SHOWPROCESS = True # Toggle the processing annotations; turning off saves 30% time vs. on
CONFIDENCELIVE = 0.2 # Confidence for infernce on live vid feed to detect shot; value ranges from 0 to 1.0
CONFIDENCEPROCESS = 0.45 # Confidence for infernce on cropped shots to be processed into data; value ranges from 0 to 1.0

# manages all global data as data object
class DataMan:
    def __init__(self, source, showProcess, confidenceLive, confidenceProcess):
        self.init_arrays()
        self.init_flags()
        self.init_counters()
        self.annotator = sv.BoxAnnotator()
        self.currentY = 173
        self.source = source
        self.showProcess = showProcess
        self.timeStr = "null"
        self.clickArr = ["Top Left", "Top Right", "Bottom Left", "Bottom Right"]
        self.confidenceLive, self.confidenceProcess = confidenceLive, confidenceProcess

    def init_arrays(self):
        # Initialize all array attributes to empty numpy arrays
        self.ballArr = self.shotArr = self.laneArr = self.laneDimArr = self.bssmDimArr = self.poly = self.croppedPoly = self.croppedLaneArr = self.curveArr = np.array([])

    def init_flags(self):
        # Initialize all boolean flags
        self.laneSet = self.vidFlag = self.done = self.scan = False

    def init_counters(self):
        # Initialize all counter attributes to zero
        self.count = self.x1 = self.x2 = self.x3 = self.x4 = 0
        self.percentage = self.y1 = self.y2 = self.y3 = self.y4 = self.frameWidth = 0
        self.shotLimit = self.frameLimit = self.frameCount = self.frameRender = self.ballCount = self.fps = 0

data = DataMan(SOURCE, SHOWPROCESS, CONFIDENCELIVE, CONFIDENCEPROCESS)

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