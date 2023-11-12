import cv2
import inference
import supervision as sv

annotator = sv.BoxAnnotator()

def render(predictions, image):
    print(predictions)
    
    image = annotator.annotate(
        scene=image, 
        detections=sv.Detections.from_roboflow(predictions)
    )

    cv2.imshow("BSSM", image)
    cv2.waitKey(1)

inference.Stream(
    source=0,
    model="bowling-model/3",
    confidence=0.2,
    output_channel_order="BGR",
    use_main_thread=True,
    on_prediction=render
)