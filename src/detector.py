from ultralytics import YOLO

vehicle_classes = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

class Detector:
    def __init__(self, model_path, confidence=0.4):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.class_ids = list(vehicle_classes.keys())

    def detect(self, frame):
        results = self.model(
            frame,
            classes=self.class_ids,
            conf=self.confidence,
            verbose=False
        )
        return results[0]


    def annotate(self, frame):
        result = self.detect(frame)
        return result.plot()

# detect = detector()
# print(detect.)