import os
import cv2

from camera import Camera
from detector import Detector

src_dir = os.path.dirname(__file__)
project_root = os.path.dirname(src_dir)

model_path = '/home/gautam/EtE_traffic_detection/models/yolo11n.pt'
video_path = '/home/gautam/EtE_traffic_detection/videos/samsung_traffic_vid.mp4'
output_path = '/home/gautam/EtE_traffic_detection/results/detected_traffic.mp4'

def main():
    camera = Camera(video_path)
    detector = Detector(model_path)

    # It is a 4-character identifier that tells a video program which codec (encoder/decoder) to use.
    # Codec compresses the each frames (1080 x 1920 x 3)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        camera.get_fps(),
        (camera.get_width(), camera.get_height()), 
    )

    try:
        while True:
            success, frame = camera.read_frame()
            if not success:
                break

            annotated_frame = detector.annotate(frame)
            cv2.imshow("Traffic Detection", annotated_frame)

            
            if cv2.waitKey(1) == 27:
                break
    finally:
        camera.release()
        writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()