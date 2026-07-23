import cv2

class Camera:
    def __init__(self, source=0):
        self.source = source
        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Unable to open video source: {source}"
            )


    def read_frame(self):
        success, frame = self.cap.read()
        return success, frame

    
    def read_frames(self):
        while True:
            success, frame = self.cap.read()
            if not success:
                break
            yield frame

    def get_width(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    def get_height(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def get_fps(self):
        return self.cap.get(cv2.CAP_PROP_FPS)

    def get_frame_count(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def release(self):
        if self.cap is not None:
            self.cap.release()

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release() 


# # import os
# video_path = '/home/gautam/EtE_traffic_detection/videos/samsung_traffic_vid.mp4'

# camera = Camera(source=video_path)
# print(camera.read_frames())
# print(camera.get_width())

# cap = cv2.VideoCapture(video_path)
# print(cap)