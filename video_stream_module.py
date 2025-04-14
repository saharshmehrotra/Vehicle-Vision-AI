import cv2
import torch
from threading import Thread

class VideoStreamProcessor:
    def __init__(self, src=0, use_gpu=True):
        self.capture = cv2.VideoCapture(src)
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        self.model.to(self.device)

    def process_frame(self, frame):
        results = self.model(frame)
        return results.render()[0]

    def start(self):
        while True:
            ret, frame = self.capture.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output = self.process_frame(frame_rgb)
            cv2.imshow("Multi-Stream Inference", output)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    processor = VideoStreamProcessor()
    processor.start()

