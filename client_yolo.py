# client_yolo.py
from ultralytics import YOLO
import cv2
import flwr as fl
import numpy as np
import base64
import io

class YOLOClient(fl.client.NumPyClient):
    def __init__(self, model_path, video_path):
        self.model = YOLO(model_path)
        self.video_path = video_path

    def get_parameters(self, config):
        return []

    def set_parameters(self, parameters):
        pass  # Not used for inference

    def fit(self, parameters, config):
        # Perform inference
        cap = cv2.VideoCapture(self.video_path)
        frame_count = 0
        detections = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_count > 5:  # Just process 5 frames
                break
            results = self.model(frame)
            det_frame = results[0].plot()

            # Encode frame to base64
            _, buffer = cv2.imencode('.jpg', det_frame)
            b64 = base64.b64encode(buffer).decode("utf-8")
            detections.append(b64)

            frame_count += 1

        cap.release()
        return [], len(detections), {"detections": detections}

    def evaluate(self, parameters, config):
        return 0.0, 0, {}

def main():
    fl.client.start_numpy_client(server_address="localhost:8080", client=YOLOClient("my_model.pt", "video.mp4"))
