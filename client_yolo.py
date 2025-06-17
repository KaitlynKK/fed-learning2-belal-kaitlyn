import os
import cv2
import argparse
import yaml
from ultralytics import YOLO
import flwr as fl

parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True, help="Comma-separated list of video file paths")
parser.add_argument("--model", default="model/my_model.pt", help="Path to YOLO model file")
parser.add_argument("--output", default="output", help="Directory for output files")
args = parser.parse_args()

class YOLOClient(fl.client.NumPyClient):
    def __init__(self):
        self.video_paths = args.video.split(",")
        self.model_path = args.model
        self.output_root = args.output
        self.input_root = "frames"
        os.makedirs(self.input_root, exist_ok=True)
        os.makedirs(self.output_root, exist_ok=True)

    def get_parameters(self, config):
        return []

    def set_parameters(self, parameters):
        pass

    def fit(self, parameters, config):
        print("[CLIENT] FIT STARTED")
        model = YOLO(self.model_path)

        result_total = 0

        for i, video_path in enumerate(self.video_paths):
            print(f"[CLIENT] Processing video: {video_path}")
            input_folder = os.path.join(self.input_root, f"video_{i+1}")
            output_folder = os.path.join(self.output_root, f"video_{i+1}")
            os.makedirs(input_folder, exist_ok=True)
            os.makedirs(output_folder, exist_ok=True)

            # Extract frames from video
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_path = os.path.join(input_folder, f"frame_{frame_count:04d}.jpg")
                cv2.imwrite(frame_path, frame)
                frame_count += 1
            cap.release()

            print(f"[CLIENT] Extracted {frame_count} frames from {video_path}")

            # Generate temporary data.yaml file for YOLOv8 training
            data_yaml_path = os.path.join(output_folder, "data.yaml")
            data_yaml = {
                "train": input_folder.replace("\\", "/"),
                "val": input_folder.replace("\\", "/"),
                "names": ["train"]  # placeholder class name
            }
            with open(data_yaml_path, "w") as f:
                yaml.dump(data_yaml, f)

            # Train YOLO
            result = model.train(
                data=data_yaml_path,
                epochs=1,
                imgsz=640,
                project=output_folder,
                name="train_result",
                exist_ok=True
            )
            result_total += 1

        print("[CLIENT] Training complete")
        return [], result_total, {}

    def evaluate(self, parameters, config):
        print("[CLIENT] EVALUATE CALLED")
        return 0.0, 0, {}

if __name__ == "__main__":
    client = YOLOClient()
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)

