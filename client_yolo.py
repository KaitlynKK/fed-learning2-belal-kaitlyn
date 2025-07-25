import os
import cv2
import argparse
import yaml
from ultralytics import YOLO
import flwr as fl
from pathlib import Path

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

        for i, video_path in enumerate(self.video_paths):
            print(f"[CLIENT] Processing video: {video_path}")
            video_name = f"video_{i+1}"
            input_folder = os.path.join(self.input_root, video_name)
            output_folder = os.path.join(self.output_root, video_name)

            train_folder = os.path.join(input_folder, "train")
            val_folder = os.path.join(input_folder, "val")
            os.makedirs(train_folder, exist_ok=True)
            os.makedirs(val_folder, exist_ok=True)
            os.makedirs(output_folder, exist_ok=True)

            # Extract frames from video
            cap = cv2.VideoCapture(video_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()

            print(f"[CLIENT] Extracted {len(frames)} frames from {video_path}")
            if len(frames) == 0:
                continue

            split_idx = int(0.8 * len(frames))
            for idx, frame in enumerate(frames):
                folder = train_folder if idx < split_idx else val_folder
                frame_name = f"frame_{idx:04d}.jpg"
                frame_path = os.path.join(folder, frame_name)
                label_path = frame_path.replace(".jpg", ".txt")

                cv2.imwrite(frame_path, frame)

                # Generate dummy label if not present
                if not os.path.exists(label_path):
                    with open(label_path, "w") as f:
                        f.write("0 0.5 0.5 0.3 0.3\n")

            # Write data.yaml
            data_yaml_path = os.path.join(output_folder, "data.yaml")
            data_yaml = {
                "train": os.path.abspath(train_folder).replace("\\", "/"),
                "val": os.path.abspath(val_folder).replace("\\", "/"),
                "nc": 1,
                "names": ["object"]
            }
            with open(data_yaml_path, "w") as f:
                yaml.dump(data_yaml, f)

            # Train YOLOv8
            model.train(
                data=os.path.abspath(data_yaml_path),
                epochs=1,
                imgsz=640,
                project=output_folder,
                name="train_result",
                exist_ok=True,
                plots=True,
                save=True
            )

            # Optional cleanup: remove unwanted files
            for f in ["results.csv", "args.yaml", "hyp.yaml"]:
                fp = os.path.join(output_folder, "train_result", f)
                if os.path.exists(fp):
                    os.remove(fp)

        print("[CLIENT] Training complete")
        return [], 1, {}

    def evaluate(self, parameters, config):
        print("[CLIENT] EVALUATE CALLED")
        return 0.0, 0, {}

if __name__ == "__main__":
    client = YOLOClient()
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)


