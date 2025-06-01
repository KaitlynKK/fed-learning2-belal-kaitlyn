import os
import cv2
from ultralytics import YOLO
import flwr as fl

class YOLOClient(fl.client.NumPyClient):
    def __init__(self, video_paths, model_path, output_root):
        self.video_paths = video_paths
        self.model_path = model_path
        self.output_root = output_root
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

            self.extract_frames(video_path, input_folder)
            frames = os.listdir(input_folder)
            print(f"[CLIENT] Frames extracted: {frames}")

            for img in frames:
                frame_path = os.path.join(input_folder, img)
                result = model(frame_path)
                if result and result[0]:
                    result[0].save(filename=os.path.join(output_folder, img))
                    result_total += 1

            print(f"[CLIENT] Finished video {i+1}, detections: {result_total}")

        return [], result_total, {}

    def evaluate(self, parameters, config):
        return 0.0, 0, {}

    def extract_frames(self, video_path, output_folder, fps_interval=1):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[CLIENT] ❌ Cannot open video at: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        interval = int(fps * fps_interval) if fps > 0 else 30
        count, saved = 0, 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if count % interval == 0:
                cv2.imwrite(os.path.join(output_folder, f"frame_{saved:04d}.jpg"), frame)
                saved += 1
            count += 1
        cap.release()

def main():
    print("[CLIENT] main() called.")
    client = YOLOClient(
        video_paths=[
            "static/uploads/video1.mp4",
            "static/uploads/video2.mp4"
        ],
        model_path="yolov8n.pt",
        output_root="static/output"
    )
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)

if __name__ == "__main__":
    main()


