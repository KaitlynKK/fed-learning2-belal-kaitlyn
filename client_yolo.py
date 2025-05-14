import os
import cv2
from ultralytics import YOLO
import flwr as fl

class YOLOClient(fl.client.NumPyClient):
    def __init__(self, video_path, model_path, output_folder):
        self.video_path = video_path
        self.model_path = model_path
        self.output_folder = output_folder
        self.input_folder = "frames"

        os.makedirs(self.input_folder, exist_ok=True)
        os.makedirs(self.output_folder, exist_ok=True)

    def get_parameters(self, config):
        return []

    def set_parameters(self, parameters):
        pass

    def fit(self, parameters, config):
        print("[CLIENT] Extracting frames...")
        self.extract_frames(self.video_path, self.input_folder)

        print("[CLIENT] Loading YOLO model...")
        model = YOLO(self.model_path)

        print("[CLIENT] Running detection...")
        for img in os.listdir(self.input_folder):
            frame_path = os.path.join(self.input_folder, img)
            result = model(frame_path)
            result[0].save(filename=os.path.join(self.output_folder, img))
            print(f"[CLIENT] Saved: {img}")

        print("[CLIENT] Detection complete.")

        return [], len(os.listdir(self.output_folder)), {}

    def evaluate(self, parameters, config):
        return 0.0, 0, {}

    def extract_frames(self, video_path, output_folder, fps_interval=1):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        interval = int(fps * fps_interval) if fps > 0 else 30
        count, saved = 0, 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if count % interval == 0:
                frame_filename = f"{output_folder}/frame_{saved:04d}.jpg"
                cv2.imwrite(frame_filename, frame)
                saved += 1
            count += 1
        cap.release()

def main():
    print("[CLIENT] main() called.")
    client = YOLOClient("static/uploads/video.mp4", "my_model.pt", "static/output")
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)

if __name__ == "__main__":
    main()
