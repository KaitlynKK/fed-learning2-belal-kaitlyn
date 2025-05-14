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
        print("[CLIENT] 🚀 FIT STARTED")
        print(f"[CLIENT] Video path: {self.video_path}")
        print(f"[CLIENT] Model path: {self.model_path}")
        print(f"[CLIENT] Output path: {self.output_folder}")

        self.extract_frames(self.video_path, self.input_folder)
        frames = os.listdir(self.input_folder)
        print(f"[CLIENT] 📸 Frames extracted: {frames}")

        if len(frames) == 0:
            print("[CLIENT] ❌ No frames extracted. Exiting early.")
            return [], 0, {}

        model = YOLO(self.model_path)
        print("[CLIENT] ✅ YOLO model loaded.")

        result_count = 0
        for img in frames:
            frame_path = os.path.join(self.input_folder, img)
            print(f"[CLIENT] 🔍 Running detection on: {frame_path}")
            result = model(frame_path)
            if result and result[0]:
                save_path = os.path.join(self.output_folder, img)
                result[0].save(filename=save_path)
                print(f"[CLIENT] 💾 Saved: {save_path}")
                result_count += 1
            else:
                print(f"[CLIENT] ⚠️ No result for: {img}")

        print(f"[CLIENT] ✅ Detection complete. {result_count} results saved.")
        return [], result_count, {}

    def evaluate(self, parameters, config):
        return 0.0, 0, {}

    def extract_frames(self, video_path, output_folder, fps_interval=1):
        print(f"[CLIENT] 🎞️ Reading from: {video_path}")
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"[CLIENT] ❌ Cannot open video at: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        interval = int(fps * fps_interval) if fps > 0 else 30
        print(f"[CLIENT] 🎯 FPS: {fps}, interval: {interval}")

        count, saved = 0, 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if count % interval == 0:
                frame_filename = os.path.join(output_folder, f"frame_{saved:04d}.jpg")
                cv2.imwrite(frame_filename, frame)
                print(f"[CLIENT] 🖼️ Saved frame: {frame_filename}")
                saved += 1
            count += 1
        cap.release()
        print(f"[CLIENT] 📦 Total frames saved: {saved}")

def main():
    print("[CLIENT] 🟢 main() called.")
    video_path = os.path.join("static", "uploads", "video.mp4")
    model_path = "my_model.pt"
    output_folder = os.path.join("static", "output")

    client = YOLOClient(video_path, model_path, output_folder)
    fl.client.start_client(server_address="localhost:8080", client=client.to_client())

if __name__ == "__main__":
    main()
