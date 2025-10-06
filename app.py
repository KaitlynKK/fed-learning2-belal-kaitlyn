from flask import Flask, request, render_template, redirect, url_for
import os
import threading
import subprocess
import base64
import time

UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/output"
TEST_RESULTS = os.path.join(OUTPUT_FOLDER, "test_results")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(TEST_RESULTS, exist_ok=True)

detection_results = []

def _gather_images(folder):
    b64s = []
    for root, _, files in os.walk(folder):
        for filename in files:
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                full_path = os.path.join(root, filename)
                with open(full_path, "rb") as img_file:
                    b64s.append(base64.b64encode(img_file.read()).decode("utf-8"))
    return b64s

@app.route("/", methods=["GET", "POST"])
def index():
    global detection_results
    return render_template("index.html", detections=detection_results)

@app.route("/start-training", methods=["POST"])
def start_training():
    # form controls are optional; if you don’t have a form yet, you can hard-code defaults below
    rounds = request.form.get("rounds", "4")
    clients = request.form.get("clients", "1")
    dataset = request.form.get("dataset", "data/labelfront1")
    data_distribution = request.form.get("data_distribution", "iid")

    # NEW: paths/args for the client
    labeled_dir = request.form.get("labeled_dir", dataset)
    work_root = request.form.get("work_root", "output/client1")
    split = request.form.get("split", "0.8")            # train/val split
    train_fraction = request.form.get("train_fraction", "0.2")  # % of TRAIN used to contribute to global model

    print(f"[INFO] Rounds={rounds}, Clients={clients}, Dataset={dataset}, Dist={data_distribution}")
    print(f"[INFO] Client args: labeled_dir={labeled_dir}, work_root={work_root}, split={split}, train_fraction={train_fraction}")

    # Start server
    threading.Thread(
        target=lambda: subprocess.run(
            ["python", "server_yolo.py", "--num_rounds", str(rounds)]
        ),
        daemon=True,
    ).start()

    # Start one client (replicate this block to start more clients with different work_root/labeled_dir)
    threading.Thread(
        target=lambda: subprocess.run(
            [
                "python", "client_yolo.py",
                "--server", "localhost:8080",
                "--model", "model/my_model.pt",
                "--labeled_dir", labeled_dir,
                "--work_root", work_root,
                "--epochs", "5",
                "--imgsz", "640",
                "--split", split,
                "--train_fraction", train_fraction,  # NEW
                "--nc", "1",
                "--names", "object",
            ]
        ),
        daemon=True,
    ).start()

    # Small wait so files start appearing; in production you might poll for completion signal instead
    time.sleep(5)

    # Show any images from output (train plots) and the post-training test results
    global detection_results
    detection_results = []
    detection_results += _gather_images(OUTPUT_FOLDER)
    detection_results += _gather_images(TEST_RESULTS)

    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
