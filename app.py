from flask import Flask, request, render_template, redirect, url_for
import os
import threading
import subprocess
import base64

UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/output"

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

detection_results = []

@app.route("/", methods=["GET", "POST"])
def index():
    global detection_results
    return render_template("index.html", detections=detection_results)

@app.route("/start-training", methods=["POST"])
def start_training():
    # Get parameters from form
    rounds = request.form.get("rounds")
    clients = request.form.get("clients")
    dataset = request.form.get("dataset")
    data_distribution = request.form.get("data_distribution")

    print(f"[INFO] Starting training: Rounds={rounds}, Clients={clients}, Dataset={dataset}, Dist={data_distribution}")

    # You can pass these parameters to your training script via environment variables or CLI args
    threading.Thread(target=lambda: subprocess.run(["python", "server_yolo.py"])).start()
    threading.Thread(target=lambda: subprocess.run(["python", "client_yolo.py"])).start()

    return "Training started! You can check back later for results.", 200

@app.route("/detect")
def detect():
    global detection_results

    detection_results.clear()
    for filename in os.listdir(OUTPUT_FOLDER):
        if filename.endswith(".jpg"):
            with open(os.path.join(OUTPUT_FOLDER, filename), "rb") as img_file:
                b64 = base64.b64encode(img_file.read()).decode("utf-8")
                detection_results.append(b64)

    return redirect(url_for("index"))

@app.route("/result")
def result_page():
    return "Result page coming soon."  # or render_template("result.html")

if __name__ == "__main__":
    app.run(debug=True)
