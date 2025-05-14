from flask import Flask, request, render_template, redirect, url_for
import os
import threading
import flwr as fl
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

    if request.method == "POST":
        if "video" not in request.files:
            return redirect(request.url)
        file = request.files["video"]
        if file.filename == "":
            return redirect(request.url)
        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], "video.mp4")
            file.save(filepath)
            return redirect(url_for("detect"))

    return render_template("index.html", detections=detection_results)

@app.route("/detect")
def detect():
    global detection_results

    threading.Thread(target=lambda: subprocess.run(["python", "server_yolo.py"])).start()
    threading.Thread(target=lambda: subprocess.run(["python", "client_yolo.py"])).start()

    import time
    time.sleep(10)

    detection_results.clear()
    for filename in os.listdir(OUTPUT_FOLDER):
        if filename.endswith(".jpg"):
            with open(os.path.join(OUTPUT_FOLDER, filename), "rb") as img_file:
                b64 = base64.b64encode(img_file.read()).decode("utf-8")
                detection_results.append(b64)

    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)