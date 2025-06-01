import os
import cv2
import pandas as pd
from ultralytics import YOLO

# ----------- COCO Class Names -----------

CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'bunny', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

train_class_id = CLASS_NAMES.index("train")
train_present = False
train_events = []

# ----------- Helper Functions -----------

def get_unique_filename(base_name, ext):
    counter = 1
    new_name = f"{base_name}_{counter}.{ext}"
    while os.path.exists(new_name):
        counter += 1
        new_name = f"{base_name}_{counter}.{ext}"
    return new_name

def extract_frames(video_path, output_folder, fps_interval=1):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * fps_interval)
    count, saved = 0, 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            cv2.imwrite(f"{output_folder}/frame_{saved:04d}.jpg", frame)
            saved += 1
        count += 1
    cap.release()

def frames_to_video(frame_folder, output_video, fps=10):
    images = sorted([img for img in os.listdir(frame_folder) if img.endswith(".jpg")])
    if not images:
        print("No frames found to convert.")
        return
    first_frame = cv2.imread(os.path.join(frame_folder, images[0]))
    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    for image in images:
        frame = cv2.imread(os.path.join(frame_folder, image))
        out.write(frame)
    out.release()
    print(f"[✓] Video saved as: {output_video}")

# ----------- Main Process -----------

video_path = r"lx01\videos\2024-10-07_074502_lx01_video.mp4"
input_folder = "frames"
output_folder = "inference_results"

excel_output = get_unique_filename("detection_results", "xlsx")
output_video = get_unique_filename("output_with_detections", "mp4")

os.makedirs(input_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

# Step 1: Extract frames
extract_frames(video_path, input_folder)

# Step 2: Load YOLO model
model = YOLO("yolov8n.pt")

# Step 3: Ground-truth labels
true_labels_dict = {
    # 'frame_0000.jpg': [0, 6]  # person, train
}

# Step 4: Inference + Annotation + Metric + Train tracking
detection_records = []

for img in sorted(os.listdir(input_folder)):
    full_path = os.path.join(input_folder, img)
    frame = cv2.imread(full_path)
    result = model(frame)
    detections = result[0].boxes
    pred_classes = [int(cls) for cls in detections.cls] if detections else []

    # Train detection logic
    train_detected = train_class_id in pred_classes

    if train_detected and not train_present:
        train_present = True
        train_events.append(("entered", img))
        print(f"[Train ENTERED] at {img}")
        cv2.putText(frame, "Train ENTERED", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    elif not train_detected and train_present:
        train_present = False
        train_events.append(("exited", img))
        print(f"[Train EXITED] at {img}")
        cv2.putText(frame, "Train EXITED", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # Metric Calculation
    true = true_labels_dict.get(img, [])
    pred = pred_classes

    if true or pred:
        true_set = set(true)
        pred_set = set(pred)

        tp = len(true_set & pred_set)
        fp_set = pred_set - true_set
        fn_set = true_set - pred_set

        fp = len(fp_set)
        fn = len(fn_set)

        acc = tp / len(true_set) if true_set else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

        fp_labels = [CLASS_NAMES[i] if i < len(CLASS_NAMES) else f'class_{i}' for i in fp_set]
        fn_labels = [CLASS_NAMES[i] if i < len(CLASS_NAMES) else f'class_{i}' for i in fn_set]
    else:
        acc = prec = f1 = 0
        fp_labels = []
        fn_labels = []

    # Annotate and record detections
    if detections:
        for box in detections:
            cls = int(box.cls[0])
            label = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f'class_{cls}'
            conf = float(box.conf[0])
            coords = box.xyxy[0].tolist()

            # Draw bounding box and label
            x1, y1, x2, y2 = map(int, coords)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            detection_records.append([img, label, conf, x1, y1, x2, y2, acc, prec, f1,
                                      ", ".join(fp_labels), ", ".join(fn_labels)])
    else:
        detection_records.append([img, None, None, None, None, None, None,
                                  acc, prec, f1, ", ".join(fp_labels), ", ".join(fn_labels)])

    # Save annotated frame
    cv2.imwrite(os.path.join(output_folder, img), frame)

# Step 5: Save Excel with metrics + train events
columns = ['Frame', 'Class', 'Confidence', 'x1', 'y1', 'x2', 'y2',
           'Accuracy', 'Precision', 'F1-Score', 'False Positive', 'False Negative']
df_detections = pd.DataFrame(detection_records, columns=columns)

with pd.ExcelWriter(excel_output) as writer:
    df_detections.to_excel(writer, sheet_name='Detections', index=False)
    if train_events:
        df_events = pd.DataFrame(train_events, columns=['Event', 'Frame'])
        df_events.to_excel(writer, sheet_name="Train event", index=False)

print(f"[✓] Excel file with detections and metrics saved: {excel_output}")

# Step 6: Save final annotated video
frames_to_video(output_folder, output_video, fps=10)

