# Created by Andrew Yang
#this is the video output version of detect_trains.py
#the input folder and output folder has to exist before running the script
# This script extracts frames from a video, runs YOLOv8 inference on each frame, and saves the results.
# It requires the ultralytics package and OpenCV for video processing.
# Make sure to install the required packages using pip:
# pip install ultralytics opencv-python
# Usage: python detect_trains_vid.py

# Import necessary libraries
import os
import cv2
from ultralytics import YOLO

# Function to extract frames from a video
def extract_frames(video_path, output_folder, fps_interval=1):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get the frames per second (FPS) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate the interval between frames to save
    frame_interval = int(fps * fps_interval)
    
    # Initialize counters for frame processing and saving
    count = 0
    saved = 0
    
    # Loop through the video frames
    while cap.isOpened():
        # Read the next frame
        ret, frame = cap.read()
        
        # Break the loop if no more frames are available
        if not ret:
            break
        
        # Save the frame if it matches the interval
        if count % frame_interval == 0:
            cv2.imwrite(f"{output_folder}/frame_{saved:04d}.jpg", frame)
            saved += 1
        
        # Increment the frame counter
        count += 1
    
    # Release the video capture object
    cap.release()

def frames_to_video(frame_folder, output_video, fps=10):
    images = sorted([img for img in os.listdir(frame_folder) if img.endswith(".jpg")])
    if not images:
        print("No frames found to convert.")
        return

    # Read first image to get frame size
    first_frame = cv2.imread(os.path.join(frame_folder, images[0]))
    height, width, _ = first_frame.shape

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Use "XVID" for AVI
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for image in images:
        frame = cv2.imread(os.path.join(frame_folder, image))
        out.write(frame)

    out.release()
    print(f"[✓] Video saved as: {output_video}")

video_path = "video2.mp4" # Replace with your video file path

# Define input and output folders
#has to be created in the same directory as the script
input_folder = "frames"
output_folder = "inference_results"

os.makedirs(input_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

# Step 1: Extract frames
extract_frames(video_path, input_folder)

# Step 2: Load YOLOv8 model
model = YOLO("my_model.pt")  # Replace with custom model if trained

# Step 3: Run inference on each frame
for img in os.listdir(input_folder):
    result = model(f"{input_folder}/{img}")
    result[0].save(filename=f"{output_folder}/{img}")

frames_to_video("inference_results", "output_with_detections.mp4", fps=10)

