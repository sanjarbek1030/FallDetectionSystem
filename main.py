"""
Fall Detection System using YOLOv8 + OpenCV
---------------------------------------------
This script reads a video file, detects people in every frame using a
pre-trained YOLOv8 model, and checks whether each person appears to have
"fallen" based on the shape of their bounding box.

Logic used (simple, beginner-friendly heuristic):
    - YOLO draws a rectangle (bounding box) around every person it detects.
    - When a person is STANDING, that box is usually taller than it is wide
      (height > width).
    - When a person has FALLEN, their body is horizontal, so the box
      becomes wider than it is tall (width > height).
    - So: if width > height  ->  we flag it as a FALL.

Requirements:
    pip install ultralytics opencv-python

Usage:
    Place a video named 'input_video.mp4' in the same folder as this script,
    then run:
        python fall_detection.py
    The processed video will be saved as 'output_video.mp4'.
"""

# ---------------------------------------------------------
# STEP 1: Import the libraries we need
# ---------------------------------------------------------
import cv2                     # OpenCV: for reading/writing video and drawing boxes
from ultralytics import YOLO   # Ultralytics: gives us the YOLOv8 model


# ---------------------------------------------------------
# STEP 2: Basic settings / file names
# ---------------------------------------------------------
INPUT_VIDEO_PATH = "input_video.mp4"    # video we want to analyze
OUTPUT_VIDEO_PATH = "output_video.mp4"  # video we will save results to
PERSON_CLASS_ID = 0                     # in the COCO dataset, class 0 = "person"
CONFIDENCE_THRESHOLD = 0.5              # ignore detections the model isn't confident about


# ---------------------------------------------------------
# STEP 3: Load the pre-trained YOLOv8 model
# ---------------------------------------------------------
# 'yolov8n.pt' is the smallest/fastest YOLOv8 model. It will be downloaded
# automatically the first time you run this script if it's not already
# present on your computer.
print("Loading YOLOv8 model...")
model = YOLO("yolov8n.pt")


# ---------------------------------------------------------
# STEP 4: Open the input video
# ---------------------------------------------------------
print(f"Opening video: {INPUT_VIDEO_PATH}")
cap = cv2.VideoCapture(INPUT_VIDEO_PATH)

if not cap.isOpened():
    raise IOError(f"Could not open video file: {INPUT_VIDEO_PATH}")

# Grab the video's original properties so our output video matches them
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video info -> FPS: {fps}, Resolution: {frame_width}x{frame_height}")


# ---------------------------------------------------------
# STEP 5: Set up the video writer (this will create our output video)
# ---------------------------------------------------------
# 'mp4v' is a codec that works well for saving .mp4 files
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))


# ---------------------------------------------------------
# STEP 6: Process the video frame by frame
# ---------------------------------------------------------
frame_count = 0
fall_frame_count = 0

print("Processing video... this may take a moment.")

while True:
    # Read one frame from the video. 'ret' is True if a frame was read
    # successfully, and 'frame' is the actual image (as a NumPy array).
    ret, frame = cap.read()

    if not ret:
        # No more frames left -> we've reached the end of the video
        break

    frame_count += 1

    # -----------------------------------------------------
    # STEP 6a: Run YOLOv8 on the current frame
    # -----------------------------------------------------
    # 'verbose=False' just stops YOLO from printing extra logs to the console
    results = model(frame, verbose=False)

    # YOLO can return multiple "results" objects (one per image passed in).
    # Since we only passed one frame, we only need the first result.
    detections = results[0].boxes

    frame_has_fall = False

    # -----------------------------------------------------
    # STEP 6b: Loop through every object YOLO detected in this frame
    # -----------------------------------------------------
    for box in detections:
        class_id = int(box.cls[0])       # what kind of object is this?
        confidence = float(box.conf[0])  # how confident is the model?

        # We only care about "person" detections above our confidence threshold
        if class_id == PERSON_CLASS_ID and confidence >= CONFIDENCE_THRESHOLD:

            # Get the bounding box coordinates: top-left (x1, y1) and
            # bottom-right (x2, y2) corners of the box, as plain integers.
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # -----------------------------------------------
            # STEP 6c: Calculate width and height of the box
            # -----------------------------------------------
            width = x2 - x1
            height = y2 - y1

            # -----------------------------------------------
            # STEP 6d: Apply the fall detection rule
            # -----------------------------------------------
            # If the box is wider than it is tall, the person is likely
            # lying down horizontally -> classify as a FALL.
            is_fall = width > height

            if is_fall:
                frame_has_fall = True

                # Draw a RED box (color format in OpenCV is BGR, not RGB)
                color = (0, 0, 255)       # red
                label = "FALL DETECTED!"
                thickness = 3             # thicker box to make it stand out

            else:
                # Draw a GREEN box for a normal, standing person
                color = (0, 255, 0)       # green
                label = "Normal"
                thickness = 2

            # Draw the rectangle around the person
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Draw the text label just above the box
            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),          # position text above the box
                cv2.FONT_HERSHEY_SIMPLEX,        # font style
                0.8,                              # font size
                color,                            # text color matches box color
                2,                                 # text thickness
                cv2.LINE_AA,
            )

    if frame_has_fall:
        fall_frame_count += 1

    # -----------------------------------------------------
    # STEP 6e: Write the processed frame to the output video
    # -----------------------------------------------------
    out.write(frame)

    # Optional: print progress every 30 frames so we know it's working
    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames...")


# ---------------------------------------------------------
# STEP 7: Clean up - release the video objects
# ---------------------------------------------------------
cap.release()
out.release()

print("-" * 50)
print("Done!")
print(f"Total frames processed: {frame_count}")
print(f"Frames where a fall was detected: {fall_frame_count}")
print(f"Output saved to: {OUTPUT_VIDEO_PATH}")
