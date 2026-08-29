# 🚨 Fall Detection System (YOLOv8 + OpenCV)

A lightweight computer vision system that automatically detects when a person **falls** in a video — built for use cases like hospitals, elderly-care facilities, and home safety monitoring.

It uses **YOLOv8** (Ultralytics) to detect people in each video frame, then applies a simple geometric heuristic on the bounding box to flag a fall.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

---

## 🧠 How It Works

1. **Detect** — Every frame of the input video is run through the pre-trained `yolov8n.pt` model, which detects people (COCO class `0`).
2. **Analyze** — For each detected person, the script measures their bounding box:
   - `width = x2 - x1`
   - `height = y2 - y1`
3. **Classify**
   - `width > height` → person is likely **horizontal** → 🔴 **FALL DETECTED**
   - `height ≥ width` → person is **upright** → 🟢 **Normal**
4. **Visualize** — Bounding boxes and labels are drawn directly on the video frames.
5. **Export** — The annotated frames are saved to a new video file.

This is a fast, dependency-light heuristic — no pose-estimation model required — making it a great starting point before scaling up to something more robust.

---

## 📦 Requirements

- Python 3.8+
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- OpenCV

```bash
pip install ultralytics opencv-python
```

---

## ▶️ Usage

1. Place your video in the project folder and name it `input_video.mp4`
2. Run the script:

```bash
python fall_detection.py
```

3. The annotated result will be saved as `output_video.mp4` in the same folder.

The `yolov8n.pt` model weights (~6MB) will be downloaded automatically the first time you run the script.

---

## 🖼️ Output Example

| Status | Box Color | Label |
|--------|-----------|-------|
| Standing / sitting upright | 🟢 Green | `Normal` |
| Fallen / horizontal | 🔴 Red | `FALL DETECTED!` |

---

## ⚙️ Configuration

You can tweak these variables at the top of `fall_detection.py`:

```python
INPUT_VIDEO_PATH = "input_video.mp4"
OUTPUT_VIDEO_PATH = "output_video.mp4"
PERSON_CLASS_ID = 0
CONFIDENCE_THRESHOLD = 0.5
```

---

## ⚠️ Limitations & Next Steps

This project uses a **simple bounding-box heuristic**, which is fast and beginner-friendly but not production-grade. Known limitations:

- Can produce false positives for crouching, bending, sitting on the floor, or stretching
- Doesn't account for camera angle — a side-on camera works far better than a top-down one
- No temporal smoothing — a single misclassified frame counts as a "fall"

**Planned improvements:**
- [ ] Add temporal smoothing (require N consecutive "fall" frames before alerting)
- [ ] Switch to pose-estimation keypoints (e.g., MediaPipe or YOLO-Pose) for more accurate fall detection
- [ ] Add real-time webcam/RTSP stream support
- [ ] Add alerting (email/SMS/webhook) when a fall is confirmed
- [ ] Track individuals across frames to avoid duplicate/flickering alerts

---

## 📄 License

MIT — free to use, modify, and build on.

---

## 🙋 Disclaimer

This project is a proof-of-concept / educational tool. It is **not a certified medical device** and should not be relied upon as the sole safety measure in a real healthcare or elder-care environment.
