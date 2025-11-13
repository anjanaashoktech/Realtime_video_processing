# Realtime Capture + Processing Demo

A simple Python demo showing how to capture frames from a webcam or RTSP stream in one thread and process them in another using a thread-safe queue. FPS is measured and displayed on the video window. The processing thread simulates inference with a small delay.

## Files
- `realtime_capture_processing.py` — main script
- `requirements.txt` — Python dependencies (see below)

## Requirements
- Python 3.8+
- OpenCV
- (optional) NumPy (OpenCV bundles enough)

Install dependencies:
```
pip install -r requirements.txt
```

## Usage
Run with default webcam:
```
 python realtime_capture_processing.py --source 0
```
Run with RTSP stream:
```
python realtime_capture_processing.py --source "rtsp://username:password@camera-ip:554/stream"
```
**Optional arguments:**
```
--max-queue-size: max frames to buffer between capture and processing (default: 64)
--sim-delay: simulated processing delay in seconds (default: 0.03)
```


