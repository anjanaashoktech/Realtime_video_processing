
"""
realtime_capture_processing.py

- Capture frames from webcam or RTSP stream in one thread
- Process frames in another thread (simulated inference)
- Use a thread-safe queue to pass frames
- Display FPS on the video window
- Skip near-duplicate frames (small downsample diff)
- Clean shutdown on 'q' (no memory leaks, proper thread termination)
"""

import cv2
import time
import argparse
import threading
import queue
from collections import deque

def is_duplicate(prev_small, frame, diff_threshold=2.0):
    """
    Returns (is_duplicate: bool, curr_small) where curr_small is the downsampled grayscale used for next comparison.
    diff_threshold: mean absolute difference threshold.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    curr_small = cv2.resize(gray, (64, 48))
    if prev_small is None:
        return False, curr_small
    diff = cv2.absdiff(prev_small, curr_small)
    mean_val = float(diff.mean())
    return mean_val < diff_threshold, curr_small


#Capture thread
def capture_thread_fn(source, frame_queue, stop_event, fps_cap=30):
    """Continuously read frames and push to queue. Drops frames when queue is full."""
    print("[CAPTURE] Starting capture:", source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("[CAPTURE] ERROR: Cannot open source:", source)
        stop_event.set()
        return

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret or frame is None:
                #short sleep on read failure to avoid busy loop
                time.sleep(0.01)
                continue

            #Put frame non-blocking; drop newest if full
            try:
                frame_queue.put(frame, block=False)
            except queue.Full:
                # queue full -> drop newest (do nothing)
                pass

            # Optionally cap capture read rate (useful for very high FPS cameras)
            # time.sleep(1.0 / fps_cap)
    finally:
        cap.release()
        print("[CAPTURE] Capture thread exiting, VideoCapture released.")


#Processing thread
def processing_thread_fn(frame_queue, stop_event, sim_processing_delay=0.03, fps_window=30):
    """
    Consume frames, simulate processing, display with FPS, skip duplicates.
    """
    print("[PROCESS] Starting processing thread.")
    timestamps = deque(maxlen=fps_window)
    prev_small = None
    window_name = "Realtime - Press 'q' to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while not stop_event.is_set():
            try:
                frame = frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            #sentinel handling
            if frame is None:
                stop_event.set()
                break

            #duplicate detection
            duplicate, curr_small = is_duplicate(prev_small, frame, diff_threshold=2.0)
            prev_small = curr_small
            if duplicate:
                #skip processing but still consider marking task done
                frame_queue.task_done()
                continue

            #simulate processing / inference
            time.sleep(sim_processing_delay)

            #FPS calculation
            now = time.time()
            timestamps.append(now)
            fps = 0.0
            if len(timestamps) >= 2:
                fps = (len(timestamps) - 1) / (timestamps[-1] - timestamps[0] + 1e-6)

            #Draw overlay
            display_frame = frame.copy()
            cv2.putText(display_frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Queue: {frame_queue.qsize()}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)

            cv2.imshow(window_name, display_frame)

            #key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                stop_event.set()
                break

            frame_queue.task_done()

    finally:
        cv2.destroyAllWindows()
        print("[PROCESS] Processing thread exiting and windows destroyed.")


def main():
    parser = argparse.ArgumentParser(description="Realtime capture + processing demo")
    parser.add_argument("--source", "-s", type=str, default="0",
                        help="Video source. Use 0 for webcam, or rtsp://... for RTSP stream")
    parser.add_argument("--max-queue-size", type=int, default=64, help="Max queue size")
    parser.add_argument("--sim-delay", type=float, default=0.03, help="Simulated processing delay (s)")
    args = parser.parse_args()

    #convert numeric webcam "0" to int
    try:
        source_val = int(args.source)
    except Exception:
        source_val = args.source

    frame_queue = queue.Queue(maxsize=args.max_queue_size)
    stop_event = threading.Event()

    cap_thread = threading.Thread(target=capture_thread_fn,
                                  args=(source_val, frame_queue, stop_event),
                                  name="CaptureThread", daemon=True)
    proc_thread = threading.Thread(target=processing_thread_fn,
                                   args=(frame_queue, stop_event, args.sim_delay, 30),
                                   name="ProcessingThread", daemon=True)

    cap_thread.start()
    proc_thread.start()

    try:
        while not stop_event.is_set():
            time.sleep(0.2)
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        # put sentinel to unblock
        try:
            frame_queue.put(None, block=False)
        except Exception:
            pass

        cap_thread.join(timeout=2.0)
        proc_thread.join(timeout=2.0)

        # clear queue for exit
        with frame_queue.mutex:
            frame_queue.queue.clear()

        print("Main thread exiting.")


if __name__ == "__main__":
    main()
