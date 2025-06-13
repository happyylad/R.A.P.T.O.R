# File: src/core_processor.py (V5 - Final with Dual Model Support)

import cv2
import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
import numpy as np  # Needed for random colors
from .gps_converter import SimpleGPSConverter
from . import config

try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    print("⚠️ CRITICAL: YOLO not installed. Run: pip install ultralytics")
    YOLO_AVAILABLE = False


class VideoStream:
    """
    A dedicated class to handle video stream reading in a separate thread.
    This prevents buffer overload and ensures we always get the latest frame.
    """

    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            raise IOError(f"Cannot open camera source: {src}")
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()


class RaptorProcessor:
    """
    R.A.P.T.O.R's processing engine, with support for multiple, selectable models.
    """

    def __init__(self, model_path, gps_bounds=None, gui_queue=None):
        if not YOLO_AVAILABLE:
            raise ImportError("YOLO/Ultralytics is not installed.")

        print(f"🧠 Loading R.A.P.T.O.R. model: {model_path}")
        self.model = YOLO(model_path)
        self.gps_converter = SimpleGPSConverter(gps_bounds) if gps_bounds else None
        self.gui_queue = gui_queue
        self.all_detections = []
        self.processing_active = False
        self.video_stream = None
        self.last_known_detections = []

        # --- DYNAMIC CLASS, COLOR, AND CONFIDENCE LOADING ---
        # This block intelligently configures the processor based on the loaded model.
        if "raptor_v1" in model_path:
            # We are using our powerful custom aerial model
            print("   Mode: Custom Aerial Model (R.A.P.T.O.R. v1)")
            self.target_classes = {
                i: name for i, name in enumerate(config.RAPTOR_MODEL_CLASSES)
            }
            self.colors = config.RAPTOR_CLASS_COLORS
            # Use a higher confidence for the specialized model to reduce noise
            self.confidence_threshold = 0.5
        else:
            # We are using the generic ground model (or any other)
            print("   Mode: Default Ground Model (COCO classes)")
            # Use the model's own built-in class names
            self.target_classes = self.model.names
            # Generate a unique random color for each COCO class
            self.colors = {
                name: [int(c) for c in np.random.randint(0, 255, size=3)]
                for name in self.model.names.values()
            }
            # Use a slightly higher confidence for the generic model
            self.confidence_threshold = 0.6

        print(
            f"✅ RaptorProcessor Initialized for {len(self.target_classes)} classes with conf: {self.confidence_threshold}"
        )

    def stop(self):
        """Signals the processing loop to terminate."""
        print("🛑 Processor stop signal received.")
        self.processing_active = False
        if self.video_stream:
            self.video_stream.stop()

    def process_video(self, video_path, output_path=None):
        """Main processing method. Dispatches to file or live processing."""
        self.all_detections = []
        self.last_known_detections = []  # Reset for the new mission
        is_live = isinstance(video_path, int)

        if is_live:
            self._process_live_stream(video_path)
        else:
            self._process_video_file(video_path, output_path)

        return self.all_detections

    def _process_live_stream(self, camera_index):
        """
        Processes a live camera stream with performance optimizations.
        """
        print(f"🎬 Starting LIVE FEED from camera index: {camera_index}")
        try:
            self.video_stream = VideoStream(src=camera_index).start()
        except IOError as e:
            print(f"❌ FATAL: {e}")
            if self.gui_queue:
                self.gui_queue.put({"type": "error", "message": str(e)})
            return

        time.sleep(2.0)

        self.processing_active = True
        frame_number = 0
        PROCESS_EVERY_N_FRAMES = 5

        try:
            while self.processing_active:
                frame = self.video_stream.read()
                if frame is None:
                    time.sleep(0.5)
                    continue

                if frame_number % PROCESS_EVERY_N_FRAMES == 0:
                    results = self.model(
                        frame, verbose=False, conf=self.confidence_threshold
                    )
                    self.last_known_detections = self._parse_results(
                        results, frame.shape, frame_number
                    )

                annotated_frame = self.draw_detections(
                    frame, self.last_known_detections
                )

                if self.gui_queue:
                    self.gui_queue.put(
                        {
                            "type": "frame",
                            "frame": annotated_frame,
                            "detections": self.last_known_detections,
                            "frame_num": frame_number,
                        }
                    )

                time.sleep(0.01)
                frame_number += 1
        finally:
            self.stop()
            print(
                f"✅ Live feed closed. Found {len(self.all_detections)} total objects."
            )

    def _process_video_file(
        self, video_path, output_path=None, process_every_n_frames=3
    ):
        print(f"🎬 Processing VIDEO FILE: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error opening video file: {video_path}")
            return

        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )
        fps, total_frames = int(cap.get(cv2.CAP_PROP_FPS)), int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        out = None
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        self.processing_active = True
        frame_number = 0
        try:
            while self.processing_active:
                ret, frame = cap.read()
                if not ret:
                    break

                annotated_frame = frame.copy()
                if frame_number % process_every_n_frames == 0:
                    results = self.model(
                        frame, verbose=False, conf=self.confidence_threshold
                    )
                    self.last_known_detections = self._parse_results(
                        results, frame.shape, frame_number
                    )
                    if self.gui_queue and total_frames > 0:
                        progress = (frame_number / total_frames) * 100
                        self.gui_queue.put(
                            {
                                "type": "progress",
                                "message": f"Processing: {progress:.1f}%",
                            }
                        )

                annotated_frame = self.draw_detections(
                    frame, self.last_known_detections
                )

                if self.gui_queue:
                    self.gui_queue.put(
                        {
                            "type": "frame",
                            "frame": annotated_frame,
                            "detections": self.last_known_detections,
                            "frame_num": frame_number,
                        }
                    )

                if out:
                    out.write(annotated_frame)
                frame_number += 1
        finally:
            self.processing_active = False
            cap.release()
            if out:
                out.release()
            print(
                f"✅ Video file processing complete. Found {len(self.all_detections)} total objects."
            )

    def _parse_results(self, results, frame_shape, frame_number):
        frame_detections = []
        height, width, _ = frame_shape
        current_batch_detections = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                class_id = int(box.cls.item())
                confidence = box.conf.item()

                if class_id in self.target_classes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                    detection = {
                        "id": f"{frame_number}-{len(self.all_detections) + len(current_batch_detections)}",
                        "frame": frame_number,
                        "class": self.target_classes[class_id],
                        "confidence": confidence,
                        "bbox": [x1, y1, x2, y2],
                        "center_pixel": {"x": center_x, "y": center_y},
                        "timestamp": datetime.now().isoformat(),
                    }
                    if self.gps_converter:
                        lat, lon = self.gps_converter.pixel_to_gps(
                            center_x, center_y, width, height
                        )
                        detection["gps"] = {"lat": lat, "lon": lon}
                    current_batch_detections.append(detection)

        self.all_detections.extend(current_batch_detections)
        return current_batch_detections

    def draw_detections(self, frame, detections):
        MAX_BOXES_TO_DRAW = 50
        top_detections = sorted(
            detections, key=lambda d: d["confidence"], reverse=True
        )[:MAX_BOXES_TO_DRAW]

        for det in top_detections:
            x1, y1, x2, y2 = [int(c) for c in det["bbox"]]
            color = self.colors.get(
                det["class"], (255, 255, 255)
            )  # Default to white if class color not found
            label = f"{det['class']}: {det['confidence']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

        if len(detections) > MAX_BOXES_TO_DRAW:
            count_text = f"Detections: {len(detections)} (Top {MAX_BOXES_TO_DRAW})"
            cv2.putText(
                frame, count_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4
            )
            cv2.putText(
                frame,
                count_text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

        cv2.putText(
            frame,
            "🦅 R.A.P.T.O.R LIVE",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 65),
            2,
        )
        return frame

    def save_detections_to_json(self, output_path):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.all_detections, f, indent=2)
        print(f"💾 Saved {len(self.all_detections)} detections to {output_path}")
