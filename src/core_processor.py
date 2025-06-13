# File: src/core_processor.py

import cv2
import json
import os
from datetime import datetime
from pathlib import Path
from gps_converter import SimpleGPSConverter

try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    print("⚠️ CRITICAL: YOLO not installed. Run: pip install ultralytics")
    YOLO_AVAILABLE = False


class RaptorProcessor:
    """
    R.A.P.T.O.R's single, authoritative processing engine for images and videos.
    Combines object detection, GPS conversion, and optional live GUI updates.
    """

    def __init__(self, model_path="yolov8n.pt", gps_bounds=None, gui_queue=None):
        if not YOLO_AVAILABLE:
            raise ImportError(
                "YOLO/Ultralytics is not installed. Cannot initialize RaptorProcessor."
            )

        self.model = YOLO(model_path)
        self.gps_converter = SimpleGPSConverter(gps_bounds) if gps_bounds else None
        self.gui_queue = gui_queue  # For sending live updates to a GUI
        self.all_detections = []

        # Consolidated from all previous processor classes
        self.target_classes = {
            0: "person",
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck",
            14: "bird",
            15: "cat",
            16: "dog",
        }
        self.colors = {
            "person": (0, 255, 0),
            "car": (255, 0, 0),
            "truck": (0, 0, 255),
            "bus": (255, 255, 0),
            "motorcycle": (255, 0, 255),
            "bird": (0, 255, 255),
            "cat": (128, 0, 128),
            "dog": (255, 165, 0),
        }

        print("✅ RaptorProcessor Initialized.")

    def stop(self):
        """Signals the processing loop to terminate."""
        print("🛑 Processor stop signal received.")
        self.processing_active = False

    def process_image(self, image_path, save_output=True):
        """
        Process a single image for tactical objects.
        (Logic migrated from main.py's RaptorDetectionSystem)
        """
        print(f"🔍 Processing Image: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Could not load image: {image_path}")
            return []

        results = self.model(img, verbose=False)
        frame_detections = self._parse_results(results, img.shape, frame_number=0)

        if save_output and frame_detections:
            annotated_frame = self.draw_detections(img, frame_detections)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"raptor_detection_{timestamp}.jpg"
            output_path = Path("output/images") / filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), annotated_frame)
            print(f"💾 Saved annotated image: {output_path}")

        print(f"✅ Found {len(frame_detections)} objects in image.")
        self.all_detections.extend(frame_detections)
        return frame_detections

    def process_video(self, video_path, output_path=None, process_every_n_frames=3):
        """
        Processes a video file or a live stream.
        - If video_path is a string, it treats it as a file.
        - If video_path is an integer, it treats it as a camera index (e.g., 0 for webcam).
        """
        is_live = isinstance(video_path, int)

        if is_live:
            print(f"🎬 Starting LIVE FEED from camera index: {video_path}")
        else:
            print(f"🎬 Processing VIDEO FILE: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error opening video source: {video_path}")
            return []

        # Get video properties (and handle live stream case)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_live else -1

        # Setup video writer (only for files)
        out = None
        if output_path and not is_live:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        self.processing_active = True
        frame_number = 0
        try:
            while self.processing_active:  # Loop is now controlled by this flag
                ret, frame = cap.read()
                if not ret:
                    if not is_live:
                        print("🏁 End of video file.")
                    break  # Exit loop if video file ends or camera disconnects

                annotated_frame = frame.copy()
                if frame_number % process_every_n_frames == 0:
                    results = self.model(frame, verbose=False)
                    detections = self._parse_results(results, frame.shape, frame_number)
                    annotated_frame = self.draw_detections(frame, detections)

                    # Live GUI Update Logic
                    if self.gui_queue:
                        self.gui_queue.put(
                            {
                                "type": "frame",
                                "frame": annotated_frame,
                                "detections": detections,
                                "frame_num": frame_number,
                            }
                        )
                        if (
                            not is_live
                            and frame_number % (process_every_n_frames * 10) == 0
                        ):
                            progress = (frame_number / total_frames) * 100
                            self.gui_queue.put(
                                {
                                    "type": "progress",
                                    "message": f"Processing: {progress:.1f}%",
                                }
                            )

                if out:
                    out.write(annotated_frame)

                frame_number += 1
        finally:
            self.processing_active = False  # Ensure flag is reset on exit
            cap.release()
            if out:
                out.release()
            print(
                f"✅ Video source closed. Found {len(self.all_detections)} total objects."
            )

        return self.all_detections

    def _parse_results(self, results, frame_shape, frame_number):
        """Helper function to parse detection results from the YOLO model."""
        frame_detections = []
        height, width, _ = frame_shape

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                class_id = int(box.cls.item())
                confidence = box.conf.item()

                if class_id in self.target_classes and confidence > 0.5:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2

                    detection = {
                        "id": f"{frame_number}-{len(self.all_detections) + len(frame_detections)}",
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

                    frame_detections.append(detection)

        self.all_detections.extend(frame_detections)
        return frame_detections

    def draw_detections(self, frame, detections):
        """Draws bounding boxes and labels on a frame."""
        for det in detections:
            x1, y1, x2, y2 = [int(c) for c in det["bbox"]]
            color = self.colors.get(det["class"], (255, 255, 255))
            label = f"{det['class']}: {det['confidence']:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

        # Watermark
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
        """Saves all collected detections to a JSON file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.all_detections, f, indent=2)
        print(f"💾 Saved {len(self.all_detections)} detections to {output_path}")
