# R.A.P.T.O.R Video Processor Module
# File: src/video_processor.py

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
    print("⚠️ YOLO not installed. Run: pip install ultralytics")
    YOLO_AVAILABLE = False

class VideoTacticalProcessor:
    def __init__(self, model_path='yolov8n.pt', gps_bounds=None):
        """
        R.A.P.T.O.R Video Processing System
        
        Args:
            model_path: Path to YOLO model
            gps_bounds: GPS boundary coordinates for mapping
        """
        if not YOLO_AVAILABLE:
            raise ImportError("YOLO not available. Install with: pip install ultralytics")
            
        self.model = YOLO(model_path)
        self.gps_converter = SimpleGPSConverter(gps_bounds) if gps_bounds else None
        self.all_detections = []
        
        self.target_classes = {
            0: 'person',
            2: 'car', 
            3: 'motorcycle',
            5: 'bus',
            7: 'truck',
            14: 'bird',
            15: 'cat',
            16: 'dog'
        }
    
    def process_video(self, video_path, output_video=None, process_every_n_frames=5):
        """
        Process video and track objects
        
        Args:
            video_path: Input video file path
            output_video: Output video path (optional)
            process_every_n_frames: Process every nth frame for performance
            
        Returns:
            List of all detections found in video
        """
        if not os.path.exists(video_path):
            print(f"❌ Video file not found: {video_path}")
            return []
            
        print(f"🎬 Processing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Video specs: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup video writer if output requested
        if output_video:
            Path(output_video).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        
        frame_number = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every nth frame to speed up
                if frame_number % process_every_n_frames == 0:
                    detections = self.process_frame(frame, frame_number)
                    
                    # Draw detections on frame
                    annotated_frame = self.draw_detections(frame, detections)
                    
                    # Show progress
                    if frame_number % (process_every_n_frames * 30) == 0:
                        progress = (frame_number / total_frames) * 100
                        print(f"📊 Progress: {progress:.1f}% - Frame {frame_number}/{total_frames}: {len(detections)} objects")
                else:
                    annotated_frame = frame
                
                # Write to output video
                if output_video:
                    out.write(annotated_frame)
                
                # Optional: Display frame (comment out for faster processing)
                # cv2.imshow('R.A.P.T.O.R Processing', annotated_frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
                
                frame_number += 1
                
        except Exception as e:
            print(f"❌ Error during video processing: {e}")
        finally:
            cap.release()
            if output_video:
                out.release()
            cv2.destroyAllWindows()
        
        print(f"✅ Processed {frame_number} frames, found {len(self.all_detections)} total objects")
        return self.all_detections
    
    def process_frame(self, frame, frame_number):
        """Process single frame for object detection"""
        try:
            results = self.model(frame, verbose=False)
            frame_detections = []
            
            height, width = frame.shape[:2]
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls.item())
                        confidence = box.conf.item()
                        
                        if class_id in self.target_classes and confidence > 0.5:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            center_x = (x1 + x2) / 2
                            center_y = (y1 + y2) / 2
                            
                            detection = {
                                'id': len(self.all_detections) + len(frame_detections),
                                'frame': frame_number,
                                'class': self.target_classes[class_id],
                                'confidence': confidence,
                                'bbox': [x1, y1, x2, y2],
                                'center': [center_x, center_y],
                                'timestamp': datetime.now().isoformat()
                            }
                            
                            # Add GPS if available
                            if self.gps_converter:
                                lat, lon = self.gps_converter.pixel_to_gps(
                                    center_x, center_y, width, height
                                )
                                detection['gps'] = {'lat': lat, 'lon': lon}
                            
                            frame_detections.append(detection)
                            self.all_detections.append(detection)
            
            return frame_detections
            
        except Exception as e:
            print(f"❌ Frame {frame_number} processing error: {e}")
            return []
    
    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels on frame"""
        colors = {
            'person': (0, 255, 0),    # Green
            'car': (255, 0, 0),       # Blue  
            'truck': (0, 0, 255),     # Red
            'bus': (255, 255, 0),     # Cyan
            'motorcycle': (255, 0, 255), # Magenta
            'bird': (0, 255, 255),    # Yellow
            'cat': (128, 0, 128),     # Purple
            'dog': (255, 165, 0)      # Orange
        }
        
        for detection in detections:
            x1, y1, x2, y2 = [int(x) for x in detection['bbox']]
            color = colors.get(detection['class'], (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{detection['class']}: {detection['confidence']:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Background for text readability
            cv2.rectangle(frame, (x1, y1-30), (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Add R.A.P.T.O.R watermark and timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, f"R.A.P.T.O.R - {timestamp}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Detection count
        cv2.putText(frame, f"Objects: {len(detections)}", 
                   (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def save_detections(self, filename=None):
        """Save all detections to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'output/detections/video_detections_{timestamp}.json'
        
        # Ensure output directory exists
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.all_detections, f, indent=2)
            print(f"💾 Saved {len(self.all_detections)} detections to {filename}")
            return filename
        except Exception as e:
            print(f"❌ Failed to save detections: {e}")
            return None
    
    def create_detection_summary(self):
        """Create summary of video detections"""
        if not self.all_detections:
            return {}
        
        # Class counts
        class_counts = {}
        for detection in self.all_detections:
            obj_class = detection['class']
            class_counts[obj_class] = class_counts.get(obj_class, 0) + 1
        
        # Confidence stats
        confidences = [d['confidence'] for d in self.all_detections]
        avg_confidence = sum(confidences) / len(confidences)
        high_confidence = len([c for c in confidences if c > 0.8])
        
        # Frame coverage
        frames_with_detections = len(set(d['frame'] for d in self.all_detections))
        
        return {
            'total_detections': len(self.all_detections),
            'class_breakdown': class_counts,
            'average_confidence': avg_confidence,
            'high_confidence_detections': high_confidence,
            'frames_with_detections': frames_with_detections,
            'gps_enabled': self.gps_converter is not None
        }


# Example usage
if __name__ == "__main__":
    # Example GPS bounds for testing
    bounds = {
        'top_left': {'lat': 36.4074, 'lon': -105.5731},
        'top_right': {'lat': 36.4074, 'lon': -105.5700},
        'bottom_left': {'lat': 36.4044, 'lon': -105.5731},
        'bottom_right': {'lat': 36.4044, 'lon': -105.5700}
    }

    processor = VideoTacticalProcessor(gps_bounds=bounds)
    
    # Test with sample video (replace with actual video path)
    test_video = "sample_drone_video.mp4"
    if os.path.exists(test_video):
        detections = processor.process_video(test_video, "output_tactical.mp4")
        processor.save_detections()
        
        summary = processor.create_detection_summary()
        print("\n📊 Detection Summary:")
        for key, value in summary.items():
            print(f"   {key}: {value}")
    else:
        print(f"⚠️ Sample video not found: {test_video}")
        print("Place a video file in the project directory to test the processor")