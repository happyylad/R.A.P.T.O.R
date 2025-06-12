# R.A.P.T.O.R - Real-time Aerial Patrol and Tactical Object Recognition
# File: src/main.py

import os
import sys
import json
import time
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("⚠️ YOLO not installed. Run: pip install ultralytics")
    YOLO_AVAILABLE = False

class RaptorDetectionSystem:
    """
    R.A.P.T.O.R - Real-time Aerial Patrol and Tactical Object Recognition
    Advanced object detection system for tactical surveillance
    """
    
    def __init__(self, model_path='yolov8n.pt', gps_bounds=None):
        self.model_path = model_path
        self.gps_bounds = gps_bounds
        self.detections = []
        self.system_status = "INITIALIZING"
        
        # Target object classes for tactical detection
        self.target_classes = {
            0: 'person',
            2: 'car',
            3: 'motorcycle', 
            5: 'bus',
            7: 'truck',
            14: 'bird',  # For wildlife monitoring
            15: 'cat',
            16: 'dog'
        }
        
        print("🦅 R.A.P.T.O.R System Initializing...")
        self.initialize_system()
    
    def initialize_system(self):
        """Initialize the RAPTOR detection system"""
        try:
            if not YOLO_AVAILABLE:
                self.system_status = "ERROR - YOLO not available"
                print("❌ YOLO not available")
                return False
            
            print("📥 Loading YOLO model...")
            self.model = YOLO(self.model_path)
            
            # Create output directories
            self.create_directories()
            
            self.system_status = "READY"
            print("✅ R.A.P.T.O.R System Ready for Operation")
            return True
            
        except Exception as e:
            self.system_status = f"ERROR - {str(e)}"
            print(f"❌ System initialization failed: {e}")
            return False
    
    def create_directories(self):
        """Create necessary output directories"""
        directories = [
            'output',
            'output/detections',
            'output/images',
            'output/videos',
            'output/maps'
        ]
        
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def process_image(self, image_path, save_output=True):
        """
        Process a single image for tactical objects
        
        Args:
            image_path: Path to input image
            save_output: Whether to save annotated output
            
        Returns:
            List of detection objects
        """
        if self.system_status != "READY":
            print(f"❌ System not ready: {self.system_status}")
            return []
        
        print(f"🔍 Processing: {image_path}")
        
        try:
            # Run detection
            results = self.model(image_path)
            detections = []
            
            # Get image dimensions
            img = cv2.imread(image_path)
            if img is None:
                print(f"❌ Could not load image: {image_path}")
                return []
            
            height, width = img.shape[:2]
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls.item())
                        confidence = box.conf.item()
                        
                        # Only process target classes with sufficient confidence
                        if class_id in self.target_classes and confidence > 0.5:
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            center_x = (x1 + x2) / 2
                            center_y = (y1 + y2) / 2
                            
                            detection = {
                                'id': len(self.detections) + len(detections),
                                'class': self.target_classes[class_id],
                                'confidence': confidence,
                                'bbox': [x1, y1, x2, y2],
                                'center': [center_x, center_y],
                                'timestamp': datetime.now().isoformat(),
                                'source_image': image_path
                            }
                            
                            # Add GPS if converter available
                            if self.gps_bounds:
                                lat, lon = self.convert_pixel_to_gps(
                                    center_x, center_y, width, height
                                )
                                detection['gps'] = {'lat': lat, 'lon': lon}
                            
                            detections.append(detection)
                            self.detections.append(detection)
            
            # Save annotated image if requested
            if save_output and detections:
                self.save_annotated_image(results[0], image_path)
            
            print(f"✅ Found {len(detections)} tactical objects")
            return detections
            
        except Exception as e:
            print(f"❌ Processing failed: {e}")
            return []
    
    def convert_pixel_to_gps(self, pixel_x, pixel_y, image_width, image_height):
        """Convert pixel coordinates to GPS coordinates"""
        if not self.gps_bounds:
            return None, None
        
        # Normalize pixel coordinates to 0-1
        norm_x = pixel_x / image_width
        norm_y = pixel_y / image_height
        
        # Simple bilinear interpolation
        lat = self.interpolate_lat(norm_x, norm_y)
        lon = self.interpolate_lon(norm_x, norm_y)
        
        return lat, lon
    
    def interpolate_lat(self, norm_x, norm_y):
        """Interpolate latitude from normalized coordinates"""
        top_lat = self.linear_interp(
            self.gps_bounds['top_left']['lat'],
            self.gps_bounds['top_right']['lat'],
            norm_x
        )
        bottom_lat = self.linear_interp(
            self.gps_bounds['bottom_left']['lat'],
            self.gps_bounds['bottom_right']['lat'],
            norm_x
        )
        return self.linear_interp(top_lat, bottom_lat, norm_y)
    
    def interpolate_lon(self, norm_x, norm_y):
        """Interpolate longitude from normalized coordinates"""
        left_lon = self.linear_interp(
            self.gps_bounds['top_left']['lon'],
            self.gps_bounds['bottom_left']['lon'],
            norm_y
        )
        right_lon = self.linear_interp(
            self.gps_bounds['top_right']['lon'],
            self.gps_bounds['bottom_right']['lon'],
            norm_y
        )
        return self.linear_interp(left_lon, right_lon, norm_x)
    
    def linear_interp(self, val1, val2, t):
        """Linear interpolation between two values"""
        return val1 + (val2 - val1) * t
    
    def save_annotated_image(self, result, original_path):
        """Save image with detection annotations"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"raptor_detection_{timestamp}.jpg"
        output_path = os.path.join('output', 'images', filename)
        
        annotated = result.plot()
        cv2.imwrite(output_path, annotated)
        print(f"💾 Saved annotated image: {output_path}")
    
    def process_video(self, video_path, output_video=None, process_every_n_frames=3):
        """
        Process video for tactical object detection
        
        Args:
            video_path: Path to input video
            output_video: Path for output video (optional)
            process_every_n_frames: Process every nth frame for performance
            
        Returns:
            List of all detections found
        """
        if self.system_status != "READY":
            print(f"❌ System not ready: {self.system_status}")
            return []
        
        print(f"🎬 Processing video: {video_path}")
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"📹 Video specs: {width}x{height}, {fps} FPS, {total_frames} frames")
            
            # Setup video writer if output requested
            if output_video:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
            
            frame_number = 0
            video_detections = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every nth frame
                if frame_number % process_every_n_frames == 0:
                    detections = self.process_frame(frame, frame_number)
                    video_detections.extend(detections)
                    
                    # Draw detections on frame
                    annotated_frame = self.draw_detections(frame, detections)
                    
                    # Progress update
                    if frame_number % (process_every_n_frames * 30) == 0:
                        progress = (frame_number / total_frames) * 100
                        print(f"📊 Progress: {progress:.1f}% - Frame {frame_number}/{total_frames}")
                else:
                    annotated_frame = frame
                
                # Write to output video
                if output_video:
                    out.write(annotated_frame)
                
                frame_number += 1
            
            cap.release()
            if output_video:
                out.release()
            
            print(f"✅ Video processing complete: {len(video_detections)} detections")
            return video_detections
            
        except Exception as e:
            print(f"❌ Video processing failed: {e}")
            return []
    
    def process_frame(self, frame, frame_number):
        """Process single video frame"""
        try:
            results = self.model(frame)
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
                                'id': len(self.detections) + len(frame_detections),
                                'frame': frame_number,
                                'class': self.target_classes[class_id],
                                'confidence': confidence,
                                'bbox': [x1, y1, x2, y2],
                                'center': [center_x, center_y],
                                'timestamp': datetime.now().isoformat()
                            }
                            
                            # Add GPS if available
                            if self.gps_bounds:
                                lat, lon = self.convert_pixel_to_gps(
                                    center_x, center_y, width, height
                                )
                                detection['gps'] = {'lat': lat, 'lon': lon}
                            
                            frame_detections.append(detection)
                            self.detections.append(detection)
            
            return frame_detections
            
        except Exception as e:
            print(f"❌ Frame processing error: {e}")
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
            
            # Draw label with confidence
            label = f"{detection['class']}: {detection['confidence']:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Background for text
            cv2.rectangle(frame, (x1, y1-30), (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Add R.A.P.T.O.R watermark
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, f"R.A.P.T.O.R - {timestamp}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def save_detections(self, filename=None):
        """Save detection results to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'output/detections/raptor_detections_{timestamp}.json'
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.detections, f, indent=2)
            print(f"💾 Saved {len(self.detections)} detections to {filename}")
            return filename
        except Exception as e:
            print(f"❌ Failed to save detections: {e}")
            return None
    
    def generate_report(self):
        """Generate tactical assessment report"""
        if not self.detections:
            print("⚠️ No detections to report")
            return
        
        print("\n" + "="*50)
        print("🦅 R.A.P.T.O.R TACTICAL REPORT")
        print("="*50)
        
        total_detections = len(self.detections)
        print(f"📊 Total Objects Detected: {total_detections}")
        
        # Class breakdown
        class_counts = {}
        for detection in self.detections:
            obj_class = detection['class']
            class_counts[obj_class] = class_counts.get(obj_class, 0) + 1
        
        print("\n📋 Detection Breakdown:")
        for obj_class, count in sorted(class_counts.items()):
            percentage = (count / total_detections) * 100
            print(f"   {obj_class.title()}: {count} ({percentage:.1f}%)")
        
        # Confidence analysis
        confidences = [d['confidence'] for d in self.detections]
        avg_confidence = sum(confidences) / len(confidences)
        high_confidence = len([c for c in confidences if c > 0.8])
        
        print(f"\n🎯 Confidence Analysis:")
        print(f"   Average Confidence: {avg_confidence:.1%}")
        print(f"   High Confidence (>80%): {high_confidence}/{total_detections}")
        
        # GPS coverage
        with_gps = len([d for d in self.detections if 'gps' in d])
        if with_gps > 0:
            print(f"\n🗺️ GPS Coverage: {with_gps}/{total_detections} ({with_gps/total_detections:.1%})")
        
        print("="*50)
    
    def get_system_status(self):
        """Get current system status"""
        return {
            'status': self.system_status,
            'total_detections': len(self.detections),
            'yolo_available': YOLO_AVAILABLE,
            'gps_enabled': self.gps_bounds is not None
        }


def main():
    """Main R.A.P.T.O.R execution function"""
    print("🦅 R.A.P.T.O.R - Real-time Aerial Patrol and Tactical Object Recognition")
    print("=" * 60)
    
    # Example GPS bounds (adjust for your area)
    gps_bounds = {
        'top_left': {'lat': 36.4074, 'lon': -105.5731},
        'top_right': {'lat': 36.4074, 'lon': -105.5700},
        'bottom_left': {'lat': 36.4044, 'lon': -105.5731},
        'bottom_right': {'lat': 36.4044, 'lon': -105.5700}
    }
    
    # Initialize R.A.P.T.O.R system
    raptor = RaptorDetectionSystem(gps_bounds=gps_bounds)
    
    # Example usage
    if raptor.system_status == "READY":
        print("\n🎯 R.A.P.T.O.R System Online - Ready for Tactical Operations")
        
        # Test with sample image (if available)
        test_image = "test_image.jpg"
        if os.path.exists(test_image):
            detections = raptor.process_image(test_image)
            raptor.generate_report()
            raptor.save_detections()
        else:
            print("💡 Place test images or videos in the project directory to begin detection")
    
    return raptor


if __name__ == "__main__":
    system = main()