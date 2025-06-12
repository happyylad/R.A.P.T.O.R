import sys
sys.path.append("src")
from video_processor import VideoTacticalProcessor
import json
from datetime import datetime

print("🦅 R.A.P.T.O.R Simple Processor")
print("=" * 40)

video_path = "C:/Users/calle/Downloads/videoplayback.mp4"
print(f"🎬 Processing: {video_path}")

bounds = {
    "top_left": {"lat": 36.4074, "lon": -105.5731},
    "top_right": {"lat": 36.4074, "lon": -105.5700}, 
    "bottom_left": {"lat": 36.4044, "lon": -105.5731},
    "bottom_right": {"lat": 36.4044, "lon": -105.5700}
}

processor = VideoTacticalProcessor(gps_bounds=bounds)
print("🔄 Starting mission...")
detections = processor.process_video(video_path, "output_mission.mp4", process_every_n_frames=5)
print(f"✅ Mission complete! Found {len(detections)} objects")
