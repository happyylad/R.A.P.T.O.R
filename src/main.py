# File: src/main.py
# This script serves as a command-line interface for the RaptorProcessor.

import os
from datetime import datetime
from pathlib import Path
import sys

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent))

from core_processor import RaptorProcessor


def main():
    """Main R.A.P.T.O.R command-line execution function."""
    print("🦅 R.A.P.T.O.R - Command-Line Interface")
    print("=" * 60)

    # --- Configuration ---
    # In the future, we will load this from a config file.
    gps_bounds = {
        "top_left": {"lat": 36.4074, "lon": -105.5731},
        "top_right": {"lat": 36.4074, "lon": -105.5700},
        "bottom_left": {"lat": 36.4044, "lon": -105.5731},
        "bottom_right": {"lat": 36.4044, "lon": -105.5700},
    }
    # Use a sample video if it exists
    INPUT_VIDEO = "sample_drone_video.mp4"

    if not os.path.exists(INPUT_VIDEO):
        print(f"❌ Input video not found: {INPUT_VIDEO}")
        print("💡 Run python setup_raptor.py to download sample data.")
        return

    # --- Initialization ---
    # Note: We pass gui_queue=None because we're not using a GUI here.
    processor = RaptorProcessor(gps_bounds=gps_bounds, gui_queue=None)

    # --- Execution ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video_path = f"output/videos/cli_mission_{timestamp}.mp4"
    output_json_path = f"output/detections/cli_mission_{timestamp}.json"

    # Process the video
    all_detections = processor.process_video(
        video_path=INPUT_VIDEO, output_path=output_video_path
    )

    # --- Reporting ---
    if all_detections:
        print(f"\n✅ Mission Complete. Processed video saved to: {output_video_path}")
        processor.save_detections_to_json(output_json_path)
    else:
        print("\n⚠️ Mission finished, but no objects were detected.")


if __name__ == "__main__":
    main()
