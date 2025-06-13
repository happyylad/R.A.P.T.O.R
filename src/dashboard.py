# R.A.P.T.O.R Dashboard Module - Video Enhanced Version
# File: src/dashboard.py

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import cv2
import json
import webbrowser
import os
import numpy as np
from PIL import Image, ImageTk
from datetime import datetime
from pathlib import Path
import sys
import queue
import time

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Force enable all modules since we know they work
VIDEO_PROCESSOR_AVAILABLE = True

from core_processor import RaptorProcessor
from qgis_mapper import TacticalQGISMapper
from performance_analyzer import PerformanceAnalyzer
from test_suite import RaptorTestSuite


class RaptorDashboard:
    def __init__(self):
        """R.A.P.T.O.R Tactical Dashboard Interface with Live Video"""
        self.root = tk.Tk()
        self.root.title("🦅 R.A.P.T.O.R - Tactical Object Detection Dashboard")
        self.root.geometry("1400x900")
        self.root.configure(bg="#1a1a1a")

        # Set window icon and theme
        self.setup_theme()

        # System state
        self.processor = None
        self.mapper = None
        self.current_detections = []
        self.is_processing = False
        self.current_video_frame = None
        self.video_playback_active = False

        # Video display queue for thread-safe updates
        self.video_queue = queue.Queue()
        self.processed_video_path = None

        # Setup UI components
        self.setup_ui()

        # Initialize output directories
        self.create_output_directories()

        # Start video display update loop
        self.update_video_display()

    def setup_theme(self):
        """Setup dark theme for tactical look"""
        style = ttk.Style()
        style.theme_use("clam")

        # Configure colors for tactical appearance
        style.configure(
            "Title.TLabel",
            background="#1a1a1a",
            foreground="#00ff41",
            font=("Arial", 16, "bold"),
        )

        style.configure(
            "Header.TLabel",
            background="#1a1a1a",
            foreground="#00ff41",
            font=("Arial", 12, "bold"),
        )

        style.configure(
            "Status.TLabel",
            background="#1a1a1a",
            foreground="#ffffff",
            font=("Courier", 10),
        )

        style.configure(
            "Tactical.TButton",
            background="#00ff41",
            foreground="#000000",
            font=("Arial", 10, "bold"),
        )

    def create_output_directories(self):
        """Create necessary output directories"""
        directories = [
            "output/detections",
            "output/images",
            "output/videos",
            "output/maps",
            "output/analysis",
            "output/testing",
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def setup_ui(self):
        """Setup the main user interface"""
        # Main container
        main_container = tk.Frame(self.root, bg="#1a1a1a")
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Title header
        title_frame = tk.Frame(main_container, bg="#1a1a1a")
        title_frame.pack(fill=tk.X, pady=(0, 20))

        title_label = ttk.Label(
            title_frame,
            text="🦅 R.A.P.T.O.R TACTICAL COMMAND CENTER",
            style="Title.TLabel",
        )
        title_label.pack()

        subtitle_label = ttk.Label(
            title_frame,
            text="Real-time Aerial Patrol and Tactical Object Recognition",
            style="Status.TLabel",
        )
        subtitle_label.pack()

        # Control Panel
        self.setup_control_panel(main_container)

        # Main content area
        content_frame = tk.Frame(main_container, bg="#1a1a1a")
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Left panel - Video/Image display
        self.setup_display_panel(content_frame)

        # Right panel - Statistics and controls
        self.setup_stats_panel(content_frame)

        # Bottom panel - Action buttons
        self.setup_action_panel(main_container)

        # Status bar
        self.setup_status_bar(main_container)

    def setup_control_panel(self, parent):
        """Setup the control panel"""
        control_frame = tk.LabelFrame(
            parent,
            text="📡 MISSION CONTROL",
            bg="#2a2a2a",
            fg="#00ff41",
            font=("Arial", 12, "bold"),
            padx=10,
            pady=10,
        )
        control_frame.pack(fill=tk.X, pady=(0, 10))

        # GPS Configuration
        gps_frame = tk.LabelFrame(
            control_frame,
            text="GPS BOUNDS",
            bg="#2a2a2a",
            fg="#00ff41",
            font=("Arial", 10, "bold"),
        )
        gps_frame.pack(fill=tk.X, pady=(0, 10))

        # GPS coordinate inputs
        coord_frame = tk.Frame(gps_frame, bg="#2a2a2a")
        coord_frame.pack(fill=tk.X, padx=5, pady=5)

        # Top-left coordinates
        tk.Label(
            coord_frame,
            text="TOP-LEFT:",
            bg="#2a2a2a",
            fg="#ffffff",
            font=("Courier", 9, "bold"),
        ).grid(row=0, column=0, sticky=tk.W, padx=5)

        self.tl_lat = tk.StringVar(value="36.4074")
        self.tl_lon = tk.StringVar(value="-105.5731")

        tk.Entry(
            coord_frame,
            textvariable=self.tl_lat,
            width=12,
            bg="#3a3a3a",
            fg="#00ff41",
            font=("Courier", 9),
        ).grid(row=0, column=1, padx=2)
        tk.Entry(
            coord_frame,
            textvariable=self.tl_lon,
            width=12,
            bg="#3a3a3a",
            fg="#00ff41",
            font=("Courier", 9),
        ).grid(row=0, column=2, padx=2)

        # Bottom-right coordinates
        tk.Label(
            coord_frame,
            text="BOTTOM-RIGHT:",
            bg="#2a2a2a",
            fg="#ffffff",
            font=("Courier", 9, "bold"),
        ).grid(row=0, column=3, sticky=tk.W, padx=(20, 5))

        self.br_lat = tk.StringVar(value="36.4044")
        self.br_lon = tk.StringVar(value="-105.5700")

        tk.Entry(
            coord_frame,
            textvariable=self.br_lat,
            width=12,
            bg="#3a3a3a",
            fg="#00ff41",
            font=("Courier", 9),
        ).grid(row=0, column=4, padx=2)
        tk.Entry(
            coord_frame,
            textvariable=self.br_lon,
            width=12,
            bg="#3a3a3a",
            fg="#00ff41",
            font=("Courier", 9),
        ).grid(row=0, column=5, padx=2)

        # File selection and processing controls
        file_frame = tk.Frame(control_frame, bg="#2a2a2a")
        file_frame.pack(fill=tk.X, pady=(10, 0))

        self.file_path = tk.StringVar()
        file_entry = tk.Entry(
            file_frame,
            textvariable=self.file_path,
            width=60,
            bg="#3a3a3a",
            fg="#ffffff",
            font=("Courier", 10),
        )
        file_entry.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(
            file_frame,
            text="📁 BROWSE VIDEO",
            command=self.browse_file,
            style="Tactical.TButton",
        ).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(
            file_frame,
            text="📷 START LIVE FEED",
            command=self.start_live_feed,
            style="Tactical.TButton",
        ).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(
            file_frame,
            text="▶️ START MISSION",
            command=self.start_processing,
            style="Tactical.TButton",
        ).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(
            file_frame,
            text="⏹️ ABORT",
            command=self.stop_processing,
            style="Tactical.TButton",
        ).pack(side=tk.LEFT)

        # Progress bar
        self.progress = ttk.Progressbar(control_frame, mode="indeterminate")
        self.progress.pack(fill=tk.X, pady=(10, 0))

    def setup_display_panel(self, parent):
        """Setup the video/image display panel with controls"""
        display_frame = tk.LabelFrame(
            parent,
            text="🎥 LIVE TACTICAL FEED",
            bg="#2a2a2a",
            fg="#00ff41",
            font=("Arial", 12, "bold"),
        )
        display_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Video display area
        self.video_frame = tk.Frame(display_frame, bg="#000000", relief=tk.SUNKEN, bd=2)
        self.video_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 5))

        self.video_label = tk.Label(
            self.video_frame,
            text="🎯 NO ACTIVE FEED\n\nSelect video file and start mission\nfor live tactical detection",
            bg="#000000",
            fg="#00ff41",
            font=("Arial", 14, "bold"),
        )
        self.video_label.pack(expand=True)

        # Video controls
        controls_frame = tk.Frame(display_frame, bg="#2a2a2a")
        controls_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        ttk.Button(
            controls_frame,
            text="▶️ PLAY RESULTS",
            command=self.play_processed_video,
            style="Tactical.TButton",
        ).pack(side=tk.LEFT, padx=(0, 5))

        ttk.Button(
            controls_frame,
            text="⏸️ PAUSE",
            command=self.pause_video,
            style="Tactical.TButton",
        ).pack(side=tk.LEFT, padx=(0, 5))

        ttk.Button(
            controls_frame,
            text="⏹️ STOP",
            command=self.stop_video,
            style="Tactical.TButton",
        ).pack(side=tk.LEFT, padx=(0, 5))

        # Detection info overlay
        self.detection_info = tk.Label(
            controls_frame,
            text="Objects: 0 | Frame: 0",
            bg="#2a2a2a",
            fg="#00ff41",
            font=("Courier", 9),
        )
        self.detection_info.pack(side=tk.RIGHT)

    def setup_stats_panel(self, parent):
        """Setup the statistics panel"""
        stats_frame = tk.LabelFrame(
            parent,
            text="📊 TACTICAL INTELLIGENCE",
            bg="#2a2a2a",
            fg="#00ff41",
            font=("Arial", 12, "bold"),
            width=400,
        )
        stats_frame.pack(side=tk.RIGHT, fill=tk.Y)
        stats_frame.pack_propagate(False)

        # Real-time stats display
        self.stats_text = tk.Text(
            stats_frame,
            height=20,
            width=45,
            bg="#1a1a1a",
            fg="#00ff41",
            font=("Courier", 10),
            insertbackground="#00ff41",
        )
        self.stats_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Initialize with welcome message
        welcome_text = """🦅 R.A.P.T.O.R SYSTEM READY

📡 AWAITING MISSION PARAMETERS
🎯 TACTICAL STATUS: STANDBY
🗺️ GPS SYSTEM: CONFIGURED
⚡ DETECTION MODEL: LOADED

SELECT TARGET VIDEO AND INITIATE SCAN

===============================
LIVE FEATURES:
• Real-time Object Detection
• Live Video Display
• Instant GPS Mapping
• Post-Mission Playback

🔐 CLEARANCE LEVEL: AUTHORIZED
"""
        self.stats_text.insert(1.0, welcome_text)
        self.stats_text.config(state=tk.DISABLED)

    def setup_action_panel(self, parent):
        """Setup the action buttons panel"""
        action_frame = tk.LabelFrame(
            parent,
            text="🎖️ TACTICAL OPERATIONS",
            bg="#2a2a2a",
            fg="#00ff41",
            font=("Arial", 12, "bold"),
            padx=10,
            pady=10,
        )
        action_frame.pack(fill=tk.X, pady=(10, 0))

        # Left side - Analysis buttons
        analysis_frame = tk.Frame(action_frame, bg="#2a2a2a")
        analysis_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Button(
            analysis_frame,
            text="🗺️ GENERATE TACTICAL MAP",
            command=self.generate_map,
            style="Tactical.TButton",
        ).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(
            analysis_frame,
            text="📊 PERFORMANCE ANALYSIS",
            command=self.analyze_performance,
            style="Tactical.TButton",
        ).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(
            analysis_frame,
            text="🌐 VIEW WEB MAP",
            command=self.view_web_map,
            style="Tactical.TButton",
        ).pack(side=tk.LEFT, padx=(0, 10))

        # Right side - System buttons
        system_frame = tk.Frame(action_frame, bg="#2a2a2a")
        system_frame.pack(side=tk.RIGHT)

        ttk.Button(
            system_frame,
            text="🧪 RUN DIAGNOSTICS",
            command=self.run_diagnostics,
            style="Tactical.TButton",
        ).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(
            system_frame,
            text="💾 EXPORT INTEL",
            command=self.export_results,
            style="Tactical.TButton",
        ).pack(side=tk.LEFT)

    def setup_status_bar(self, parent):
        """Setup the status bar"""
        status_frame = tk.Frame(parent, bg="#1a1a1a", relief=tk.SUNKEN, bd=1)
        status_frame.pack(fill=tk.X, pady=(10, 0))

        self.status_var = tk.StringVar(value="🟢 SYSTEM READY - AWAITING ORDERS")
        status_label = ttk.Label(
            status_frame, textvariable=self.status_var, style="Status.TLabel"
        )
        status_label.pack(side=tk.LEFT, padx=10)

        # System time
        self.time_var = tk.StringVar()
        time_label = ttk.Label(
            status_frame, textvariable=self.time_var, style="Status.TLabel"
        )
        time_label.pack(side=tk.RIGHT, padx=10)

        # Update time every second
        self.update_time()

    def update_time(self):
        """Update the system time display"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_var.set(f"🕒 {current_time}")
        self.root.after(1000, self.update_time)

    def browse_file(self):
        """Browse for video file"""
        filename = filedialog.askopenfilename(
            title="Select Target Video for Analysis",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*"),
            ],
        )
        if filename:
            self.file_path.set(filename)
            self.status_var.set(f"🎯 TARGET ACQUIRED: {os.path.basename(filename)}")

    def get_gps_bounds(self):
        """Get GPS bounds from input fields"""
        try:
            return {
                "top_left": {
                    "lat": float(self.tl_lat.get()),
                    "lon": float(self.tl_lon.get()),
                },
                "top_right": {
                    "lat": float(self.tl_lat.get()),
                    "lon": float(self.br_lon.get()),
                },
                "bottom_left": {
                    "lat": float(self.br_lat.get()),
                    "lon": float(self.tl_lon.get()),
                },
                "bottom_right": {
                    "lat": float(self.br_lat.get()),
                    "lon": float(self.br_lon.get()),
                },
            }
        except ValueError:
            messagebox.showerror(
                "GPS Error", "Invalid GPS coordinates. Please check input values."
            )
            return None

    def start_processing(self):
        """Start video processing mission with live display"""
        if not self.file_path.get():
            messagebox.showerror(
                "Mission Error",
                "No target video selected. Please browse for a video file.",
            )
            return

        if not os.path.exists(self.file_path.get()):
            messagebox.showerror("Mission Error", "Selected video file not found.")
            return

        if self.is_processing:
            messagebox.showwarning(
                "Mission Warning", "Processing mission already in progress."
            )
            return

        bounds = self.get_gps_bounds()
        if not bounds:
            return

        self.is_processing = True
        self.progress.start()
        self.status_var.set("🔄 MISSION ACTIVE - LIVE TACTICAL ANALYSIS")

        # Update video display
        self.video_label.config(
            text="🔄 INITIALIZING LIVE FEED\n\nAI Detection Starting...", fg="#ffaa00"
        )

        # Update stats display
        self.update_stats_display(
            "🚀 MISSION INITIATED\n\nLive video analysis active...\nReal-time object detection...\n"
        )

        # Start processing in separate thread
        threading.Thread(
            target=self.process_video_thread,
            args=(self.file_path.get(), bounds),
            daemon=True,
        ).start()

    def start_live_feed(self):
        """Start a mission using the live webcam feed."""
        if self.is_processing:
            messagebox.showwarning(
                "Mission Warning", "Processing mission already in progress."
            )
            return

        self.is_processing = True
        self.progress.start()
        self.status_var.set("📷 LIVE FEED ACTIVE - REAL-TIME TACTICAL ANALYSIS")
        self.video_label.config(text="📷 INITIALIZING LIVE FEED...", fg="#ffaa00")
        self.update_stats_display(
            "🚀 LIVE MISSION INITIATED\n\nReal-time object detection from webcam...\n"
        )

        # Start live processing in a separate thread
        threading.Thread(target=self.process_live_thread, daemon=True).start()

    def process_live_thread(self):
        """Process the live feed in a separate thread."""
        try:
            # Initialize processor for live feed.
            self.processor = RaptorProcessor(
                gps_bounds=None, gui_queue=self.video_queue
            )

            # Process the live stream (camera 0). This will loop until stop() is called.
            detections = self.processor.process_video(video_path=0)

            # This part will be reached only after stop_processing is called.
            self.root.after(0, self.processing_complete, detections)

        except Exception as e:
            self.root.after(0, self.processing_error, str(e))

    def process_video_thread(self, video_path, bounds):
        """Process video in separate thread with live display"""
        try:
            # Initialize processor
            self.processor = RaptorProcessor(
                gps_bounds=bounds, gui_queue=self.video_queue
            )

            # Create output video path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"output/videos/raptor_mission_{timestamp}.mp4"

            # Process video with live updates
            detections = self.processor.process_video_live(
                video_path, output_path, process_every_n_frames=3
            )

            # Store processed video path
            self.processed_video_path = output_path

            # Update UI on main thread
            self.root.after(0, self.processing_complete, detections)

        except Exception as e:
            self.root.after(0, self.processing_error, str(e))

    def update_video_display(self):
        """Update video display from queue"""
        try:
            while not self.video_queue.empty():
                frame_data = self.video_queue.get_nowait()

                if frame_data["type"] == "frame":
                    self.display_frame(
                        frame_data["frame"],
                        frame_data["detections"],
                        frame_data["frame_num"],
                    )
                elif frame_data["type"] == "progress":
                    self.update_progress_display(frame_data["message"])
        except queue.Empty:
            pass

        # Schedule next update
        self.root.after(50, self.update_video_display)  # 20 FPS update rate

    def display_frame(self, frame, detections, frame_num):
        """Display video frame with detections in GUI"""
        try:
            # Resize frame to fit display
            display_height = 400
            height, width = frame.shape[:2]
            aspect_ratio = width / height
            display_width = int(display_height * aspect_ratio)

            # Resize frame
            display_frame = cv2.resize(frame, (display_width, display_height))

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

            # Convert to PIL Image
            pil_image = Image.fromarray(rgb_frame)

            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(pil_image)

            # Update video label
            self.video_label.config(image=photo, text="")
            self.video_label.image = photo  # Keep a reference

            # Update detection info
            detection_count = len(detections) if detections else 0
            self.detection_info.config(
                text=f"Objects: {detection_count} | Frame: {frame_num}"
            )

        except Exception as e:
            print(f"Display error: {e}")

    def update_progress_display(self, message):
        """Update progress in stats panel"""
        self.update_stats_display(f"\n{message}")

    def play_processed_video(self):
        """Play the processed video with detections"""
        if not self.processed_video_path or not os.path.exists(
            self.processed_video_path
        ):
            messagebox.showwarning(
                "Playback", "No processed video available. Run a mission first."
            )
            return

        if self.video_playback_active:
            messagebox.showinfo("Playback", "Video already playing.")
            return

        self.video_playback_active = True
        self.status_var.set("📺 PLAYING MISSION RESULTS")

        # Start playback in separate thread
        threading.Thread(
            target=self.playback_thread, args=(self.processed_video_path,), daemon=True
        ).start()

    def playback_thread(self, video_path):
        """Playback processed video in separate thread"""
        try:
            cap = cv2.VideoCapture(video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_delay = 1.0 / fps

            frame_num = 0
            while self.video_playback_active and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Send frame to display queue
                self.video_queue.put(
                    {
                        "type": "frame",
                        "frame": frame,
                        "detections": [],  # Already annotated
                        "frame_num": frame_num,
                    }
                )

                frame_num += 1
                time.sleep(frame_delay)

            cap.release()
            self.video_playback_active = False

            # Update status
            self.root.after(0, lambda: self.status_var.set("✅ PLAYBACK COMPLETE"))

        except Exception as e:
            self.video_playback_active = False
            print(f"Playback error: {e}")

    def pause_video(self):
        """Pause video playback"""
        self.video_playback_active = False
        self.status_var.set("⏸️ PLAYBACK PAUSED")

    def stop_video(self):
        """Stop video playback"""
        self.video_playback_active = False
        self.status_var.set("⏹️ PLAYBACK STOPPED")

        # Reset video display
        self.video_label.config(
            image="",
            text="🎯 TACTICAL FEED READY\n\nSelect mission or play results",
            fg="#00ff41",
        )
        self.video_label.image = None
        self.detection_info.config(text="Objects: 0 | Frame: 0")

    def processing_complete(self, detections):
        """Handle processing completion"""
        self.is_processing = False
        self.progress.stop()
        self.current_detections = detections

        # Update video display
        self.video_label.config(
            text="✅ MISSION COMPLETE\n\nClick 'PLAY RESULTS' to review\ndetected objects",
            fg="#00ff41",
        )

        # Save detections
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        detection_file = f"output/detections/mission_detections_{timestamp}.json"

        try:
            with open(detection_file, "w") as f:
                json.dump(detections, f, indent=2)
        except Exception as e:
            print(f"Failed to save detections: {e}")

        # Update statistics display
        self.update_mission_stats(detections)

        self.status_var.set(
            f"✅ MISSION COMPLETE - {len(detections)} TACTICAL OBJECTS IDENTIFIED"
        )

        messagebox.showinfo(
            "Mission Complete",
            f"Mission successful!\n\n"
            f"Objects detected: {len(detections)}\n"
            f"Intelligence saved to: {detection_file}\n\n"
            f"Click 'PLAY RESULTS' to review annotated video!",
        )

    def processing_error(self, error_msg):
        """Handle processing errors"""
        self.is_processing = False
        self.progress.stop()
        self.status_var.set("❌ MISSION FAILED - SYSTEM ERROR")

        self.video_label.config(
            text="❌ MISSION FAILED\n\nCheck system status\nand try again", fg="#ff4444"
        )

        self.update_stats_display(f"\n❌ MISSION FAILED\nError: {error_msg}")

        messagebox.showerror(
            "Mission Failed", f"Processing failed with error:\n{error_msg}"
        )

    def stop_processing(self):
        """Stop any active processing mission (file or live)."""
        if not self.is_processing:
            messagebox.showinfo("Status", "No active mission to abort.")
            return

        # This is the key change: we signal the processor to stop its loop.
        if self.processor:
            self.processor.stop()

        self.is_processing = False
        self.video_playback_active = False  # Also stop any playback
        self.progress.stop()
        self.status_var.set("⏹️ MISSION ABORTED BY OPERATOR")
        self.update_stats_display("\n⏹️ MISSION ABORTED\n")

        self.video_label.config(
            text="⏹️ MISSION ABORTED\n\nSelect new target\nand start mission",
            fg="#ffaa00",
        )

    def update_stats_display(self, new_text):
        """Update the statistics display"""
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.insert(tk.END, new_text)
        self.stats_text.see(tk.END)  # Scroll to bottom
        self.stats_text.config(state=tk.DISABLED)

    def update_mission_stats(self, detections):
        """Update display with mission statistics"""
        if not detections:
            self.update_stats_display("\n⚠️ NO OBJECTS DETECTED")
            return

        # Calculate statistics
        import pandas as pd

        df = pd.DataFrame(detections)

        stats_text = f"""
✅ MISSION COMPLETE

📊 TACTICAL INTELLIGENCE:
━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 Total Objects: {len(detections)}
📈 Avg Confidence: {df['confidence'].mean():.1%}

🚗 OBJECT BREAKDOWN:
"""

        class_counts = df["class"].value_counts()
        for obj_class, count in class_counts.items():
            percentage = (count / len(detections)) * 100
            stats_text += f"   {obj_class.upper()}: {count} ({percentage:.1f}%)\n"

        # Confidence analysis
        high_conf = len(df[df["confidence"] > 0.8])
        medium_conf = len(df[(df["confidence"] >= 0.6) & (df["confidence"] <= 0.8)])
        low_conf = len(df[df["confidence"] < 0.6])

        stats_text += f"""
🎚️ CONFIDENCE LEVELS:
   HIGH (>80%): {high_conf}
   MEDIUM (60-80%): {medium_conf}
   LOW (<60%): {low_conf}

🗺️ GPS STATUS:
"""

        with_gps = len([d for d in detections if "gps" in d])
        stats_text += f"   MAPPED OBJECTS: {with_gps}/{len(detections)}\n"
        stats_text += f"   COVERAGE: {with_gps/len(detections)*100:.1f}%\n"

        # Tactical assessment
        stats_text += f"""
🎖️ TACTICAL ASSESSMENT:
   SURVEILLANCE: {'OPTIMAL' if df['confidence'].mean() > 0.8 else 'ADEQUATE' if df['confidence'].mean() > 0.6 else 'LIMITED'}
   READINESS: {'DEPLOYMENT READY' if len(detections) > 10 and df['confidence'].mean() > 0.7 else 'OPERATIONAL'}
   
📺 CLICK 'PLAY RESULTS' TO REVIEW
⏰ ANALYSIS: {datetime.now().strftime('%H:%M:%S')}
━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

        # Clear and update display
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats_text)
        self.stats_text.config(state=tk.DISABLED)

    def generate_map(self):
        """Generate tactical map"""
        if not self.current_detections:
            messagebox.showwarning(
                "Map Generation", "No detection data available. Run a mission first."
            )
            return

        try:
            self.status_var.set("🗺️ GENERATING TACTICAL MAP...")

            # Save current detections temporarily
            temp_file = "temp_detections.json"
            with open(temp_file, "w") as f:
                json.dump(self.current_detections, f)

            # Create mapper and generate map
            if not hasattr(self, "mapper") or self.mapper is None:
                self.mapper = TacticalQGISMapper()

            self.mapper.load_detections(temp_file)
            self.mapper.create_tactical_map_project("raptor_tactical_map")

            # Clean up
            if os.path.exists(temp_file):
                os.remove(temp_file)

            self.status_var.set("✅ TACTICAL MAP GENERATED")
            messagebox.showinfo(
                "Success",
                "Tactical map generated successfully!\nCheck output/maps/ directory.",
            )

        except Exception as e:
            self.status_var.set("❌ MAP GENERATION FAILED")
            messagebox.showerror("Error", f"Map generation failed:\n{str(e)}")

    def analyze_performance(self):
        """Run performance analysis"""
        if not self.current_detections:
            messagebox.showwarning(
                "Analysis", "No detection data available. Run a mission first."
            )
            return

        try:
            self.status_var.set("📊 ANALYZING PERFORMANCE...")

            # Save current detections for analysis
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            analysis_file = f"output/detections/analysis_data_{timestamp}.json"

            with open(analysis_file, "w") as f:
                json.dump(self.current_detections, f, indent=2)

            # Run performance analysis
            analyzer = PerformanceAnalyzer(analysis_file)
            analyzer.analyze_detection_stats()
            analyzer.create_visualizations()
            analyzer.tactical_assessment()
            report_file = analyzer.generate_performance_report()

            self.status_var.set("✅ PERFORMANCE ANALYSIS COMPLETE")

            result_msg = f"Performance analysis complete!\n\n"
            result_msg += f"Report saved: {report_file}\n"
            result_msg += f"Visualizations: output/analysis/\n\n"
            result_msg += f"Open analysis report?"

            if messagebox.askyesno("Analysis Complete", result_msg):
                if report_file and os.path.exists(report_file):
                    (
                        os.startfile(report_file)
                        if os.name == "nt"
                        else os.system(f'open "{report_file}"')
                    )

        except Exception as e:
            self.status_var.set("❌ ANALYSIS FAILED")
            messagebox.showerror("Error", f"Performance analysis failed:\n{str(e)}")

    def view_web_map(self):
        """Open interactive web map"""
        if not self.current_detections:
            messagebox.showwarning(
                "Web Map", "No detection data available. Run a mission first."
            )
            return

        try:
            self.status_var.set("🌐 GENERATING WEB MAP...")

            # Save current detections
            temp_file = "temp_web_detections.json"
            with open(temp_file, "w") as f:
                json.dump(self.current_detections, f)

            # Create web map
            if not hasattr(self, "mapper") or self.mapper is None:
                self.mapper = TacticalQGISMapper()

            self.mapper.load_detections(temp_file)
            web_map_path = "output/maps/raptor_tactical_map.html"
            self.mapper.create_web_map(web_map_path)

            # Clean up
            if os.path.exists(temp_file):
                os.remove(temp_file)

            # Open in browser
            if os.path.exists(web_map_path):
                webbrowser.open(f"file://{os.path.abspath(web_map_path)}")
                self.status_var.set("✅ WEB MAP OPENED IN BROWSER")
            else:
                raise Exception("Web map file not created")

        except Exception as e:
            self.status_var.set("❌ WEB MAP FAILED")
            messagebox.showerror("Error", f"Web map creation failed:\n{str(e)}")

    def run_diagnostics(self):
        """Run system diagnostics"""
        try:
            self.status_var.set("🧪 RUNNING SYSTEM DIAGNOSTICS...")

            # Run test suite in separate thread to avoid blocking UI
            def run_tests():
                test_suite = RaptorTestSuite()
                test_suite.run_all_tests()

                # Show results on main thread
                def show_results():
                    passed = len(
                        [
                            r
                            for r in test_suite.test_results.values()
                            if r["status"] == "PASS"
                        ]
                    )
                    total = len(test_suite.test_results)
                    success_rate = (passed / total) * 100 if total > 0 else 0

                    self.status_var.set(
                        f"✅ DIAGNOSTICS COMPLETE - {success_rate:.0f}% PASS RATE"
                    )

                    result_msg = f"System Diagnostics Complete\n\n"
                    result_msg += f"Tests Passed: {passed}/{total}\n"
                    result_msg += f"Success Rate: {success_rate:.1f}%\n\n"

                    if success_rate >= 80:
                        result_msg += "🟢 SYSTEM OPERATIONAL\n"
                    elif success_rate >= 60:
                        result_msg += "🟡 SYSTEM FUNCTIONAL\n"
                    else:
                        result_msg += "🔴 SYSTEM NEEDS ATTENTION\n"

                    result_msg += "\nView detailed test report?"

                    if messagebox.askyesno("Diagnostics Complete", result_msg):
                        # Open test report if available
                        test_dir = Path("output/testing")
                        if test_dir.exists():
                            reports = list(test_dir.glob("raptor_test_report_*.md"))
                            if reports:
                                latest_report = max(
                                    reports, key=lambda x: x.stat().st_mtime
                                )
                                (
                                    os.startfile(latest_report)
                                    if os.name == "nt"
                                    else os.system(f'open "{latest_report}"')
                                )

                self.root.after(0, show_results)

            threading.Thread(target=run_tests, daemon=True).start()

        except Exception as e:
            self.status_var.set("❌ DIAGNOSTICS FAILED")
            messagebox.showerror("Error", f"Diagnostics failed:\n{str(e)}")

    def export_results(self):
        """Export mission results"""
        if not self.current_detections:
            messagebox.showwarning(
                "Export", "No detection data available. Run a mission first."
            )
            return

        try:
            # Ask user for export location
            export_dir = filedialog.askdirectory(title="Select Export Directory")
            if not export_dir:
                return

            self.status_var.set("💾 EXPORTING INTELLIGENCE...")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"raptor_mission_{timestamp}"

            # Export JSON
            json_file = os.path.join(export_dir, f"{base_name}.json")
            with open(json_file, "w") as f:
                json.dump(self.current_detections, f, indent=2)

            # Export CSV
            import pandas as pd

            df = pd.DataFrame(self.current_detections)
            csv_file = os.path.join(export_dir, f"{base_name}.csv")
            df.to_csv(csv_file, index=False)

            # Create summary report
            summary_file = os.path.join(export_dir, f"{base_name}_summary.txt")
            with open(summary_file, "w") as f:
                f.write(f"R.A.P.T.O.R MISSION SUMMARY\n")
                f.write(f"{'='*40}\n")
                f.write(
                    f"Mission Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
                f.write(f"Total Objects: {len(self.current_detections)}\n")
                f.write(f"Average Confidence: {df['confidence'].mean():.1%}\n")
                f.write(f"\nObject Breakdown:\n")

                class_counts = df["class"].value_counts()
                for obj_class, count in class_counts.items():
                    f.write(f"  {obj_class}: {count}\n")

            self.status_var.set("✅ INTELLIGENCE EXPORTED")

            messagebox.showinfo(
                "Export Complete",
                f"Mission intelligence exported to:\n\n"
                f"• {json_file}\n"
                f"• {csv_file}\n"
                f"• {summary_file}",
            )

        except Exception as e:
            self.status_var.set("❌ EXPORT FAILED")
            messagebox.showerror("Error", f"Export failed:\n{str(e)}")

    def run(self):
        """Start the dashboard application"""
        self.root.mainloop()

    def on_closing(self):
        """Handle application closing"""
        if self.is_processing:
            if messagebox.askokcancel("Quit", "Mission in progress. Abort and quit?"):
                self.is_processing = False
                self.video_playback_active = False
                self.root.destroy()
        else:
            self.root.destroy()


def main():
    """Main function to launch R.A.P.T.O.R dashboard"""
    print("🦅 Launching R.A.P.T.O.R Tactical Dashboard with Live Video...")

    try:
        dashboard = RaptorDashboard()
        dashboard.root.protocol("WM_DELETE_WINDOW", dashboard.on_closing)
        dashboard.run()
    except Exception as e:
        print(f"❌ Dashboard launch failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
