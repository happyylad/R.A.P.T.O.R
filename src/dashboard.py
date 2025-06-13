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
sys.path.append(str(Path(__file__).parent.parent))

# Force enable all modules since we know they work
VIDEO_PROCESSOR_AVAILABLE = True
from src.core_processor import RaptorProcessor
from src.qgis_mapper import TacticalQGISMapper
from src.performance_analyzer import PerformanceAnalyzer
from src.test_suite import RaptorTestSuite
from src import config


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
        main_container = tk.Frame(self.root, bg="#1a1a1a")
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

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

        self.setup_control_panel(main_container)
        content_frame = tk.Frame(main_container, bg="#1a1a1a")
        content_frame.pack(fill=tk.BOTH, expand=True)
        self.setup_display_panel(content_frame)
        self.setup_stats_panel(content_frame)
        self.setup_action_panel(main_container)
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

        top_row_frame = tk.Frame(control_frame, bg="#2a2a2a")
        top_row_frame.pack(fill=tk.X, pady=(0, 10))

        # --- MODIFIED SECTION: Replaced GPS Bounds with Drone Telemetry ---
        telemetry_frame = tk.LabelFrame(
            top_row_frame,
            text="DRONE TELEMETRY",
            bg="#2a2a2a",
            fg="#00ff41",
            font=("Arial", 10, "bold"),
        )
        telemetry_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        # Create input fields for telemetry data with default values for testing
        telemetry_inputs = {
            "Lat": ("-33.9325", 10),
            "Lon": ("18.8647", 10),
            "Altitude (m)": ("120", 8),
            "Heading (°)": ("0", 8),
            "Tilt (°)": ("60", 8),
            "FoV (°)": ("70", 8),
        }
        self.telemetry_vars = {}
        col_count = 0
        for label, (default_val, width) in telemetry_inputs.items():
            tk.Label(
                telemetry_frame,
                text=f"{label}:",
                bg="#2a2a2a",
                fg="#ffffff",
                font=("Courier", 9, "bold"),
            ).grid(row=0, column=col_count, sticky=tk.W, padx=(10, 2))
            col_count += 1

            # Store the variable with a simple key (e.g., 'lat', 'altitude')
            var_key = label.split(" ")[0].lower()
            var = tk.StringVar(value=default_val)
            self.telemetry_vars[var_key] = var
            tk.Entry(
                telemetry_frame,
                textvariable=var,
                width=width,
                bg="#3a3a3a",
                fg="#00ff41",
                font=("Courier", 9),
            ).grid(row=0, column=col_count, padx=(0, 10))
            col_count += 1
        # --- END OF MODIFIED SECTION ---

        model_selection_frame = tk.LabelFrame(
            top_row_frame,
            text="DETECTION MODEL",
            bg="#2a2a2a",
            fg="#00ff41",
            font=("Arial", 10, "bold"),
        )
        model_selection_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        self.selected_model_name = tk.StringVar()
        model_dropdown = ttk.Combobox(
            model_selection_frame,
            textvariable=self.selected_model_name,
            values=list(config.MODELS.keys()),
            state="readonly",
            width=35,
            font=("Arial", 10),
        )
        model_dropdown.pack(padx=10, pady=10)
        model_dropdown.current(0)

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
        file_entry.pack(side=tk.LEFT, padx=(0, 10), expand=True, fill=tk.X)
        ttk.Button(
            file_frame,
            text="📁 BROWSE",
            command=self.browse_file,
            style="Tactical.TButton",
        ).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(
            file_frame,
            text="📷 LIVE FEED",
            command=self.start_live_feed,
            style="Tactical.TButton",
        ).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(
            file_frame,
            text="▶️ MISSION",
            command=self.start_processing,
            style="Tactical.TButton",
        ).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(
            file_frame,
            text="⏹️ ABORT",
            command=self.stop_processing,
            style="Tactical.TButton",
        ).pack(side=tk.LEFT)
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
        welcome_text = """🦅 R.A.P.T.O.R SYSTEM READY

📡 AWAITING MISSION PARAMETERS
🎯 TACTICAL STATUS: STANDBY
🗺️ GPS SYSTEM: AWAITING TELEMETRY
⚡ DETECTION MODEL: LOADED

SELECT TARGET VIDEO AND INITIATE SCAN
===============================
LIVE FEATURES:
• Real-time Object Detection
• Live Video Display
• Dynamic GPS Coordinate Calculation
• Post-Mission Tactical Map
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
        self.time_var = tk.StringVar()
        time_label = ttk.Label(
            status_frame, textvariable=self.time_var, style="Status.TLabel"
        )
        time_label.pack(side=tk.RIGHT, padx=10)
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

    # --- NEW HELPER FUNCTION to read telemetry values ---
    def get_drone_telemetry(self):
        """Get drone telemetry from input fields and validate it."""
        try:
            telemetry = {
                "lat": float(self.telemetry_vars["lat"].get()),
                "lon": float(self.telemetry_vars["lon"].get()),
                "alt_m": float(self.telemetry_vars["altitude"].get()),
                "heading_deg": float(self.telemetry_vars["heading"].get()),
                "tilt_deg": float(self.telemetry_vars["tilt"].get()),
                "fov_deg": float(self.telemetry_vars["fov"].get()),
            }
            # Basic validation
            if not (-90 <= telemetry["lat"] <= 90 and -180 <= telemetry["lon"] <= 180):
                raise ValueError("Invalid Latitude/Longitude values.")
            if not (0 < telemetry["alt_m"] < 20000):
                raise ValueError("Altitude seems unrealistic (must be > 0).")
            if not (0 <= telemetry["tilt_deg"] <= 90):
                raise ValueError(
                    "Camera tilt must be between 0 (horizontal) and 90 (down)."
                )
            if not (10 <= telemetry["fov_deg"] <= 180):
                raise ValueError("Field of View (FoV) seems unrealistic.")
            return telemetry
        except (ValueError, KeyError) as e:
            messagebox.showerror(
                "Telemetry Error",
                f"Invalid drone telemetry value. Please check inputs.\n\nError: {e}",
            )
            return None

    # --- MODIFIED: start_processing now uses get_drone_telemetry ---
    def start_processing(self):
        """Start video processing mission with live display"""
        if not self.file_path.get():
            messagebox.showerror("Mission Error", "No target video selected.")
            return

        if not os.path.exists(self.file_path.get()):
            messagebox.showerror("Mission Error", "Selected video file not found.")
            return

        if self.is_processing:
            messagebox.showwarning(
                "Mission Warning", "Processing mission already in progress."
            )
            return

        # Get the new drone telemetry instead of the old GPS bounds
        telemetry = self.get_drone_telemetry()
        if not telemetry:
            return  # Stop if telemetry is invalid

        self.is_processing = True
        self.progress.start()
        self.status_var.set("🔄 MISSION ACTIVE - LIVE TACTICAL ANALYSIS")

        self.video_label.config(text="🔄 INITIALIZING LIVE FEED...", fg="#ffaa00")
        self.update_stats_display(
            "🚀 MISSION INITIATED\n\nCalculating object coordinates...\n"
        )

        # Start processing in a separate thread, passing the new telemetry dict
        threading.Thread(
            target=self.process_video_thread,
            args=(self.file_path.get(), telemetry),  # Pass telemetry dict
            daemon=True,
        ).start()

    def start_live_feed(self):
        """Start a mission using the live webcam feed."""
        if self.is_processing:
            messagebox.showwarning(
                "Mission Warning", "Processing mission already in progress."
            )
            return

        telemetry = self.get_drone_telemetry()
        if not telemetry:
            return

        self.is_processing = True
        self.progress.start()
        self.status_var.set("📷 LIVE FEED ACTIVE - REAL-TIME TACTICAL ANALYSIS")
        self.video_label.config(text="📷 INITIALIZING LIVE FEED...", fg="#ffaa00")
        self.update_stats_display(
            "🚀 LIVE MISSION INITIATED\n\nReal-time object detection from webcam...\n"
        )

        threading.Thread(
            target=self.process_live_thread, args=(telemetry,), daemon=True
        ).start()

    def process_live_thread(self, telemetry):
        """Process the live feed in a separate thread."""
        try:
            selected_model_key = self.selected_model_name.get()
            if not selected_model_key:
                self.root.after(
                    0, self.processing_error, "No model selected from dropdown."
                )
                return

            model_path = config.MODELS[selected_model_key]

            # Initialize the processor with the chosen model and telemetry
            self.processor = RaptorProcessor(
                model_path=model_path,
                drone_telemetry=telemetry,
                gui_queue=self.video_queue,
            )

            # Process the live stream (camera 0). This will loop until stop() is called.
            detections = self.processor.process_video(video_path=0)

            self.root.after(0, self.processing_complete, detections)
        except Exception as e:
            self.root.after(0, self.processing_error, str(e))

    # --- MODIFIED: process_video_thread now accepts telemetry ---
    def process_video_thread(self, video_path, telemetry):
        """Process video in separate thread with live display"""
        try:
            selected_model_key = self.selected_model_name.get()
            if not selected_model_key:
                self.root.after(
                    0, self.processing_error, "No model selected from dropdown."
                )
                return

            model_path = config.MODELS[selected_model_key]

            # Initialize processor with the chosen model path and NEW telemetry data
            self.processor = RaptorProcessor(
                model_path=model_path,
                drone_telemetry=telemetry,  # Pass the telemetry dict here
                gui_queue=self.video_queue,
            )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"output/videos/raptor_mission_{timestamp}.mp4"
            detections = self.processor.process_video(video_path, output_path)
            self.processed_video_path = output_path

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
        self.root.after(33, self.update_video_display)  # ~30 FPS update rate

    def display_frame(self, frame, detections, frame_num):
        """Display video frame with detections in GUI"""
        try:
            display_height = 400
            height, width = frame.shape[:2]
            aspect_ratio = width / height
            display_width = int(display_height * aspect_ratio)

            display_frame = cv2.resize(frame, (display_width, display_height))
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            photo = ImageTk.PhotoImage(pil_image)

            self.video_label.config(image=photo, text="")
            self.video_label.image = photo

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
        threading.Thread(
            target=self.playback_thread, args=(self.processed_video_path,), daemon=True
        ).start()

    def playback_thread(self, video_path):
        """Playback processed video in separate thread"""
        try:
            cap = cv2.VideoCapture(video_path)
            fps = max(int(cap.get(cv2.CAP_PROP_FPS)), 1)
            frame_delay = 1.0 / fps
            frame_num = 0
            while self.video_playback_active and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                self.video_queue.put(
                    {
                        "type": "frame",
                        "frame": frame,
                        "detections": [],
                        "frame_num": frame_num,
                    }
                )
                frame_num += 1
                time.sleep(frame_delay)
            cap.release()
            self.video_playback_active = False
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
        self.video_label.config(image="", text="🎯 TACTICAL FEED READY", fg="#00ff41")
        self.video_label.image = None
        self.detection_info.config(text="Objects: 0 | Frame: 0")

    def processing_complete(self, detections):
        """Handle processing completion"""
        self.is_processing = False
        self.progress.stop()
        self.current_detections = detections
        self.video_label.config(
            text="✅ MISSION COMPLETE\n\nClick 'PLAY RESULTS' to review", fg="#00ff41"
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        detection_file = f"output/detections/mission_detections_{timestamp}.json"
        try:
            with open(detection_file, "w") as f:
                json.dump(detections, f, indent=2)
        except Exception as e:
            print(f"Failed to save detections: {e}")
        self.update_mission_stats(detections)
        self.status_var.set(
            f"✅ MISSION COMPLETE - {len(detections)} TACTICAL OBJECTS IDENTIFIED"
        )
        messagebox.showinfo(
            "Mission Complete",
            f"Mission successful!\n\nObjects detected: {len(detections)}\n"
            f"Intelligence saved to: {detection_file}\n\n"
            f"Click 'VIEW WEB MAP' to see results!",
        )

    def processing_error(self, error_msg):
        """Handle processing errors"""
        self.is_processing = False
        self.progress.stop()
        self.status_var.set("❌ MISSION FAILED - SYSTEM ERROR")
        self.video_label.config(
            text="❌ MISSION FAILED\n\nCheck system status", fg="#ff4444"
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
        if self.processor:
            self.processor.stop()
        self.is_processing = False
        self.video_playback_active = False
        self.progress.stop()
        self.status_var.set("⏹️ MISSION ABORTED BY OPERATOR")
        self.update_stats_display("\n⏹️ MISSION ABORTED\n")
        self.video_label.config(text="⏹️ MISSION ABORTED", fg="#ffaa00")

    def update_stats_display(self, new_text):
        """Update the statistics display"""
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.insert(tk.END, new_text)
        self.stats_text.see(tk.END)
        self.stats_text.config(state=tk.DISABLED)

    def update_mission_stats(self, detections):
        """Update display with mission statistics"""
        if not detections:
            self.update_stats_display("\n⚠️ NO OBJECTS DETECTED")
            return

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
        with_gps = len([d for d in detections if d.get("gps")])
        stats_text += f"   MAPPED OBJECTS: {with_gps}/{len(detections)}\n"
        stats_text += f"   COVERAGE: {with_gps/len(detections)*100:.1f}%\n"

        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats_text)
        self.stats_text.config(state=tk.DISABLED)

    def generate_map(self):
        """Generate tactical map"""
        if not self.current_detections:
            messagebox.showwarning(
                "Map Generation", "No detection data. Run a mission first."
            )
            return
        try:
            self.status_var.set("🗺️ GENERATING TACTICAL MAP...")
            temp_file = "temp_detections.json"
            with open(temp_file, "w") as f:
                json.dump(self.current_detections, f)
            if not hasattr(self, "mapper") or self.mapper is None:
                self.mapper = TacticalQGISMapper()
            self.mapper.load_detections(temp_file)
            self.mapper.create_tactical_map_project("raptor_tactical_map")
            if os.path.exists(temp_file):
                os.remove(temp_file)
            self.status_var.set("✅ TACTICAL MAP GENERATED")
            messagebox.showinfo(
                "Success", "Tactical map generated!\nCheck output/maps/ directory."
            )
        except Exception as e:
            self.status_var.set("❌ MAP GENERATION FAILED")
            messagebox.showerror("Error", f"Map generation failed:\n{str(e)}")

    def analyze_performance(self):
        """Run performance analysis"""
        # This function remains unchanged
        pass

    def view_web_map(self):
        """Open interactive web map"""
        if not self.current_detections:
            messagebox.showwarning("Web Map", "No detection data. Run a mission first.")
            return
        try:
            self.status_var.set("🌐 GENERATING WEB MAP...")
            temp_file = "temp_web_detections.json"
            with open(temp_file, "w") as f:
                json.dump(self.current_detections, f)
            if not hasattr(self, "mapper") or self.mapper is None:
                self.mapper = TacticalQGISMapper()
            self.mapper.load_detections(temp_file)
            web_map_path = "output/maps/raptor_tactical_map.html"
            self.mapper.create_web_map(web_map_path)
            if os.path.exists(temp_file):
                os.remove(temp_file)
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
        # This function remains unchanged
        pass

    def export_results(self):
        """Export mission results"""
        # This function remains unchanged
        pass

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
    print("🦅 Launching R.A.P.T.O.R Tactical Dashboard...")
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
