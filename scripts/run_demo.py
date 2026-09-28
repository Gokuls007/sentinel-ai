"""
Sentinel AI — Demo Recording Engine
Processes a video through the full pipeline and outputs an annotated MP4 + JSON alert log.

Usage:
    python scripts/run_demo.py --demo corridor            # downloads the sample if needed
    python scripts/run_demo.py --demo hallway
    python scripts/run_demo.py --source my_clip.mp4 --config config/demo/corridor_demo.json
"""
import cv2
import time
import json
import os
import sys
import subprocess
import imageio_ffmpeg
from datetime import datetime
from pathlib import Path

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

from core.pipeline import SentinelPipeline  # noqa: E402
from config.settings import SentinelConfig  # noqa: E402
from core.samples import SAMPLES, ensure_sample  # noqa: E402
from main import apply_demo_config  # noqa: E402


class DemoEngine:
    def __init__(self, video_path: str, output_path: str = None, config_path: str = None,
                 sector_id: str = "SECTOR-04-NORTH", scenario: str = None):
        self.video_path = video_path
        self.sector_id = sector_id
        self.scenario = scenario
        
        # Auto-deduce output path if not given
        if output_path is None:
            base = Path(video_path).stem.replace("_sample", "_demo")
            output_path = os.path.join("outputs", f"{base}.mp4")
        self.output_path = output_path
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {video_path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        native_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2 * 2
        native_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2 * 2

        # Same configuration path as the live server: .env, then the scenario file
        # (zones + overrides). Frames are processed at the video's native size, and the
        # zone monitor normalises coordinates against that size.
        self.sentinel_config = SentinelConfig.from_env()
        self.sentinel_config.source = video_path
        self.sentinel_config.frame_width, self.sentinel_config.frame_height = native_w, native_h
        self.sentinel_config.target_fps = int(round(self.fps))
        if config_path and os.path.exists(config_path):
            apply_demo_config(self.sentinel_config, config_path)
            print(f"Loaded zones and overrides from {config_path}")
        self.pipeline = SentinelPipeline(self.sentinel_config)
        self.pipeline.clip_recorder.set_fps(self.fps)
        # Real wall-clock timestamps (video timeline anchored at start) for the event DB.
        self.time_origin = time.time()
        self.width = None
        self.height = None
        self.process = None
        self.alerts_log = []

    def _init_ffmpeg(self, w, h):
        self.width = w // 2 * 2
        self.height = h // 2 * 2
        
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        self.ffmpeg_cmd = [
            ffmpeg_exe,
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{self.width}x{self.height}',
            '-pix_fmt', 'bgr24',
            '-r', str(self.fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'ultrafast',
            '-crf', '23',
            self.output_path
        ]
        
        print(f"Initializing FFmpeg Pipe: {self.width}x{self.height} @ {self.fps}fps")
        self.process = subprocess.Popen(self.ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def _draw_glass_header(self, frame, stats):
        """Resolution-adaptive semi-transparent header bar."""
        header_height = max(50, int(self.height * 0.08))
        overlay = frame.copy()
        
        # Glass background
        cv2.rectangle(overlay, (0, 0), (self.width, header_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.line(frame, (0, header_height), (self.width, header_height), (255, 229, 0), 1)
        
        # Title — adaptive font size
        title_scale = max(0.5, self.width / 1600.0)
        title_text = "SENTINEL AI // HUD_ACTIVE"
        cv2.putText(frame, title_text, (20, int(header_height * 0.65)), 
                    cv2.FONT_HERSHEY_SIMPLEX, title_scale, (255, 229, 0), 2)
        
        # Metrics — positioned relative to width
        metrics = [
            f"SECTOR: {self.sector_id}",
            f"TRACKS: {stats.get('active_tracks', 0)}",
            f"ALERTS: {stats.get('alert_count', 0)}",
            f"LATENCY: {int(stats.get('processing_time_ms', 0))}ms"
        ]
        
        metric_start_x = int(self.width * 0.35)
        metric_spacing = int(self.width * 0.15)
        metric_scale = max(0.35, self.width / 2500.0)
        
        for i, m in enumerate(metrics):
            x = metric_start_x + i * metric_spacing
            if x + 150 > self.width:
                break
            cv2.line(frame, (x - 10, int(header_height * 0.25)), 
                     (x - 10, int(header_height * 0.75)), (255, 229, 0), 1)
            cv2.putText(frame, m, (x, int(header_height * 0.6)), 
                        cv2.FONT_HERSHEY_SIMPLEX, metric_scale, (255, 255, 255), 1)
        
        return header_height

    def _draw_scenario_badge(self, frame, header_height):
        """Draw scenario name badge below header."""
        if not self.scenario:
            return
        
        badge_h = 30
        badge_w = len(self.scenario) * 14 + 40
        x = self.width - badge_w - 20
        y = header_height + 10
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + badge_w, y + badge_h), (0, 100, 200), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (x, y), (x + badge_w, y + badge_h), (0, 165, 255), 1)
        cv2.putText(frame, self.scenario, (x + 15, y + 21), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    def _draw_forensic_overlay(self, frame, frame_idx):
        """Timestamp + recording indicator at bottom."""
        # Synthetic timestamp based on frame index (consistent replay)
        base_time = datetime(2026, 4, 15, 14, 30, 0)
        frame_time = base_time.timestamp() + (frame_idx / self.fps)
        ts = datetime.fromtimestamp(frame_time).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        text = f"REC {ts} // {self.sector_id}"
        cv2.putText(frame, text, (20, self.height - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Corner reticle
        margin = 30
        size = 20
        cv2.line(frame, (self.width - margin, margin), (self.width - margin - size, margin), (255, 229, 0), 1)
        cv2.line(frame, (self.width - margin, margin), (self.width - margin, margin + size), (255, 229, 0), 1)
        
        # Bottom-left corner reticle  
        cv2.line(frame, (margin, self.height - margin), (margin + size, self.height - margin), (255, 229, 0), 1)
        cv2.line(frame, (margin, self.height - margin), (margin, self.height - margin - size), (255, 229, 0), 1)

    def run(self):
        print(f"\n{'=' * 60}")
        print("  SENTINEL AI — Demo Pipeline")
        print(f"  Source:   {self.video_path}")
        print(f"  Output:   {self.output_path}")
        print(f"  Scenario: {self.scenario or 'General'}")
        print(f"  Frames:   ~{self.total_frames}")
        print(f"{'=' * 60}\n")
        
        frame_idx = 0
        start_time = time.time()
        
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Lazy init FFmpeg with real frame dimensions
                if self.process is None:
                    self._init_ffmpeg(frame.shape[1], frame.shape[0])
                
                # Resize if necessary to match the output (even dimensions)
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))
                
                timestamp = self.time_origin + frame_idx / self.fps
                result = self.pipeline.process_frame(frame, timestamp)
                annotated = result.annotated_frame
                
                # Draw HUD overlays
                header_h = self._draw_glass_header(annotated, result.to_dict()["stats"])
                self._draw_scenario_badge(annotated, header_h)
                self._draw_forensic_overlay(annotated, frame_idx)
                
                # Write to FFmpeg Pipe
                try:
                    self.process.stdin.write(annotated.tobytes())
                except Exception as e:
                    print(f"FFmpeg pipe write error: {e}")
                    break
                
                for alert in result.alerts:
                    self.alerts_log.append({
                        "id": alert.alert_id,
                        "type": alert.alert_type,
                        "video_time_s": round(frame_idx / self.fps, 2),
                        "timestamp": alert.timestamp,
                        "message": alert.message,
                        "severity": alert.severity,
                        "track_id": alert.track_id,
                        "confidence": alert.confidence,
                        "details": alert.details,
                    })
                
                frame_idx += 1
                
                # Progress reporting
                if frame_idx % 50 == 0:
                    elapsed = time.time() - start_time
                    fps_actual = frame_idx / elapsed if elapsed > 0 else 0
                    if self.total_frames > 0:
                        pct = (frame_idx / self.total_frames) * 100
                        eta = (self.total_frames - frame_idx) / fps_actual if fps_actual > 0 else 0
                        print(f"  Frame {frame_idx}/{self.total_frames} ({pct:.0f}%) | {fps_actual:.1f} fps | ETA: {eta:.0f}s")
                    else:
                        print(f"  Frame {frame_idx} | {fps_actual:.1f} fps")
                    
        finally:
            self.cap.release()
            if self.process and self.process.stdin:
                self.process.stdin.close()
                self.process.wait()
            self.pipeline.clip_recorder.flush()  # finish clips of late alerts
            if self.pipeline.webhook:
                self.pipeline.webhook.close()

            # Save alerts log
            log_path = self.output_path.replace(".mp4", "_alerts.json")
            with open(log_path, "w") as f:
                json.dump(self.alerts_log, f, indent=4, default=str)
            
            elapsed = time.time() - start_time
            print("\n  Demo capture complete.")
            print(f"  Frames: {frame_idx} | Time: {elapsed:.1f}s | Alerts: {len(self.alerts_log)}")
            print(f"  Output: {self.output_path}")
            print(f"  Log:    {log_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sentinel AI — Demo Recording Engine")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--demo", choices=sorted(SAMPLES), help="Bundled sample scenario")
    group.add_argument("--source", help="Path to input video file")
    parser.add_argument("--output", default=None, help="Path for output MP4 (auto-deduced if omitted)")
    parser.add_argument("--config", default=None, help="Path to demo config JSON with zone definitions")
    parser.add_argument("--sector", default="SECTOR-04-NORTH", help="Sector ID for HUD display")
    parser.add_argument("--scenario", default=None, help="Scenario label (e.g. 'FALL DETECTION')")
    args = parser.parse_args()
    if args.demo:
        sample = SAMPLES[args.demo]
        args.source = str(ensure_sample(args.demo))
        args.config = args.config or str(sample.config_path)
        args.scenario = args.scenario or args.demo.upper()
        args.output = args.output or os.path.join("outputs", f"{args.demo}_demo.mp4")

    engine = DemoEngine(
        video_path=args.source,
        output_path=args.output,
        config_path=args.config,
        sector_id=args.sector,
        scenario=args.scenario
    )
    engine.run()
