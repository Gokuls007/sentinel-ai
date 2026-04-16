"""
Sentinel AI — Post-Production Pipeline
Creates title cards, normalizes segments, stitches demo reel, and extracts GIF.

Usage:
    python scripts/post_production.py
"""
import cv2
import numpy as np
import os
import sys
import subprocess
import imageio_ffmpeg

# Reel parameters
TARGET_W, TARGET_H = 1280, 720
TARGET_FPS = 25
SEGMENT_DURATION = 18  # seconds per demo clip (trimmed)


def create_title_card(text, subtitle, filename, duration=1.5, fps=TARGET_FPS, 
                      size=(TARGET_W, TARGET_H)):
    """Create a cinematic title card with fade effect and surveillance aesthetic."""
    num_frames = int(duration * fps)
    
    # Use FFmpeg pipe for reliable encoding
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe, '-y',
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{size[0]}x{size[1]}',
        '-pix_fmt', 'bgr24', '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-preset', 'ultrafast', '-crf', '18',
        filename
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    
    for i in range(num_frames):
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        
        # Fade factor (fade in for first 30%, hold, fade out for last 20%)
        progress = i / num_frames
        if progress < 0.3:
            alpha = progress / 0.3
        elif progress > 0.8:
            alpha = (1.0 - progress) / 0.2
        else:
            alpha = 1.0
        
        # Grid background (subtle surveillance grid)
        grid_color = int(12 * alpha)
        for gx in range(0, size[0], 40):
            cv2.line(frame, (gx, 0), (gx, size[1]), (grid_color, grid_color + 2, grid_color), 1)
        for gy in range(0, size[1], 40):
            cv2.line(frame, (0, gy), (size[0], gy), (grid_color, grid_color + 2, grid_color), 1)
        
        # Horizontal accent lines
        line_color = (0, int(229 * alpha), int(255 * alpha))
        y_center = size[1] // 2
        cv2.line(frame, (100, y_center - 55), (size[0] - 100, y_center - 55), line_color, 2)
        cv2.line(frame, (100, y_center + 55), (size[0] - 100, y_center + 55), line_color, 2)
        
        # Corner brackets
        bracket_size = 30
        for cx, cy in [(80, y_center - 70), (size[0] - 80, y_center - 70),
                       (80, y_center + 70), (size[0] - 80, y_center + 70)]:
            dx = bracket_size if cx < size[0] // 2 else -bracket_size
            dy = bracket_size if cy < y_center else -bracket_size
            cv2.line(frame, (cx, cy), (cx + dx, cy), line_color, 2)
            cv2.line(frame, (cx, cy), (cx, cy + dy), line_color, 2)
        
        # Main title text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale = 1.3
        text_size = cv2.getTextSize(text, font, text_scale, 3)[0]
        text_x = (size[0] - text_size[0]) // 2
        text_y = y_center + text_size[1] // 2 - 5
        text_color = (int(255 * alpha), int(255 * alpha), int(255 * alpha))
        cv2.putText(frame, text, (text_x, text_y), font, text_scale, text_color, 3)
        
        # Subtitle
        sub_scale = 0.5
        sub_size = cv2.getTextSize(subtitle, font, sub_scale, 1)[0]
        sub_color = (0, int(229 * alpha), int(255 * alpha))
        cv2.putText(frame, subtitle, 
                    ((size[0] - sub_size[0]) // 2, text_y + 35), 
                    font, sub_scale, sub_color, 1)
        
        proc.stdin.write(frame.tobytes())
    
    proc.stdin.close()
    proc.wait()
    print(f"  Title card created: {filename}")


def normalize_segment(input_file, output_file, max_duration=SEGMENT_DURATION):
    """Re-encode a segment to common format (H.264, 1280x720, 25fps) and trim."""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe, '-y',
        '-i', input_file,
        '-t', str(max_duration),
        '-vf', f'scale={TARGET_W}:{TARGET_H}:force_original_aspect_ratio=decrease,pad={TARGET_W}:{TARGET_H}:(ow-iw)/2:(oh-ih)/2',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-preset', 'fast', '-crf', '20',
        '-r', str(TARGET_FPS),
        '-an',  # No audio
        output_file
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f"  WARNING: Normalize failed for {input_file}: {result.stderr.decode()[-200:]}")
        return False
    print(f"  Normalized: {input_file} -> {output_file}")
    return True


def stitch_videos(segments, output_file):
    """Concatenate video segments using FFmpeg concat demuxer."""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    
    # Filter out non-existent files
    valid = [s for s in segments if os.path.exists(s)]
    if len(valid) < len(segments):
        missing = set(segments) - set(valid)
        print(f"  WARNING: Missing segments: {missing}")
    
    if not valid:
        print("  ERROR: No valid segments to stitch!")
        return False
    
    # Create concat list
    list_file = os.path.join(os.path.dirname(output_file) or ".", "_concat_list.txt")
    with open(list_file, "w") as f:
        for seg in valid:
            safe_path = os.path.abspath(seg).replace('\\', '/')
            f.write(f"file '{safe_path}'\n")
    
    cmd = [
        ffmpeg_exe, "-y", "-f", "concat", "-safe", "0",
        "-i", list_file, "-c", "copy", output_file
    ]
    print(f"  Stitching {len(valid)} segments -> {output_file}")
    result = subprocess.run(cmd, capture_output=True)
    
    try:
        os.remove(list_file)
    except OSError:
        pass
    
    if result.returncode != 0:
        print(f"  ERROR: Stitch failed: {result.stderr.decode()[-200:]}")
        return False
    
    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"  Demo reel created: {output_file} ({size_mb:.1f} MB)")
    return True


def extract_gif(input_file, output_file, start_time=0, duration=5, width=800, fps=12):
    """Extract a high-quality GIF using palette-based encoding."""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    
    # Two-pass palette-based GIF for better quality
    palette_file = output_file.replace(".gif", "_palette.png")
    
    # Pass 1: Generate palette
    cmd1 = [
        ffmpeg_exe, '-y',
        '-ss', str(start_time), '-t', str(duration),
        '-i', input_file,
        '-vf', f'fps={fps},scale={width}:-1:flags=lanczos,palettegen',
        palette_file
    ]
    subprocess.run(cmd1, capture_output=True)
    
    if os.path.exists(palette_file):
        # Pass 2: Use palette for encoding
        cmd2 = [
            ffmpeg_exe, '-y',
            '-ss', str(start_time), '-t', str(duration),
            '-i', input_file,
            '-i', palette_file,
            '-lavfi', f'fps={fps},scale={width}:-1:flags=lanczos [x]; [x][1:v] paletteuse',
            output_file
        ]
        subprocess.run(cmd2, capture_output=True)
        try:
            os.remove(palette_file)
        except OSError:
            pass
    else:
        # Fallback: single-pass
        cmd = [
            ffmpeg_exe, '-y',
            '-ss', str(start_time), '-t', str(duration),
            '-i', input_file,
            '-vf', f'fps={fps},scale={width}:-1:flags=lanczos',
            output_file
        ]
        subprocess.run(cmd, capture_output=True)
    
    if os.path.exists(output_file):
        size_kb = os.path.getsize(output_file) / 1024
        print(f"  GIF extracted: {output_file} ({size_kb:.0f} KB)")
    else:
        print(f"  WARNING: GIF extraction failed")


if __name__ == "__main__":
    print("=" * 60)
    print("  SENTINEL AI — Post-Production Pipeline")
    print("=" * 60)
    
    os.makedirs("outputs/temp", exist_ok=True)
    os.makedirs("assets", exist_ok=True)
    
    # 1. Create Title Cards
    print("\n[1/4] Creating title cards...")
    create_title_card("PERSISTENT TRACKING", "SENTINEL_AI // MULTI-TARGET_IDENTIFICATION", "outputs/temp/t1.mp4")
    create_title_card("FALL DETECTION", "SENTINEL_AI // ANOMALY_ENGINE_ACTIVE", "outputs/temp/t2.mp4")
    create_title_card("ZONE MONITORING", "SENTINEL_AI // PERIMETER_ENFORCEMENT", "outputs/temp/t3.mp4")
    create_title_card("INTRUSION DETECTION", "SENTINEL_AI // SECTOR_BREACH_PROTOCOL", "outputs/temp/t4.mp4")
    
    # 2. Normalize demo segments
    print("\n[2/4] Normalizing segments...")
    demo_segments = {
        "walking": ("outputs/walking_demo.mp4", "outputs/temp/n_walking.mp4"),
        "fall": ("outputs/fall_demo.mp4", "outputs/temp/n_fall.mp4"),
        "multi": ("outputs/multi_person_demo.mp4", "outputs/temp/n_multi.mp4"),
        "corridor": ("outputs/corridor_demo.mp4", "outputs/temp/n_corridor.mp4"),
    }
    
    for name, (src, dst) in demo_segments.items():
        if os.path.exists(src):
            normalize_segment(src, dst)
        else:
            print(f"  SKIP: {src} not found")
    
    # 3. Stitch reel
    print("\n[3/4] Stitching demo reel...")
    sequence = [
        "outputs/temp/t1.mp4",
        "outputs/temp/n_walking.mp4",
        "outputs/temp/t2.mp4",
        "outputs/temp/n_fall.mp4",
        "outputs/temp/t3.mp4",
        "outputs/temp/n_multi.mp4",
        "outputs/temp/t4.mp4",
        "outputs/temp/n_corridor.mp4",
    ]
    stitch_videos(sequence, "assets/demo_reel.mp4")
    
    # 4. Extract GIF from fall detection
    print("\n[4/4] Extracting demo GIF...")
    if os.path.exists("outputs/fall_demo.mp4"):
        extract_gif("outputs/fall_demo.mp4", "assets/demo.gif", start_time=0, duration=5, width=800, fps=12)
    elif os.path.exists("assets/demo_reel.mp4"):
        extract_gif("assets/demo_reel.mp4", "assets/demo.gif", start_time=15, duration=5, width=800, fps=12)
    
    print(f"\n{'=' * 60}")
    print("  Post-production complete!")
    if os.path.exists("assets/demo_reel.mp4"):
        size = os.path.getsize("assets/demo_reel.mp4") / (1024 * 1024)
        print(f"  Demo Reel: assets/demo_reel.mp4 ({size:.1f} MB)")
    if os.path.exists("assets/demo.gif"):
        size = os.path.getsize("assets/demo.gif") / 1024
        print(f"  Demo GIF:  assets/demo.gif ({size:.0f} KB)")
    print(f"{'=' * 60}")
