import argparse
import logging
import os
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Action labels consistent with TemporalClassifier
ACTION_LABELS = ["walking", "running", "standing", "sitting", "fallen", "fighting", "loitering"]

def extract_windows(sequences, seq_length, overlap):
    """
    sequences: List of ndarray shape (17, 3)
    """
    if len(sequences) < seq_length:
        return []
    
    windows = []
    step = int(seq_length * (1 - overlap))
    step = max(step, 1)
    
    for i in range(0, len(sequences) - seq_length + 1, step):
        window = sequences[i:i+seq_length]
        windows.append(np.array(window))
        
    return windows

def process_videos(input_dir, output_dir, seq_length, overlap, augment=False):
    model = YOLO("yolov8n-pose.pt")
    
    for label in ACTION_LABELS:
        label_dir = os.path.join(input_dir, label)
        if not os.path.exists(label_dir):
            logger.warning(f"Label directory not found: {label_dir}")
            continue
            
        save_dir = os.path.join(output_dir, label)
        os.makedirs(save_dir, exist_ok=True)
        
        video_files = [f for f in os.listdir(label_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        logger.info(f"Processing {len(video_files)} videos for label: {label}")
        
        for video_file in tqdm(video_files, desc=f"Status: {label}"):
            video_path = os.path.join(label_dir, video_file)
            cap = cv2.VideoCapture(video_path)
            video_stem = os.path.splitext(video_file)[0]
            
            # Store sequences per track_id
            track_sequences = defaultdict(list)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Run tracking
                results = model.track(frame, persist=True, verbose=False)
                
                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.id.cpu().numpy().astype(int)
                    keypoints = results[0].keypoints.data.cpu().numpy() # (N, 17, 3)
                    
                    for i, track_id in enumerate(boxes):
                        # Extract keypoints for this track
                        # YOLOv8 keypoints: x, y, conf
                        kpts = keypoints[i] 
                        track_sequences[track_id].append(kpts)
                        
            cap.release()
            
            # Process accumulated sequences into windows
            total_windows = 0
            for track_id, sequences in track_sequences.items():
                windows = extract_windows(sequences, seq_length, overlap)
                
                for idx, window in enumerate(windows):
                    # Save normal window
                    file_name = f"{video_stem}_t{track_id}_w{idx:03d}.npy"
                    np.save(os.path.join(save_dir, file_name), window)
                    total_windows += 1
                    
                    # Augmentation: Horizontal Flip
                    if augment:
                        flipped_window = window.copy()
                        # Flip X coordinates (column 0)
                        # We don't have frame width here, but ActionLSTM normalizes to bbox anyway.
                        # So just negative X works if we normalize correctly in training.
                        # Actually, better to just flip the order of symmetric joints if needed,
                        # but simple X flip is fine for LSTM as long as relative position holds.
                        flipped_window[:, :, 0] *= -1 
                        aug_file_name = f"{video_stem}_t{track_id}_w{idx:03d}_flip.npy"
                        np.save(os.path.join(save_dir, aug_file_name), flipped_window)
                        total_windows += 1
            
            logger.info(f"Finished {video_file}: {total_windows} windows extracted.")

def main():
    parser = argparse.ArgumentParser(description="Sentinel AI Data Preparation")
    parser.add_argument("--input_dir", type=str, default="data/raw_videos", help="Directory with labeled video folders")
    parser.add_argument("--output_dir", type=str, default="data/poses",
                        help="Directory to save extracted .npy sequences")
    parser.add_argument("--seq_length", type=int, default=30, help="Number of frames per sequence")
    parser.add_argument("--overlap", type=float, default=0.5, help="Overlap between sliding windows (0-1)")
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation (Horizontal Flip)")
    
    args = parser.parse_args()
    
    process_videos(args.input_dir, args.output_dir, args.seq_length, args.overlap, args.augment)
    
    # Print Summary
    print("\n--- Summary ---")
    for label in ACTION_LABELS:
        path = os.path.join(args.output_dir, label)
        if os.path.exists(path):
            count = len([f for f in os.listdir(path) if f.endswith(".npy")])
            print(f"{label.capitalize()}: {count} samples")

if __name__ == "__main__":
    main()
