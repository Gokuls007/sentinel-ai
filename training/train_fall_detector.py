import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import logging

# Add project root to path to import backend components
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))
from anomaly.temporal_model import ActionLSTM

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ACTION_LABELS = ["walking", "running", "standing", "sitting", "fallen", "fighting", "loitering"]

class PoseSequenceDataset(Dataset):
    def __init__(self, data_dir, seq_length=30):
        self.samples = []
        self.labels = []
        self.seq_length = seq_length
        self.label_to_idx = {label: i for i, label in enumerate(ACTION_LABELS)}
        
        logger.info(f"Loading dataset from {data_dir}...")
        
        for label in ACTION_LABELS:
            label_dir = os.path.join(data_dir, label)
            if not os.path.exists(label_dir):
                continue
                
            idx = self.label_to_idx[label]
            files = [f for f in os.listdir(label_dir) if f.endswith(".npy")]
            
            for f in files:
                # Load sequence (seq_len, 17, 3)
                seq = np.load(os.path.join(label_dir, f))
                # Take only x, y (drop confidence)
                seq = seq[:, :, :2]
                
                # Normalize using the same logic as production
                norm_seq = self._normalize_sequence(seq)
                
                self.samples.append(torch.FloatTensor(norm_seq))
                self.labels.append(idx)
        
        logger.info(f"Dataset loaded. Total samples: {len(self.samples)}")
        for label, idx in self.label_to_idx.items():
            count = self.labels.count(idx)
            logger.info(f"  {label}: {count}")

    def _normalize_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """
        sequence: (seq_len, 17, 2)
        """
        seq_len = sequence.shape[0]
        pts = sequence.reshape(seq_len, 17, 2)
        
        # Filter zero points (invalid keypoints)
        valid_pts = pts[pts.sum(axis=2) != 0]
        
        if len(valid_pts) == 0:
            return sequence.reshape(seq_len, 34)
            
        # Get sequence-wide bbox
        min_x, min_y = np.min(valid_pts, axis=0)
        max_x, max_y = np.max(valid_pts, axis=0)
        
        w = max_x - min_x
        h = max_y - min_y
        scale = max(w, h, 1e-6)
        
        norm_pts = np.zeros_like(pts)
        for i in range(seq_len):
            for j in range(17):
                if pts[i, j, 0] != 0 or pts[i, j, 1] != 0:
                    norm_pts[i, j, 0] = (pts[i, j, 0] - min_x) / scale
                    norm_pts[i, j, 1] = (pts[i, j, 1] - min_y) / scale
                    
        return norm_pts.reshape(seq_len, 34)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")
    
    # Load data
    full_dataset = PoseSequenceDataset(args.data_dir, args.seq_length)
    if len(full_dataset) == 0:
        logger.error("No data found in directory. Aborting.")
        return
        
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Init Model
    model = ActionLSTM(num_classes=len(ACTION_LABELS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True) if os.path.dirname(args.save_path) else None

    logger.info("Starting training loop...")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(targets).sum().item()
            
        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(targets).sum().item()
                
        # Epoch metrics
        train_loss /= len(train_dataset)
        train_acc = train_correct / len(train_dataset)
        val_loss /= len(val_dataset)
        val_acc = val_correct / len(val_dataset)
        
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch+1}/{args.epochs} | "
                    f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | LR: {current_lr:.6f}")
        
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.save_path)
            logger.info(f"  --> Saved best model to {args.save_path}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= args.patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Sentinel AI Action Classifier")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to pose dataset (.npy files)")
    parser.add_argument("--save_path", type=str, default="models/action_lstm.pt", help="Path to save best model")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seq_length", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    
    args = parser.parse_args()
    train(args)
