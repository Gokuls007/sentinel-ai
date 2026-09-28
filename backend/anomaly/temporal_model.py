import torch
import torch.nn as nn
import numpy as np
import os
import logging
from typing import Optional, Tuple

logger = logging.getLogger("sentinel.anomaly.temporal")

class ActionLSTM(nn.Module):
    def __init__(self, input_size=34, hidden_size=128, num_layers=2, num_classes=7):
        super(ActionLSTM, self).__init__()
        self.layer_norm = nn.LayerNorm(input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.dropout_lstm = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout_fc = nn.Dropout(0.15)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, 34)
        x = self.layer_norm(x)
        out, _ = self.lstm(x)
        # Take the last hidden state
        out = out[:, -1, :]
        out = self.dropout_lstm(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout_fc(out)
        out = self.fc2(out)
        return out

class TemporalClassifier:
    ACTION_LABELS = ["walking", "running", "standing", "sitting", "fallen", "fighting", "loitering"]
    ANOMALY_ACTIONS = {"fallen", "fighting", "loitering"}

    def __init__(self, model_path: str = None, hidden_size=128, num_layers=2, 
                 device="auto", min_sequence_length=15):
        
        self.min_seq_len = min_sequence_length
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.model = ActionLSTM(hidden_size=hidden_size, num_layers=num_layers).to(self.device)
        # Only a model with trained weights may raise alerts: a randomly initialised
        # LSTM would emit meaningless "fighting"/"fallen" alerts.
        self.loaded = False

        if model_path and os.path.exists(model_path):
            try:
                state = torch.load(model_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state)
                self.loaded = True
                logger.info(f"Loaded temporal model from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load temporal model: {e}")
        else:
            logger.info("No trained temporal model at %s; behaviour classification disabled "
                        "(rule-based fall/zone/loitering detection still runs)", model_path)

        self.model.eval()

    def predict(self, pose_sequence: np.ndarray) -> Optional[Tuple[str, float, np.ndarray]]:
        """
        pose_sequence: (seq_len, 34)
        Returns: (label, confidence, all_probs)
        """
        if len(pose_sequence) < self.min_seq_len:
            return None
            
        # 1. Normalize
        norm_sequence = self._normalize_sequence(pose_sequence)
        
        # 2. Convert to tensor
        x = torch.FloatTensor(norm_sequence).unsqueeze(0).to(self.device)
        
        # 3. Predict
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
        idx = np.argmax(probs)
        label = self.ACTION_LABELS[idx]
        confidence = float(probs[idx])
        
        return label, confidence, probs

    def is_anomaly(self, pose_sequence: np.ndarray, threshold=0.7) -> Optional[Tuple[str, float]]:
        if not self.loaded:
            return None
        res = self.predict(pose_sequence)
        if res:
            label, conf, _ = res
            if label in self.ANOMALY_ACTIONS and conf >= threshold:
                return label, conf
        return None

    def _normalize_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """
        sequence: (seq_len, 34)
        Scales keypoints to [0,1] based on the sequence-wide bounding box.
        """
        seq_len = sequence.shape[0]
        pts = sequence.reshape(seq_len, 17, 2)
        
        # Filter zero points (invalid keypoints)
        valid_pts = pts[pts.sum(axis=2) != 0]
        
        if len(valid_pts) == 0:
            return sequence
            
        # Get sequence-wide bbox
        min_x, min_y = np.min(valid_pts, axis=0)
        max_x, max_y = np.max(valid_pts, axis=0)
        
        w = max_x - min_x
        h = max_y - min_y
        
        # Avoid division by zero
        scale = max(w, h, 1e-6)
        
        norm_pts = np.zeros_like(pts)
        for i in range(seq_len):
            for j in range(17):
                if pts[i, j, 0] != 0 or pts[i, j, 1] != 0:
                    norm_pts[i, j, 0] = (pts[i, j, 0] - min_x) / scale
                    norm_pts[i, j, 1] = (pts[i, j, 1] - min_y) / scale
                    
        return norm_pts.reshape(seq_len, 34)
