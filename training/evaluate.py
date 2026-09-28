import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from anomaly.temporal_model import ActionLSTM  # noqa: E402
from train_fall_detector import PoseSequenceDataset, ACTION_LABELS  # noqa: E402

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Evaluating on device: {device}")
    
    # Load data
    dataset = PoseSequenceDataset(args.data_dir, args.seq_length)
    if len(dataset) == 0:
        logger.error("No data found. Aborting.")
        return
        
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Load Model
    model = ActionLSTM(num_classes=len(ACTION_LABELS)).to(device)
    if not os.path.exists(args.model_path):
        logger.error(f"Model file not found at {args.model_path}")
        return
        
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    all_preds = []
    all_targets = []
    
    logger.info("Running inference...")
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.numpy())
            
    # Metrics
    report = classification_report(all_targets, all_preds, target_names=ACTION_LABELS)
    print("\n--- Classification Report ---")
    print(report)
    
    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ACTION_LABELS)
    disp.plot(cmap='Blues', ax=ax, xticks_rotation=45)
    plt.title("Action Classification Confusion Matrix")
    
    save_path = "training/output/confusion_matrix.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    logger.info(f"Saved confusion matrix plot to {save_path}")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Sentinel AI Action Classifier")
    parser.add_argument("--model_path", type=str, default="models/action_lstm.pt")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to pose dataset (.npy files)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_length", type=int, default=30)
    
    args = parser.parse_args()
    evaluate(args)
