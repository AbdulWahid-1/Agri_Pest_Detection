# YOLOv9 automatically generates complex evaluation metrics during training.
# Since I stopped training at Epoch 26 via Ctrl+C, the final combined charts might not have generated.
# However, YOLO saves the raw CSV data. I updated this script to read the raw results.csv 
# and automatically draw my own custom evaluation graph for my GitHub portfolio.

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def generate_evaluation_metrics():
    # Smart search to find where YOLO hid the results.csv file
    possible_dirs = [
        Path("runs/detect/output/pest_model"),
        Path("runs/detect/pest_model"),
        Path("output/pest_model"),
        Path("runs/detect/train")
    ]
    
    csv_file = None
    for d in possible_dirs:
        if d.exists() and (d / "results.csv").exists():
            csv_file = d / "results.csv"
            break
            
    if not csv_file:
        print("[-] Cannot find the results.csv file from training.")
        return
        
    print(f"[+] Found raw training data at: {csv_file}")
    print("[+] Generating custom evaluation graphs...")
    
    # Read the CSV. YOLO's CSV headers usually have leading spaces, so we strip them.
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip()
    
    # The columns we care about
    # Usually they are: 'epoch', 'train/box_loss', 'metrics/mAP50(B)'
    epoch = df['epoch']
    
    # Let's dynamically find the right column names in case Ultralytics changed them
    map50_col = next((col for col in df.columns if 'mAP50' in col and '95' not in col), None)
    train_loss_col = next((col for col in df.columns if 'train/box_loss' in col), None)
    val_loss_col = next((col for col in df.columns if 'val/box_loss' in col), None)
    
    if not map50_col:
        print("[-] Could not find mAP metrics in the CSV.")
        return

    # Create the output directory
    dest_dir = Path("output/final_metrics")
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Draw the graphs
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Graph 1: Mean Average Precision (mAP50)
    axes[0].plot(epoch, df[map50_col], marker='o', color='b', label='mAP50', linewidth=2)
    axes[0].set_title('AI Accuracy Over Time (mAP50)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy (0.0 to 1.0)')
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].legend()
    
    # Graph 2: Training vs Validation Box Loss
    if train_loss_col and val_loss_col:
        axes[1].plot(epoch, df[train_loss_col], marker='s', color='r', label='Train Box Loss', linewidth=2)
        axes[1].plot(epoch, df[val_loss_col], marker='^', color='g', label='Validation Box Loss', linewidth=2)
        axes[1].set_title('Learning Efficiency (Box Loss)')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss (Lower is better)')
        axes[1].grid(True, linestyle='--', alpha=0.7)
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, 'Loss Data Unavailable', ha='center', va='center')
        
    plt.tight_layout()
    out_path = dest_dir / "custom_evaluation_graph.png"
    plt.savefig(out_path, dpi=300)
    
    print(f"=== MY AI'S REPORT CARD (AT EPOCH {int(epoch.iloc[-1])}) ===")
    print(f"Final Accuracy (mAP50): {df[map50_col].iloc[-1]:.4f}")
    print(f"[+] Successfully generated and saved evaluation graph to: {out_path}")

if __name__ == "__main__":
    generate_evaluation_metrics()