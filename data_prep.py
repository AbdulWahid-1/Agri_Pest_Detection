# data_prep.py
# I wrote this script to dynamically generate the YOLOv9 configuration file.
# This prevents pathing errors by using absolute system paths.

import os
import yaml
from pathlib import Path

def create_yolo_config():
    base_dir = Path(__file__).resolve().parent
    # Updated to match the new leonidkulyk dataset folder
    dataset_dir = base_dir / "ip102-yolov5" 
    
    if not dataset_dir.exists():
        print(f"[-] Error: I cannot find the dataset folder at {dataset_dir}")
        print("Please ensure you downloaded the Kaggle dataset into the ip102-yolov5 folder.")
        return

    data_config = {
        'path': str(dataset_dir),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 102, 
        'names': [f'pest_{i}' for i in range(102)] 
    }

    yaml_path = base_dir / "dataset.yaml"
    
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)

    print(f"[+] Successfully generated YOLO configuration file at: {yaml_path}")
    print("[+] The dataset is ready for training.")

if __name__ == "__main__":
    create_yolo_config()