# I updated this script to fix the cuDNN crashes on my Windows laptop.
# It lowers the worker count to prevent CPU bottlenecks and automatically resumes from my last saved checkpoint.

import torch
from ultralytics import YOLO
from pathlib import Path

def train_pest_detector():
    # 1. Fix for the cuDNN Internal Error on Windows laptops
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    yaml_path = Path(__file__).resolve().parent / "dataset.yaml"
    
    # 2. Look for the backup file to save my 8 hours of work
    last_weights = Path("runs/detect/output/pest_model/weights/last.pt")
    # Note: Sometimes YOLO saves to the root output folder instead
    alt_weights = Path("output/pest_model/weights/last.pt")

    if last_weights.exists():
        print(f"[+] Found backup weights at {last_weights}!")
        print("[+] Resuming training from previous Epoch to save time...")
        model = YOLO(str(last_weights))
        
        # Resume training, but force the workers down to 2 to stop the crash
        model.train(resume=True, workers=2)
        
    elif alt_weights.exists():
        print(f"[+] Found backup weights at {alt_weights}!")
        print("[+] Resuming training from Epoch 11 to save time...")
        model = YOLO(str(alt_weights))
        model.train(resume=True, workers=2)
        
    else:
        print("[+] No backup found. Starting fresh training...")
        if not yaml_path.exists():
            print("[-] Error: dataset.yaml not found.")
            return

        model = YOLO('yolov9c.pt') 
        
        model.train(
            data=str(yaml_path),
            epochs=26,
            imgsz=640,
            batch=8,
            device=0,
            amp=True,
            workers=2, # CRITICAL FIX: Dropped from 8 to 2 to stop the "Slow image access" crash
            project='output',
            name='pest_model',
            exist_ok=True
        )
    
    print("[+] Training complete. Ready for inference.")

if __name__ == "__main__":
    train_pest_detector()