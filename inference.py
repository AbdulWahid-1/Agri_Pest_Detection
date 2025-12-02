# Built this script to handle the actual scanning. 
# It takes the YOLO model, feeds it either a single image or a full directory,
# draws the bounding boxes, maps the real biological names, and dumps a CSV report.

import os
import cv2
import pandas as pd
from ultralytics import YOLO
from pathlib import Path

def run_scanner(model_path, source_path, output_folder):
    # Quick sanity check before we start loading heavy models into VRAM
    if not model_path or not Path(model_path).exists():
        print("Error: Couldn't find the weights file.")
        print("Make sure training actually completed at least one epoch.")
        return
        
    print(f"Loading weights from: {model_path}")
    model = YOLO(model_path)
    
    # The dataset outputs classes as integers (0 to 101). 
    # We need to map these back to the actual insect names using classes.txt.
    class_file = Path("classes.txt")
    if class_file.exists():
        with open(class_file, 'r', encoding='utf-8') as f:
            real_names = {}
            for line in f:
                if line.strip():
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) == 2:
                        # The IP102 dataset txt file starts at 1, but YOLO starts counting at 0.
                        class_id = int(parts[0]) - 1 
                        real_names[class_id] = parts[1]
            
            # Ultralytics made the .names property read-only recently. 
            # We have to bypass it by injecting directly into the base PyTorch model.
            model.model.names = real_names
            print("Loaded real biological pest names.")
    else:
        print("Warning: classes.txt not found. Defaulting to standard YOLO integer labels.")
    
    # Make sure the output folder actually exists so it doesn't crash later
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Figure out if the user passed a single file or a whole directory
    source = Path(source_path)
    image_paths = []
    
    if source.is_file():
        image_paths = [source]
    elif source.is_dir():
        # Grab all common image formats just in case
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"]:
            image_paths.extend(source.glob(ext))
            
    if not image_paths:
        print(f"Didn't find any images in {source_path}. Exiting.")
        return

    print(f"Starting scan on {len(image_paths)} image(s)...")
    report_data = []
    
    # Variables to track unique insects so our README doesn't just show 8 of the same bug.
    showcase_count = 1
    max_showcase = 8
    seen_species = set()

    for img_path in image_paths:
        # Run the actual prediction. conf=0.4 filters out weak guesses.
        results = model(str(img_path), conf=0.4, verbose=False)[0]
        pest_count = len(results.boxes)
        
        # Save the standard output image with the bounding boxes drawn
        save_path = Path(output_folder) / img_path.name
        results.save(filename=str(save_path))
        
        # Extract the actual names of whatever we found in this image
        detected_names = []
        has_new_species = False
        
        if pest_count > 0:
            for box in results.boxes:
                class_id = int(box.cls[0].item())
                pest_name = model.model.names.get(class_id, f"pest_{class_id}")
                detected_names.append(pest_name)
                
                # Check if this is a species we haven't showcased yet
                if pest_name not in seen_species:
                    has_new_species = True
                    seen_species.add(pest_name)
        
        # If we found a bug we haven't seen before, save a copy for the README
        if has_new_species and showcase_count <= max_showcase:
            readme_path = Path(output_folder) / f"readme_example_{showcase_count}.jpg"
            results.save(filename=str(readme_path))
            showcase_count += 1
        
        # Basic logic to grade the infestation level
        if pest_count == 0:
            severity = "Clean"
        elif pest_count <= 3:
            severity = "Low"
        elif pest_count <= 7:
            severity = "Moderate"
        else:
            severity = "Severe"
            
        # Flatten the list of names into a clean string, removing duplicates
        pest_types_str = ", ".join(list(set(detected_names))) if detected_names else "None"
            
        report_data.append({
            "Image_Name": img_path.name,
            "Total_Pests_Detected": pest_count,
            "Detected_Species": pest_types_str,
            "Infestation_Level": severity
        })
        
    # Dump everything into a CSV so the data is actually usable 
    df = pd.DataFrame(report_data)
    df.to_csv(Path(output_folder) / "infestation_report.csv", index=False)
    print(f"Scan complete. Found {len(seen_species)} unique species.")
    print(f"Exported {showcase_count - 1} unique examples for the README showcase.")

def find_my_weights():
    # YOLO sometimes saves in runs/detect/train, sometimes in just output/
    # depending on if we hit a crash or stopped it early. This checks all common paths.
    possible_paths = [
        Path("runs/detect/output/pest_model/weights/best.pt"),
        Path("runs/detect/pest_model/weights/best.pt"),
        Path("output/pest_model/weights/best.pt")
    ]
    for path in possible_paths:
        if path.exists(): 
            return str(path)
    return None

if __name__ == "__main__":
    my_weights = find_my_weights()
    results_dir = "output/scanner_results"
    
    # Just point this to the validation folder to run the mass test
    test_source = "ip102-yolov5/images/val" 
    
    run_scanner(my_weights, test_source, results_dir)