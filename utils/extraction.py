import os
import shutil
from pathlib import Path

# --- CONFIGURATION ---
# 1. PATH TO YOUR LIST OF FILES (The .txt file you uploaded)
list_file_path = '/home/s124z/Code/YOLO/heico_dataset/training_images.txt'

# 2. PATH TO YOUR SOURCE ROOT FOLDER (The "FO_frames" folder shown in your screenshot)
# Update this to the actual path on your computer
source_root_folder = '/home/s124z/E130-Projekte/Saliq_datasets/Heico/FO_frames' 

# 3. DESTINATION FOLDER
destination_folder = '/home/s124z/Code/YOLO/heico_dataset/train'
# ---------------------

def collect_images():
    # Step 1: Create destination folder
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        print(f"Created folder: {destination_folder}")

    # Step 2: Index the source folder
    # We walk through FO_frames and remember where every single file is located.
    print(f"Scanning '{source_root_folder}' for images... this might take a moment.")
    file_index = {}
    
    for root, dirs, files in os.walk(source_root_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Save the full path so we can find 'Prokto_3_clip_0431.jpg' instantly
                file_index[file] = os.path.join(root, file)
    
    print(f"Indexing complete. Found {len(file_index)} total images in source.")

    # Step 3: Read the target list and copy files
    print(f"Reading target list from {list_file_path}...")
    
    try:
        with open(list_file_path, 'r') as f:
            target_files = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print("Error: Could not find your text file. Make sure 'train_list.txt' is in this folder.")
        return

    copied_count = 0
    missing_count = 0
    missing_files = []

    print("Starting copy process...")
    
    for filename in target_files:
        # Check if we found this file during our scan
        if filename in file_index:
            src_path = file_index[filename]
            
            # --- OPTIONAL: ORGANIZE BY CLASS ---
            # If you want subfolders like training/0, training/1, uncomment the next 3 lines:
            # class_id = filename.rsplit('.', 1)[0].split('_')[-1] # Gets the last number
            # dst_dir = os.path.join(destination_folder, class_id)
            # os.makedirs(dst_dir, exist_ok=True)
            # dst_path = os.path.join(dst_dir, filename)
            
            # Default: Flat copy (all in one folder)
            dst_path = os.path.join(destination_folder, filename)
            
            try:
                shutil.copy2(src_path, dst_path)
                copied_count += 1
            except Exception as e:
                print(f"Error copying {filename}: {e}")
        else:
            missing_count += 1
            missing_files.append(filename)

    # Step 4: Final Report
    print("-" * 40)
    print(f"DONE.")
    print(f"Successfully copied: {copied_count} images")
    print(f"Missing images:      {missing_count}")
    
    if missing_files:
        print("\nCould not find these files (check spelling or if they are in FO_frames):")
        for m in missing_files[:10]:
            print(f" - {m}")
        if len(missing_files) > 10:
            print(f"... and {len(missing_files) - 10} more.")

if __name__ == "__main__":
    collect_images()