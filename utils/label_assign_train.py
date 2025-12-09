import json
import os
"""

This code assigns labels to each of the images preesent in the .txt file and then stores them in a json file


used for training set 

"""
# --- CONFIGURATION ---
# 1. Your list of images (the .txt file you used earlier)
list_file_path = '/home/s124z/Code/YOLO/heico_dataset/training_images.txt'

# 2. The master JSON file containing ALL labels
master_json_path = '/home/s124z/E130-Projekte/SAVEOR_SST/Heico/labels/2dbb/heico_2dbb_video.json'  # Rename this to your actual json filename

# 3. The output file name for the filtered labels
output_json_path = '/home/s124z/Code/YOLO/heico_dataset/train_labels.json'
# ---------------------

def filter_json_labels():
    print("--- Starting Label Extraction ---")

    # Step 1: Read the list of target filenames
    print(f"Reading file list from: {list_file_path}")
    try:
        with open(list_file_path, 'r') as f:
            # Use a set for faster lookups
            target_files = set(line.strip() for line in f if line.strip())
        print(f"-> Target images found in list: {len(target_files)}")
    except FileNotFoundError:
        print("Error: Could not find the text file.")
        return

    # Step 2: Load the Master JSON
    print(f"Loading master JSON from: {master_json_path} ... (this might take a few seconds)")
    try:
        with open(master_json_path, 'r') as jf:
            master_data = json.load(jf)
    except FileNotFoundError:
        print("Error: Could not find the master JSON file.")
        return
    except json.JSONDecodeError:
        print("Error: The master JSON file is not valid JSON.")
        return

    # Step 3: Filter the Data
    print("Filtering data...")
    filtered_data = {}
    found_count = 0
    missing_count = 0
    
    # We iterate through the target list to ensure we only get what we want
    for filename in target_files:
        if filename in master_data:
            filtered_data[filename] = master_data[filename]
            found_count += 1
        else:
            # This happens if an image is in your text file but has no labels in the JSON
            missing_count += 1
            # Optional: Print missing files
            # print(f"Warning: No labels found for {filename}")

    # Step 4: Save the new JSON
    print(f"Saving filtered labels to: {output_json_path}")
    with open(output_json_path, 'w') as out_f:
        # indent=2 makes the file human-readable (pretty printed)
        json.dump(filtered_data, out_f, indent=2)

    # Step 5: Summary
    print("-" * 30)
    print("EXTRACTION COMPLETE")
    print(f"Images in list:      {len(target_files)}")
    print(f"Labels found/copied: {found_count}")
    print(f"Labels missing:      {missing_count}")
    print("-" * 30)

if __name__ == "__main__":
    filter_json_labels()