import json

# --- CONFIGURATION ---
list_file_path = '/home/s124z/Code/YOLO/heico_dataset/training_images.txt'

# 2. The master JSON file containing ALL labels
master_json_path = '/home/s124z/E130-Projekte/SAVEOR_SST/Heico/labels/2dbb/heico_2dbb_video.json'  # Rename this to your actual json filename

# 3. The output file name for the filtered labels
output_report = '/home/s124z/Code/YOLO/heico_dataset/missing.json'
# ---------------------
# ---------------------

def find_missing_files():
    # 1. Load the list of filenames you WANT
    try:
        with open(list_file_path, 'r') as f:
            target_files = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print("Error: Could not find the text file.")
        return

    # 2. Load the JSON keys (we only need keys, not the full data)
    print("Loading JSON keys...")
    try:
        with open(master_json_path, 'r') as jf:
            master_data = json.load(jf)
            json_keys = set(master_data.keys()) # Convert to set for fast lookup
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return

    # 3. Compare
    missing_files = []
    for filename in target_files:
        if filename not in json_keys:
            missing_files.append(filename)

    # 4. Output Results
    print(f"Found {len(missing_files)} missing labels.")
    
    if len(missing_files) > 0:
        print(f"Saving list to {output_report}...")
        with open(output_report, 'w') as f:
            f.write("The following images are in your text file but have NO entry in the JSON:\n")
            f.write("========================================================================\n")
            for name in missing_files:
                f.write(f"{name}\n")
                # Also print to console if you want to see them immediately
                print(f"MISSING: {name}")
    else:
        print("Great! No labels are missing.")

if __name__ == "__main__":
    find_missing_files()