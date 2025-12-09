import os
import shutil

def consolidate_files(source_dir, destination_dir, extensions=None):
    """
    Moves files with specified extensions from all subdirectories 
    of source_dir into the destination_dir.
    """
    
    # 1. Ensure the destination directory exists
    os.makedirs(destination_dir, exist_ok=True)
    
    # Optional: Define common image extensions
    if extensions is None:
        extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
        print(f"Searching for file types: {', '.join(extensions)}")

    # Keep track of file moves and duplicates
    files_moved = 0
    duplicates_skipped = 0
    
    # 2. Walk through the source directory and its subdirectories
    # os.walk yields (dirpath, dirnames, filenames)
    for dirpath, _, filenames in os.walk(source_dir):
        # Skip the destination folder if it is inside the source folder
        if dirpath == destination_dir:
            continue
            
        print(f"Scanning folder: {dirpath}")
        
        for filename in filenames:
            # Check if the file has one of the target extensions (case-insensitive)
            if filename.lower().endswith(extensions):
                source_path = os.path.join(dirpath, filename)
                destination_path = os.path.join(destination_dir, filename)
                
                # 3. Handle Duplicate File Names
                # If a file with the same name exists in the destination, rename the incoming file
                if os.path.exists(destination_path):
                    name, ext = os.path.splitext(filename)
                    i = 1
                    # Keep trying new names until a unique one is found
                    new_filename = f"{name}_{i}{ext}"
                    new_destination_path = os.path.join(destination_dir, new_filename)
                    while os.path.exists(new_destination_path):
                        i += 1
                        new_filename = f"{name}_{i}{ext}"
                        new_destination_path = os.path.join(destination_dir, new_filename)
                    
                    print(f"  * Duplicate found. Renaming: {filename} -> {new_filename}")
                    destination_path = new_destination_path
                    duplicates_skipped += 1
                    
                # 4. Move the file
                try:
                    shutil.move(source_path, destination_path)
                    files_moved += 1
                except Exception as e:
                    print(f"  ! Error moving {source_path}: {e}")

    print("\n--- Consolidation Complete ---")
    print(f"Total files moved: {files_moved}")
    print(f"Total files renamed (duplicates handled): {duplicates_skipped}")
    print("------------------------------")
    
# --- CONFIGURATION ---

# 🛑 CHANGE THESE TWO LINES TO YOUR ACTUAL FOLDER PATHS 🛑
SOURCE_DIRECTORY = "/home/s124z/Desktop/E130-Projekte/Saliq_datasets/Heico/FO_frames/"
DESTINATION_DIRECTORY = "/home/s124z/Code/YOLO/heico_dataset/test/imagess"

# Run the function
consolidate_files(SOURCE_DIRECTORY, DESTINATION_DIRECTORY)