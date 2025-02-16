import os
import shutil

# Set the path to the root folder of the dataset
source_dir = "C:/Users/maria/VST files/dataset/"
destination_dir = "C:/Users/maria/VST files/dataset"

# Create the destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# Walk through all subdirectories to find MIDI files
for root, _, files in os.walk(source_dir):
    for file in files:
        if file.endswith(".mid") or file.endswith(".midi"):  # Check for MIDI files
            source_file = os.path.join(root, file)
            destination_file = os.path.join(destination_dir, file)
            
            # Copy MIDI file to the destination directory
            shutil.copy2(source_file, destination_file)
            print(f"Copied: {source_file} -> {destination_file}")

print("All MIDI files have been extracted to:", destination_dir)
