import os
import csv

# Define the directory containing the audio folders
base_directory = "validation_audio"

# Initialize a list to store rows for the CSV
data = []

# Iterate through each subdirectory in the base directory
for proficiency_level in os.listdir(base_directory):
    proficiency_path = os.path.join(base_directory, proficiency_level)
    
    # Check if the path is a directory
    if os.path.isdir(proficiency_path):
        # Iterate through each file in the directory
        for audio_file in os.listdir(proficiency_path):
            # Check if the file has a .wav extension
            if audio_file.endswith(".wav"):
                # Add the file name and proficiency level to the data list
                data.append([audio_file, proficiency_level])

# Define the output CSV file name
output_csv = "validation_audio/audio.csv"

# Write the data to the CSV
with open(output_csv, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(["Audio", "Fluency"])
    # Write the data
    writer.writerows(data)

print(f"CSV file '{output_csv}' created successfully.")
