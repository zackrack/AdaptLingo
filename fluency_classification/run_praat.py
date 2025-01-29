import subprocess
from pathlib import Path
import os

# Define the parameters for the Praat script
pre_processing = "Band pass (300..3300 Hz)"
silence_threshold = -25.0
minimum_dip_near_peak = 2.0
minimum_pause_duration = 0.3
detect_filled_pauses = True
language = "English"
filled_pause_threshold = 1.0
data_output = "Save as text file"
data_collection_type = "OverWriteData"
keep_objects = True

# Path to the Praat executable and script
praat_executable = "C:/Program Files/Praat/praat.exe"  # Replace with the path to your Praat executable
script_path = "Nuclei.praat"  # Replace with the path to your Praat script

# Base directory for the audio folders
base_folder = "audio"
audio_directories = ["A1", "A2", "B1", "B2", "C1", "C2", "Unknown"]

# Function to process a single folder
def process_folder(level):
    sliced_folder = Path(base_folder) / level

    # Check if the sliced folder exists
    if not sliced_folder.exists():
        print(f"[DEBUG] Sliced folder not found: {sliced_folder}", flush=True)
        return

    # Define file specification for all .wav files in the sliced folder
    file_spec = str(sliced_folder / "*.wav")

    # Build the Praat command and arguments
    args = [
        praat_executable,
        "--run",
        script_path,
        file_spec,
        pre_processing,
        str(silence_threshold),
        str(minimum_dip_near_peak),
        str(minimum_pause_duration),
        "yes" if detect_filled_pauses else "no",
        language,
        str(filled_pause_threshold),
        data_output,
        data_collection_type,
        "yes" if keep_objects else "no"
    ]

    # Try running the Praat script for the current folder
    try:
        print(f"[DEBUG] Processing all files in folder: {sliced_folder}", flush=True)
        print(f"[DEBUG] Running command: {' '.join(args)}", flush=True)

        # Execute the Praat command and stream real-time output
        with subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
        ) as process:
            for line in process.stdout:
                print(f"[Praat stdout]: {line.strip()}", flush=True)
            for line in process.stderr:
                print(f"[Praat stderr]: {line.strip()}", flush=True)

        # Verify the output file was created and rename it to avoid overwriting
        output_file = "SyllableNuclei.txt"
        if os.path.exists(output_file):
            new_output_name = f"SyllableNuclei_{level}.txt"
            os.rename(output_file, new_output_name)
            print(f"[INFO] Output saved to: {new_output_name}", flush=True)
        else:
            print(f"[ERROR] Expected output file '{output_file}' not found for folder: {sliced_folder}", flush=True)

        print(f"[INFO] Successfully processed folder: {sliced_folder}", flush=True)

    except subprocess.CalledProcessError as e:
        # Log the error with detailed output and standard error messages
        print(f"[ERROR] Error processing folder {sliced_folder} with CalledProcessError:", flush=True)
        print(f"[ERROR] Standard Output:\n{e.stdout}", flush=True)
        print(f"[ERROR] Standard Error:\n{e.stderr}", flush=True)
    except Exception as e:
        # Catch any other unexpected exceptions
        print(f"[ERROR] Unexpected error processing folder {sliced_folder}: {e}", flush=True)

# Process each folder sequentially
for level in audio_directories:
    process_folder(level)

print("[INFO] Processing completed for all folders.", flush=True)
