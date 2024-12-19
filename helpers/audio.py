import os
from werkzeug.utils import secure_filename
import subprocess
import os

def transcribe_audio(audio_file, whisper_model):
    # Save the audio file temporarily
    audio_path = f"/tmp/{secure_filename(audio_file.filename)}"
    audio_file.save(audio_path)

    # Transcribe the audio to text
    transcription = whisper_model.transcribe(audio_path)
    user_input = transcription['text'].strip()

    # Clean up the temporary audio file
    os.remove(audio_path)

    return user_input, audio_path

def run_praat_script(audio_file_path):
    # Define your input parameters
    file_spec = audio_file_path  # Path to the audio file
    pre_processing = "None"
    silence_threshold = -25.0
    minimum_dip_near_peak = 2.0
    minimum_pause_duration = 0.3
    detect_filled_pauses = True
    language = "English"
    filled_pause_threshold = 1.0
    data_output = "Save as text file"
    data_collection_type = "OverWriteData"
    keep_objects = True

    # Define the command and arguments
    praat_executable = "/usr/bin/praat"  # Replace with the path to your Praat executable
    script_path = "praat/syllable_nuclei.praat"  # Path to your Praat script
    output_dir = "praat/output_files"  # Directory to store the Praat output files

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    args = [
        praat_executable,
        "--run",  # Run the script in batch mode
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

    # Execute the command
    result = subprocess.run(args, capture_output=True, text=True)

    if result.returncode != 0:
        print("Praat script execution failed:")
        print("Standard Output:\n", result.stdout)
        print("Standard Error:\n", result.stderr)
        return None

    # The Praat script writes output to a file; determine the output file path
    # Assuming the output file is saved in the same directory as the audio file with a .txt extension
    output_file_path = os.path.splitext(audio_file_path)[0] + ".txt"

    return output_file_path

def read_praat_output(output_file_path):
    try:
        with open(output_file_path, 'r') as f:
            content = f.read()
        # Optionally, parse the content if needed
        return content
    except FileNotFoundError:
        print(f"Praat output file not found: {output_file_path}")
        return "No audio input or Praat output not available."

