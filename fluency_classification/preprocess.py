import os
import librosa
import soundfile as sf
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

def normalize_audio(y, target_dBFS=-20.0):
    """
    Normalizes the audio volume to a target decibel level.
    """
    rms = librosa.feature.rms(y=y).mean()
    scalar = (10**(target_dBFS / 20)) / max(rms, 1e-10)
    return y * scalar

def preprocess_and_filter_audio(input_path, output_path, min_duration=0.06, target_dBFS=-20.0):
    """
    Preprocesses the audio by trimming silence, normalizing volume, and ensuring it meets the minimum duration.
    If the audio is shorter than min_duration, deletes the file.
    """
    try:
        # Step 1: Load and trim silence
        y, sr = librosa.load(input_path, sr=None)
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)

        # Step 2: Normalize volume
        y_normalized = normalize_audio(y_trimmed, target_dBFS=target_dBFS)

        # Step 3: Check duration
        duration = librosa.get_duration(y=y_normalized, sr=sr)
        if duration < min_duration:
            print(f"File {input_path} is too short after processing; deleting.")
            os.remove(input_path)  # Delete file if it's too short
            return False  # Skip this file
        
        # Save the processed audio back to the output path
        sf.write(output_path, y_normalized, sr)
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False

def process_folder_in_parallel(folder_path):
    """
    Processes all .wav files in the specified folder in parallel and removes short files.
    """
    audio_files = list(folder_path.glob("*.wav"))
    with ProcessPoolExecutor() as executor:
        results = executor.map(
            lambda audio_file: preprocess_and_filter_audio(audio_file, audio_file),
            audio_files,
        )
    print(f"Completed processing for: {folder_path}")

if __name__ == "__main__":
    # Base directory containing the proficiency folders
    base_folder = Path("audio")
    audio_directories = ["A1", "A2", "B1", "B2", "C1", "C2", "Unknown"]

    # Process each .wav file in the sliced directories in parallel
    for level in audio_directories:
        sliced_folder = base_folder / level
        
        if not sliced_folder.exists():
            print(f"Sliced folder not found: {sliced_folder}")
            continue
        
        print(f"Processing files in: {sliced_folder}")
        process_folder_in_parallel(sliced_folder)

    print("All files have been preprocessed, normalized, filtered, and replaced.")
