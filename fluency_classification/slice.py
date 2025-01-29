import os
import csv
import pandas as pd
import numpy as np
from pathlib import Path
from pydub import AudioSegment
from silero_vad import load_silero_vad, get_speech_timestamps, read_audio
import soundfile as sf
import librosa
from tempfile import TemporaryDirectory
import concurrent.futures
from tqdm import tqdm

# Silence Trimming Function
def trim_silence(y):
    """Trim leading and trailing silence from an audio signal."""
    y_trimmed, _ = librosa.effects.trim(y, top_db=20, frame_length=2, hop_length=500)
    return y_trimmed

# Convert non-wav files to wav
def convert_to_wav(input_file: Path, output_dir: Path) -> Path:
    """Convert any non-wav audio file to wav format."""
    audio = AudioSegment.from_file(input_file)
    output_file = output_dir / (input_file.stem + ".wav")
    audio.export(output_file, format="wav")
    return output_file

def split_wav(vad_model, audio_file: Path, output_dir: Path, min_sec=8, max_sec=30, min_silence_dur_ms=300):
    """Split audio file based on voice activity detection (VAD)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    wav = read_audio(str(audio_file), sampling_rate=16000)
    speech_timestamps = get_speech_timestamps(
        wav,
        vad_model,
        min_silence_duration_ms=min_silence_dur_ms,
        min_speech_duration_ms=int(min_sec * 1000),
        max_speech_duration_s=max_sec,
        return_seconds=True
    )

    data, sr = sf.read(audio_file)
    segment_paths = []

    for i, ts in enumerate(speech_timestamps):
        start_sample = int(ts['start'] * sr)
        end_sample = int(ts['end'] * sr)
        segment = data[start_sample:end_sample]
        
        segment_file = output_dir / f"{audio_file.stem}_part_{i}.wav"
        sf.write(segment_file, segment, sr)
        segment_paths.append(segment_file)

    return segment_paths



# Process a single file
def process_single_file(file_record, output_base_dir):
    folder = file_record['label']
    file_path = Path(file_record['file_path'])
    output_dir = output_base_dir / folder
    
    if file_path.is_file() and file_path.suffix.lower() in [".mp3", ".wav", ".m4a", ".flac"]:
        print(f"Processing file: {file_path}")
        vad_model = load_silero_vad()

        try:
            with TemporaryDirectory() as temp_dir:
                temp_dir_path = Path(temp_dir)

                if file_path.suffix.lower() != ".wav":
                    file_path = convert_to_wav(file_path, temp_dir_path)

                y, sr = librosa.load(str(file_path), sr=None)
                y_trimmed = trim_silence(y)

                preprocessed_file = temp_dir_path / f"{file_path.stem}_preprocessed.wav"
                sf.write(preprocessed_file, y_trimmed, sr)

                segment_paths = split_wav(vad_model, preprocessed_file, output_dir)

                return [{"file_path": segment.as_posix(), "label": folder} for segment in segment_paths]

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

    return None

# Parallel processing of audio files
def process_audio_files_in_parallel(audio_files_csv, output_base_dir):
    df = pd.read_csv(audio_files_csv)

    df['output_base_dir'] = output_base_dir.as_posix()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(process_single_file_wrapper, df.to_dict('records')),
            total=len(df)
        ))
    
    features_list = [item for result in results if result is not None for item in result]
    
    return pd.DataFrame(features_list)

# Wrapper function for parallel processing
def process_single_file_wrapper(file_record):
    output_base_dir = Path(file_record.pop('output_base_dir'))
    return process_single_file(file_record, output_base_dir)

# Main script execution
if __name__ == "__main__":
    input_dir = Path("audio_all_old")  # Root directory containing audio/A1, audio/A2, etc.
    output_base_dir = Path("audio_sliced")
    output_base_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_base_dir / "sliced_audio_files.csv"
    
    # Generate the list of audio files and labels
    audio_files = []
    for folder in ["A1", "A2", "B1", "B2", "C1", "C2"]:
        folder_path = input_dir / folder
        for file in folder_path.rglob("*"):
            if file.is_file() and file.suffix.lower() in [".mp3", ".wav", ".m4a", ".flac"]:
                audio_files.append({"file_path": file.as_posix(), "label": folder})

    # Save the list to a CSV for parallel processing
    audio_files_csv = input_dir / "audio_files.csv"
    pd.DataFrame(audio_files).to_csv(audio_files_csv, index=False)

    # Process audio files in parallel and save the output CSV
    processed_data = process_audio_files_in_parallel(audio_files_csv, output_base_dir)
    processed_data.to_csv(output_csv, index=False)
    print("Processing complete. Sliced files and CSV saved.")
