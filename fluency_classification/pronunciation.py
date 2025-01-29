
import os
import parselmouth
from parselmouth.praat import call
import pandas as pd
from multiprocessing import Pool, cpu_count
from sklearn.metrics.pairwise import cosine_similarity
from jiwer import wer
from transformers import AutoTokenizer, AutoModel
from pydub import AudioSegment, silence  # For volume normalization
import numpy as np
import openai
import csv
import logging
import whisper
from openai import OpenAI
import glob
from pydub.utils import mediainfo
import subprocess

# Disable TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# Set OpenAI API key
openai = OpenAI(api_key="")

# Load embeddings model and tokenizer (all-MiniLM-L6-v2)
print("Loading sentence-transformer model for embeddings...")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
print("Model loaded successfully.")

# CSV output path
output_path = "audio/processed_results2.csv"

# Initialize the CSV with headers
def initialize_csv(output_path):
    print(f"Initializing CSV file: {output_path}")
    with open(output_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "name", "Mean Pitch (Hz)", "Min Pitch (Hz)", "Max Pitch (Hz)",
            "F1 (Hz)", "F2 (Hz)", "Mean Intensity (dB)", "Spectral Tilt",
            "Cosine Similarity", "Cosine Distance", "WER", "Transcription",
            "Corrected Transcription", "Final dBFS", "Total Gain (dB)", "Normalization Success"
        ])
    print("CSV initialized successfully.")

# Function to write a single row to the CSV
def write_to_csv(output_path, row):
    with open(output_path, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(row)
    print(f"Written to CSV: {row[:3]}...")  # Preview for clarity

from pydub.utils import mediainfo
import subprocess

def ensure_wav_format(file_path, output_dir="processed_audio"):
    """
    Ensure the audio file is in proper WAV format. Convert if necessary.

    Args:
        file_path (str): Path to the input audio file.
        output_dir (str): Directory to save converted files.

    Returns:
        str: Path to the WAV file (converted or original).
    """
    info = mediainfo(file_path)
    file_format = info.get("format_name")
    file_basename = os.path.basename(file_path)
    
    # If the file is already a valid WAV, return the original path
    if file_format == "wav":
        return file_path
    
    print(f"Converting {file_basename} to WAV format...")
    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the output file path
    wav_path = os.path.join(output_dir, os.path.splitext(file_basename)[0] + ".wav")
    
    # Convert to WAV using ffmpeg
    try:
        subprocess.run(
            ["ffmpeg", "-i", file_path, "-c:a", "pcm_s16le", "-ar", "44100", wav_path],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        print(f"Converted {file_basename} to {wav_path}")
        return wav_path
    except subprocess.CalledProcessError as e:
        print(f"Failed to convert {file_basename}: {e}")
        return None
    
def normalize_audio(file_path, output_dir="processed_audio", max_iterations=40, threshold_db=-1.00):
    """
    Normalize and trim silence from the audio file iteratively, then export it to the specified directory.

    Args:
        file_path (str): Path to the input audio file.
        output_dir (str): Directory where the processed file will be saved (default: "processed_audio").
        max_iterations (int): Maximum iterations to adjust gain.
        threshold_db (float): Target dBFS for normalization (default: -1.0).

    Returns:
        tuple: (output_path, final_dbfs, total_gain, success)
            output_path (str): Path to the trimmed and normalized audio file.
            final_dbfs (float): Final dBFS of the normalized audio.
            total_gain (float): Total gain applied during normalization.
            success (bool): Whether the audio was successfully normalized to the target.
    """
    file_path = ensure_wav_format(file_path, output_dir)
    if file_path is None:
        return None, None, None, False  # Skip invalid or unconvertible files

    print(f"Normalizing and trimming silence for {os.path.basename(file_path)}...")

    # Load and process audio
    try:
        audio = AudioSegment.from_file(file_path, format="wav")
    except:
        print(f"Could not decode file {file_path}")
        return None, None, None, False  # Return placeholders for errors
    
    os.makedirs(output_dir, exist_ok=True)

    # Trim silence from the beginning and end
    trimmed_ranges = silence.detect_nonsilent(audio, min_silence_len=500, silence_thresh=-40)
    if not trimmed_ranges:
        print(f"No significant speech detected in {os.path.basename(file_path)}. Using the normalized audio.")
        trimmed_audio = audio
    else:
        start_trim = trimmed_ranges[0][0]
        end_trim = trimmed_ranges[-1][1]
        trimmed_audio = audio[start_trim:end_trim]

    total_gain = 0.0
    success = False
    
    for iteration in range(max_iterations):
        current_dbfs = trimmed_audio.max_dBFS
        if current_dbfs >= -1.01:
            success = True
            print(f"Audio normalized successfully after {iteration} iterations. Final dBFS: {current_dbfs:.2f}")
            break
        gain_needed = threshold_db - current_dbfs
        trimmed_audio = trimmed_audio.apply_gain(gain_needed)
        total_gain += gain_needed
        print(f"Iteration {iteration + 1}: Applied additional gain of {gain_needed:.2f} dB. New dBFS: {trimmed_audio.max_dBFS:.2f}")

    if not success:
        print(f"Warning: Audio file {os.path.basename(file_path)} could not be perfectly normalized. Final dBFS: {trimmed_audio.max_dBFS:.2f}")

    output_path = os.path.join(output_dir, os.path.basename(file_path))
    trimmed_audio.export(output_path, format="wav")
    print(f"Trimmed and normalized audio saved to {output_path}")
    return output_path, trimmed_audio.max_dBFS, total_gain, success

def get_embedding(text):
    print(f"Generating embeddings for text: {text[:30]}...")
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**tokens)
    embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    print("Embeddings generated successfully.")
    return embedding

def transcribe_audio(file_path):
    print(f"Transcribing {os.path.basename(file_path)} using Whisper...")
    model = whisper.load_model("small")
    result = model.transcribe(file_path)
    transcription = result["text"]
    print(f"Transcription completed: {transcription[:50]}...")
    return transcription

def correct_transcription(transcription):
    print(f"Correcting transcription: {transcription[:50]}...")
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful grammar correction assistant that produces grammatically correct and natural corrections."},
            {"role": "user", "content": f"Here is a poorly transcribed passage. Please rewrite it clearly and concisely while maintaining the original meaning. Only provide the corrected version without any additional explanation or notes:\n\n{transcription}"}
        ]
    )
    corrected_text = response.choices[0].message.content
    print(f"Correction completed: {corrected_text[:50]}...")
    return corrected_text.strip()

from pydub import silence

def calculate_pitch_bounds(sound):
    """
    Dynamically calculate pitch bounds based on speech activity.
    Args:
        sound (parselmouth.Sound): Audio object.
    Returns:
        (float, float): Min and max pitch bounds.
    """
    print("Dynamically calculating pitch bounds...")
    
    # Convert to Pydub AudioSegment for silence detection
    duration = sound.get_total_duration()
    sampling_rate = sound.sampling_frequency
    audio_array = sound.values.T[0] * 32767  # Convert to integer range for Pydub
    audio_segment = AudioSegment(
        audio_array.tobytes(),
        frame_rate=sampling_rate,
        sample_width=2,
        channels=1
    )

    # Detect nonsilent regions
    nonsilent_ranges = silence.detect_nonsilent(audio_segment, min_silence_len=200, silence_thresh=-40)
    
    if nonsilent_ranges:
        # Select the largest nonsilent range or one near the center
        start, end = max(nonsilent_ranges, key=lambda r: r[1] - r[0])
        speech_segment = sound.extract_part(from_time=start / 1000.0, to_time=end / 1000.0)
        spectrum = call(speech_segment, "To Spectrum", False)
        spectral_mean_freq = call(spectrum, "Get spectral centroid")
        
        # Calculate pitch bounds
        min_pitch = max(50, spectral_mean_freq / 4)
        max_pitch = min(600, spectral_mean_freq * 4)
    else:
        print("No speech detected. Using default pitch bounds.")
        min_pitch = 75  # Default for no detected speech
        max_pitch = 600

    print(f"Calculated pitch bounds: min_pitch={min_pitch:.2f}, max_pitch={max_pitch:.2f}")
    return min_pitch, max_pitch

def analyze_audio(file_data):
    file_index, total_files, file_path = file_data
    print(f"Processing audio {file_index + 1}/{total_files}: {os.path.basename(file_path)}")
    
    # Normalize audio volume and collect normalization details
    normalized_path, final_dbfs, total_gain, _ = normalize_audio(file_path)
    
    # Load the normalized audio into Parselmouth
    sound = parselmouth.Sound(normalized_path)
    duration = sound.get_total_duration()
    
    # Dynamically determine pitch bounds
    # Calculate pitch bounds dynamically
    min_pitch = 75  # Default lower bound for human speech
    max_pitch = 600  # Default upper bound for human speech

    # Use these bounds for pitch extraction
    pitch = call(sound, "To Pitch", 0.0, min_pitch, max_pitch)
    
    print(f"Calculated pitch bounds: min_pitch={min_pitch:.2f}, max_pitch={max_pitch:.2f}")
    
    # Calculate pitch (intonation)
    print("Calculating pitch...")
    pitch = call(sound, "To Pitch", 0.0, min_pitch, max_pitch)
    mean_pitch = call(pitch, "Get mean", 0, 0, "Hertz")
    min_pitch_actual = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")
    max_pitch_actual = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")
    
    # Calculate formants (vowel quality)
    print("Calculating formants...")
    formant = call(sound, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
    f1_values = [
        call(formant, "Get value at time", 1, t, "Hertz", "Linear")
        for t in np.arange(0, duration, 0.01)
    ]
    f2_values = [
        call(formant, "Get value at time", 2, t, "Hertz", "Linear")
        for t in np.arange(0, duration, 0.01)
    ]
    f1_avg = np.nanmean(f1_values) if len(f1_values) > 0 else float('nan')
    f2_avg = np.nanmean(f2_values) if len(f2_values) > 0 else float('nan')
    
    # Calculate intensity (stress)
    print("Calculating intensity...")
    intensity = call(sound, "To Intensity", 75, 0.0)
    mean_intensity = call(intensity, "Get mean", 0, 0)
    
    # Spectral Tilt
    print("Calculating spectral tilt...")
    spectrum = call(sound, "To Spectrum", False)
    low_freq_energy = call(spectrum, "Get band energy", 0, 500)
    high_freq_energy = call(spectrum, "Get band energy", 2000, 5000)
    spectral_tilt = low_freq_energy - high_freq_energy
    
    # Transcription and correction
    transcription = transcribe_audio(normalized_path)
    corrected_transcription = correct_transcription(transcription)
    
    # Word Error Rate (WER)
    print("Calculating Word Error Rate...")
    word_error_rate = wer(corrected_transcription, transcription)
    
    # Embedding-based similarity
    print("Calculating cosine similarity...")
    transcription_embedding = get_embedding(transcription)
    corrected_embedding = get_embedding(corrected_transcription)
    cos_sim = cosine_similarity(transcription_embedding, corrected_embedding)[0][0]
    cos_distance = 1 - cos_sim
    
    # Prepare the row, including normalization details
    row = [
        os.path.basename(file_path), mean_pitch, min_pitch_actual, max_pitch_actual, f1_avg, f2_avg,
        mean_intensity, spectral_tilt, cos_sim, cos_distance, word_error_rate,
        transcription, corrected_transcription, final_dbfs
    ]
    write_to_csv(output_path, row)  # Write result to CSV
    
    # Clean up temporary file
    os.remove(normalized_path)
    print(f"Temporary file removed: {normalized_path}")

def cleanup_temp_files(output_dir="processed_audio"):
    for temp_file in glob.glob(f"{output_dir}/*.wav"):
        try:
            os.remove(temp_file)
        except Exception as e:
            print(f"Error cleaning up {temp_file}: {e}")

def get_audio_file_paths(folder_path):
    print(f"Collecting audio files from: {folder_path}")
    file_paths = []
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith(".wav"):
                file_paths.append(os.path.join(root, file_name))
    print(f"Found {len(file_paths)} audio files.")
    return file_paths

if __name__ == "__main__":
    folder_path = "audio"
    initialize_csv(output_path)
    file_paths = get_audio_file_paths(folder_path)
    total_files = len(file_paths)
    file_data = [(i, total_files, path) for i, path in enumerate(file_paths)]

    print(f"Starting audio analysis with {cpu_count()} workers...")
    num_workers = cpu_count()
    with Pool(num_workers) as pool:
        pool.map(analyze_audio, file_data)

    print(f"Analysis complete. Results saved to {output_path}.")
    cleanup_temp_files()
