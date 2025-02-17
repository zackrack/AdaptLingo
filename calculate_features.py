import numpy as np
import parselmouth
import librosa
import re
from scipy.signal import savgol_filter
from helpers import transcribe_audio  # Make sure this function works with your new ASR pipeline
import nltk
from nltk.corpus import cmudict

cmu_dict = cmudict.dict()

def compute_tempo_metrics(
    transcription: str,
    audio_path: str,
    top_db: float = 20.0,
    sr_downsample: int = 16000,
    skip_pitch_check: bool = False,
) -> dict:
    """
    Combines:
      - Syllable counting via syllable_estimate_transcription()
      - Silence detection via librosa.effects.split()
    
    Returns a dict of:
       n_syllables, speech_rate, articulation_rate, asd
    """
    n_syllables = syllable_estimate_transcription(transcription)

    # Librosa load & downsample for silence detection
    y, sr = librosa.load(audio_path, sr=sr_downsample)
    total_duration = len(y) / sr
    if total_duration <= 0:
        return {
            "n_syllables": n_syllables,
            "speech_rate": 0.0,
            "articulation_rate": 0.0,
            "asd": 0.0
        }

    # Identify non-silent intervals
    intervals = librosa.effects.split(y, top_db=top_db)
    speaking_time = sum((end - start) for start, end in intervals)
    speaking_time_s = speaking_time / sr
    pause_time_s = total_duration - speaking_time_s
    if pause_time_s < 0:
        pause_time_s = 0.0

    # Compute speech metrics
    speech_rate = (n_syllables / total_duration) if total_duration > 0 else 0.0
    articulation_rate = (n_syllables / speaking_time_s) if speaking_time_s > 0 else 0.0
    asd = (speaking_time_s / n_syllables) if (n_syllables > 0 and speaking_time_s > 0) else 0.0

    return {
        "n_syllables": n_syllables,
        "speech_rate": speech_rate,
        "articulation_rate": articulation_rate,
        "asd": asd
    }

def count_syllables(word):
    """
    Count syllables using CMUdict. If the word is not found,
    fall back to a simple heuristic.
    """
    word_lower = word.lower()
    if word_lower in cmu_dict:
        syllable_counts = [
            len([phoneme for phoneme in pron if phoneme[-1].isdigit()])
            for pron in cmu_dict[word_lower]
        ]
        return min(syllable_counts)
    else:
        return heuristic_count(word_lower)

def heuristic_count(word):
    """
    Fallback heuristic for syllable counting when a word isn't in CMUdict.
    It counts vowel groups as syllables.
    """
    vowels = "aeiouy"
    count = 0
    if word and word[0] in vowels:
        count += 1
    for i in range(1, len(word)):
        if word[i] in vowels and word[i-1] not in vowels:
            count += 1
    if word.endswith("e") and count > 1:
        count -= 1
    return count if count > 0 else 1

def syllable_estimate_transcription(text):
    # Clean text: extract words using regex to skip punctuation.
    words = re.findall(r'\b\w+\b', text.lower())
    filler_words = {"uh": 1, "um": 1, "ah": 1, "er": 1, "hmm": 1, "mmm": 1, "oh": 1, "eh": 1, "yeah": 1}
    total_syllables = 0
    for word in words:
        if word in filler_words:
            total_syllables += filler_words[word]
        else:
            total_syllables += count_syllables(word)
    return total_syllables

def calculate_all_features(transcription, audio_file):
    metrics = compute_tempo_metrics(
        transcription,
        audio_file,
        top_db=20.0,
        sr_downsample=16000,
        skip_pitch_check=False
    )
    return metrics['n_syllables'], metrics['speech_rate'], metrics['articulation_rate'], metrics['asd']

if __name__ == "__main__":
    # Import just the CrisperWhisper initialization
    from initialize_crisperwhisper import initialize_crisperwhisper

    # Initialize CrisperWhisper pipeline only
    crisperwhisper_pipe = initialize_crisperwhisper()

    audio_file = "audio_236.wav"
    # Use the CrisperWhisper pipeline for transcription with a seed prompt for fillers
    transcription = transcribe_audio(
        audio_file,
        crisperwhisper_pipe  # Passing the new ASR pipeline
    )
    print("Transcription:", transcription)
    
    # Compute tempo metrics
    metrics = compute_tempo_metrics(
        transcription,
        audio_file,
        top_db=20.0,
        sr_downsample=16000,
        skip_pitch_check=False
    )
    print("=== Tempo Metrics ===")
    print(f"  Syllables         = {metrics['n_syllables']}")
    print(f"  Speech Rate       = {metrics['speech_rate']:.2f} syll/sec")
    print(f"  Articulation Rate = {metrics['articulation_rate']:.2f} syll/sec")
    print(f"  ASD               = {metrics['asd']:.2f} sec/syll")