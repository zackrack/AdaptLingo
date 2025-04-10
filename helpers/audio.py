import os
# from werkzeug.utils import secure_filename
# import subprocess
import os
# from pathlib import Path
# import pandas
import uuid
import re 
# from wordsegment import load, segment
import torch
import uuid
import os
import soundfile as sf
import librosa
import numpy as np

# load()

def transcribe_audio(audio_file, model, processor):
    """
    Hardened transcription function using CrisperWhisper (manual model+processor, not HF pipeline).
    Handles stereo audio, resampling to 16kHz, silence check, and model-safe input.

    Args:
        audio_file: path to .wav file or Flask FileStorage object
        model: HuggingFace AutoModelForSpeechSeq2Seq (CrisperWhisper)
        processor: HuggingFace AutoProcessor (CrisperWhisper)

    Returns:
        transcription (str), audio_path (str)
    """

    # If it's a Flask-style file object, save to disk
    if not isinstance(audio_file, str):
        filename = f"{uuid.uuid4()}.wav"
        audio_dir = os.path.join("static", "audio")
        os.makedirs(audio_dir, exist_ok=True)
        audio_path = os.path.join(audio_dir, filename)
        audio_file.save(audio_path)
    else:
        audio_path = audio_file

    # Read and preprocess audio
    try:
        audio_array, sr = sf.read(audio_path)

        if len(audio_array.shape) > 1:
            audio_array = np.mean(audio_array, axis=1)  # Convert stereo to mono

        if np.max(np.abs(audio_array)) < 1e-4:
            raise ValueError("Audio input is silent or empty.")

        if sr != 16000:
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
            sr = 16000

    except Exception as e:
        raise RuntimeError(f"[Audio Error] Failed to load or preprocess audio: {e}")

    # Tokenize input
    inputs = processor(
        audio=audio_array,
        sampling_rate=sr,
        return_tensors="pt"
    )

    input_features = inputs["input_features"].to(device=model.device, dtype=model.dtype)
    attention_mask = (input_features != 0.0).long()

    model.config.forced_decoder_ids = None

    with torch.no_grad():
        generated = model.generate(
            input_features=input_features,
            attention_mask=attention_mask,
            max_new_tokens=440,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=False,
            language="en",
        )

    transcription = processor.batch_decode(generated.sequences, skip_special_tokens=True)[0]
    return transcription.strip(), audio_path

import re
from transformers import pipeline

# Grammar correction model
fixer = pipeline("text2text-generation", model="vennify/t5-base-grammar-correction")

# Filler word list
filler_words = {"uh": 1, "um": 1, "ah": 1, "er": 1, "hmm": 1, "mmm": 1, "oh": 1, "eh": 1, "yeah": 1}

# Dynamic filler tag maps
FILLER_MAP = {f"[{word.upper()}]": word for word in filler_words}
REVERSE_FILLER_MAP = {word: f"[{word.upper()}]" for word in filler_words}

def prepare_text_with_fillers(raw):
    """Convert [FILLER] tags to actual filler words like 'um'."""
    text = raw
    for tag, word in FILLER_MAP.items():
        text = re.sub(re.escape(tag), word, text, flags=re.IGNORECASE)
    return text

def postprocess_fillers(text):
    """Convert filler words like 'um' back to [FILLER] tags."""
    for word, tag in REVERSE_FILLER_MAP.items():
        text = re.sub(rf"\b{re.escape(word)}\b", tag, text, flags=re.IGNORECASE)
    return text

def smart_space_recover(text):
    # Step 1: Replace filler tags with natural words
    prepped = prepare_text_with_fillers(text)

    # Step 2: Grammar correction
    prompt = f"grammar: {prepped}"
    result = fixer(prompt, max_length=128, clean_up_tokenization_spaces=True)[0]['generated_text']

    # Step 3: Re-tag fillers (optional)
    final = postprocess_fillers(result)
    return final

def classify_fluency(model, speechrate, artrate, asd):
    level = model.predict([[speechrate, artrate, asd]])
    return level