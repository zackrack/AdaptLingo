import os
import re
import uuid

# load()

def transcribe_audio(audio_file, model):
    """
    Transcribe speech with CrisperWhisper 2.0 in verbatim mode.

    Args:
        audio_file: path to .wav file or Flask FileStorage object
        model: CrisperWhisperModel configured with the Turbo CT2 backend

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

    try:
        result = model.transcribe(
            audio_path,
            language="en",
            mode="verbatim",
        )
    except Exception as e:
        raise RuntimeError(f"[ASR Error] Failed to transcribe audio: {e}") from e

    return result.text.strip(), audio_path

import re
from transformers import pipeline

# Grammar correction model, loaded on first use (only smart_space_recover needs it)
_fixer = None

def _get_fixer():
    global _fixer
    if _fixer is None:
        _fixer = pipeline("text2text-generation", model="vennify/t5-base-grammar-correction")
    return _fixer

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
    result = _get_fixer()(prompt, max_length=128, clean_up_tokenization_spaces=True)[0]['generated_text']

    # Step 3: Re-tag fillers (optional)
    final = postprocess_fillers(result)
    return final

def classify_fluency(model, speechrate, artrate, asd):
    level = model.predict([[speechrate, artrate, asd]])
    return level
