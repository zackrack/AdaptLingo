import os
from werkzeug.utils import secure_filename
import subprocess
import os
from pathlib import Path
import pandas
import uuid
import re 
from wordsegment import load, segment
load()

def transcribe_audio(audio_file, crisperwhisper_pipe):
    """
    Transcribe an audio file using the crisperwhisper_pipe pipeline.
    If audio_file is a string, it's assumed to be a local file path.
    Otherwise, it's a file-like object from Flask (request.files['audio']),
    which we'll save to disk before transcription.
    """

    # Check if audio_file is just a path (str) or a file-like object
    if isinstance(audio_file, str):
        # Assume it's already a valid filepath
        audio_path = audio_file
    else:
        # It's likely a Flask 'FileStorage' object; save to a unique .wav file in static/audio
        filename = f"{uuid.uuid4()}.wav"
        audio_dir = os.path.join("static", "audio")
        if not os.path.exists(audio_dir):
            os.makedirs(audio_dir)

        audio_path = os.path.join(audio_dir, filename)
        audio_file.save(audio_path)

    # Use the CrisperWhisper pipeline
    # pipeline_output = crisperwhisper_pipe(audio_path)

    # Combine all chunk texts
    # transcription = " ".join(chunk["text"].strip() for chunk in pipeline_output["chunks"])
    # print("TRANSCRIPTION: ", transcription)
    # Return both transcription and the saved audio_path

    segments, info = crisperwhisper_pipe.transcribe(
        audio_path,
        beam_size=5,
        language="en",
        word_timestamps=False  # ← this is default, but safe to be explicit
    )

    # 👇 Force evaluation of generator ONCE
    segments = list(segments)
    print("Segments: ", segments)
    # 👇 Join all text pieces with space
    transcription = smart_space_recover(" ".join(seg.text.strip() for seg in segments))

    print("TRANSCRIPTION:", transcription)

    return transcription, audio_path


def smart_space_recover(text):
    """
    Split glued words (like 'Thisisatestsentence') into actual words.
    Keeps punctuation and filler words intact.
    """
    # 1. Split at punctuation (so we treat sentences independently)
    parts = re.split(r'([.?!])', text)
    
    fixed = []
    for i in range(0, len(parts), 2):
        chunk = parts[i].strip()
        punct = parts[i + 1] if i + 1 < len(parts) else ""
        
        # Only run segmenter if chunk is more than one word and glued
        if " " not in chunk and len(chunk) > 8:
            split_chunk = " ".join(segment(chunk))
        else:
            split_chunk = chunk
        
        fixed.append(split_chunk + punct)

    return " ".join(fixed)

def classify_fluency(model, speechrate, artrate, asd):
    level = model.predict([[speechrate, artrate, asd]])
    return level