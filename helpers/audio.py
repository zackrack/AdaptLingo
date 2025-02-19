import os
from werkzeug.utils import secure_filename
import subprocess
import os
from pathlib import Path
import pandas

import os
import uuid

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
    pipeline_output = crisperwhisper_pipe(audio_path)

    # Combine all chunk texts
    transcription = " ".join(chunk["text"].strip() for chunk in pipeline_output["chunks"])

    # Return both transcription and the saved audio_path
    return transcription, audio_path


def classify_fluency(model, speechrate, artrate, asd):
    level = model.infer(speechrate, artrate, asd)
    return level