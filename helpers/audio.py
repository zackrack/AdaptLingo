import os
from werkzeug.utils import secure_filename
import subprocess
import os
from pathlib import Path
import pandas

def transcribe_audio(audio_file, crisperwhisper_pipe):
    if isinstance(audio_file, str):
        audio_path = audio_file
    elif hasattr(audio_file, "name"):
        audio_path = audio_file.name
    else:
        raise ValueError("Invalid audio_file type. Expected a file path (str) or a file-like object.")

    # Use CrisperWhisper's pipeline with the seed prompt.
    pipeline_output = crisperwhisper_pipe(audio_path)
    
    # Optionally, if you want to adjust pauses/timestamps, you can call a helper function here.
    # pipeline_output = adjust_pauses_for_hf_pipeline_output(pipeline_output)

    # Concatenate the text chunks to form the full transcription.
    transcription = " ".join(chunk["text"].strip() for chunk in pipeline_output["chunks"])
    return transcription

def classify_fluency(model, speechrate, artrate, asd):
    level = model.infer(speechrate, artrate, asd)
    return level