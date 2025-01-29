import os
from werkzeug.utils import secure_filename
import subprocess
import os
from pathlib import Path
import pandas

def transcribe_audio(audio_file, whisper_model):
    audio_path = f"/tmp/{secure_filename(audio_file.filename)}"
    audio_file.save(audio_path)

    transcription = whisper_model.transcribe(audio_path)
    user_input = transcription['text'].strip()

    os.remove(audio_path)

    return user_input, audio_path

def run_praat_script(base_folder):
    pre_processing = "Band pass (300..3300 Hz)"
    silence_threshold = -25.0
    minimum_dip_near_peak = 2.0
    minimum_pause_duration = 0.3
    detect_filled_pauses = True
    language = "English"
    filled_pause_threshold = 1.0
    data_output = "Save as text file"
    data_collection_type = "OverWriteData"
    keep_objects = True

    praat_executable = "C:/Program Files/Praat/praat.exe"
    script_path = "Nuclei.praat" 
    base_folder = "audio/static"

    sliced_folder = Path(base_folder)

    file_spec = str(sliced_folder / "*.wav")

    # Build the Praat command and arguments
    args = [
        praat_executable,
        "--run",
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

    # Try running the Praat script for the current folder
    try:
        print(f"[DEBUG] Processing all files in folder: {sliced_folder}", flush=True)
        print(f"[DEBUG] Running command: {' '.join(args)}", flush=True)

        # Execute the Praat command and stream real-time output
        with subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
        ) as process:
            for line in process.stdout:
                print(f"[Praat stdout]: {line.strip()}", flush=True)
            for line in process.stderr:
                print(f"[Praat stderr]: {line.strip()}", flush=True)

        print(f"[INFO] Successfully processed folder: {sliced_folder}", flush=True)

    except subprocess.CalledProcessError as e:
        # Log the error with detailed output and standard error messages
        print(f"[ERROR] Error processing folder {sliced_folder} with CalledProcessError:", flush=True)
        print(f"[ERROR] Standard Output:\n{e.stdout}", flush=True)
        print(f"[ERROR] Standard Error:\n{e.stderr}", flush=True)
    except Exception as e:
        # Catch any other unexpected exceptions
        print(f"[ERROR] Unexpected error processing folder {sliced_folder}: {e}", flush=True)

    print("[INFO] Processing completed for all folders.", flush=True)

def read_praat_output():
    with open('SyllableNuclei.txt', 'r') as f:
        content = f.read()
    return content

# name, nsyll, npause, dur(s), phonationtime(s), speechrate(nsyll/dur), articulation_rate(nsyll/phonationtime), ASD(speakingtime/nsyll), nrFP, tFP(s)
# audio_1314, 37, 9, 18.04, 10.59, 2.05, 3.49, 0.286, 5, 0.760 
def parse_praat_output(output):
    df = pandas.DataFrame(output)
    speechrate = df[' speechrate(nsyll/dur)'][0]
    artrate = df[' articulation_rate(nsyll/phonationtime)'][0]
    asd = df[' ASD(speakingtime/nsyll)'][0]
    return speechrate, artrate, asd

def classify_fluency(model, speechrate, artrate, asd):
    level = model.infer(speechrate, artrate, asd)
    return level