import os
import json
from datetime import datetime
import uuid

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

def log_interaction(session_id, user_input, assistant_response, fluency_level, audio_path=None):
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "session_id": session_id,
        "user_input": user_input,
        "assistant_response": assistant_response,
        "fluency_level": fluency_level,
        "audio_path": audio_path
    }

    log_file = os.path.join(LOG_DIR, f"{session_id}.jsonl")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")