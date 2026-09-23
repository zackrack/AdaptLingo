import gradio as gr
import io
import os
import numpy as np
from threading import Lock
import tempfile
import soundfile as sf
import argparse
from helpers import (
    create_boost_processor,
    create_stopping_criteria,
    knn_search,
    build_prompt,
    generate_chat_response,
    load_initial_data,
    transcribe_audio,
    classify_fluency
)
from initialize import initialize
from calculate_features import calculate_all_features
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn
import openai
from dotenv import load_dotenv
from gradio import mount_gradio_app
from pyngrok import ngrok
import time

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
load_dotenv()

# ------------------------------
# Global initialization and state
# ------------------------------
init_data = {}
config_lock = Lock()

def load_app_data():
    global init_data
    with config_lock:
        init_data = load_initial_data(lambda: initialize(load_local_tts=False))
        print("Loaded Configuration")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
client = openai.OpenAI(api_key=api_key)

load_app_data()
stopping_criteria = create_stopping_criteria(init_data['tokenizer'])

def warmup():
    """
    Run each model once at startup so the first real request doesn't pay
    one-time CUDA/kernel setup costs.
    """
    started_at = time.perf_counter()
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            warmup_wav = f.name
        sf.write(warmup_wav, np.zeros(16000, dtype=np.float32), 16000)
        transcribe_audio(warmup_wav, init_data['crisperwhisper_model'])
        calculate_all_features("hello there", warmup_wav)
        os.remove(warmup_wav)

        for key in ('beginner_collection', 'intermediate_collection', 'advanced_collection'):
            knn_search("hello there", init_data['embedding_model'], init_data[key])

        generate_chat_response(
            init_data['model'], init_data['tokenizer'], [{"role": "user", "content": "Hello!"}],
            create_boost_processor(init_data['tokenizer'], [], 0), stopping_criteria,
            init_data['device']
        )
        print(f"🔥 Warmup done in {time.perf_counter() - started_at:.1f}s")
    except Exception as e:
        print(f"⚠️ Warmup failed (app will still run): {e}")

warmup()

# Sage or shimmer are the best

def generate_speech_openai(text, fluency_level, voice="nova", model="gpt-4o-mini-tts", response_format="wav"):
    """
    Calls OpenAI TTS and returns raw bytes in WAV format, with detailed logging and error tracing.
    """
    try:
        speed = {0: 0.8, 1: 0.9, 2: 1.0}.get(fluency_level, 1.0)

        response = client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            response_format=response_format,
            speed=speed
        )

        if not hasattr(response, "content") or not response.content:
            raise ValueError("OpenAI TTS response missing audio content.")

        return response.content

    except Exception as e:
        print(f"❌ [TTS ERROR] {e}")
        return None

def process_user_audio_openai(audio_np, history):
    request_started_at = time.perf_counter()
    timings = {}

    def record_timing(stage, started_at):
        elapsed = time.perf_counter() - started_at
        timings[stage] = elapsed
        print(f"⏱️ [TIMING] {stage}: {elapsed:.3f}s", flush=True)

    def safe_return(history, audio, label):
        if not isinstance(audio, tuple) or len(audio) != 2:
            audio = (None, None)

        sr, data = audio
        if sr is None or data is None:
            audio = (24000, np.zeros(1, dtype=np.float32))

        total_elapsed = time.perf_counter() - request_started_at
        timing_summary = ", ".join(
            f"{stage}={elapsed:.3f}s" for stage, elapsed in timings.items()
        )
        print(f"⏱️ [TIMING] total: {total_elapsed:.3f}s", flush=True)
        print(f"⏱️ [TIMING SUMMARY] {timing_summary}", flush=True)
        # Send `history` to both Chatbot and State
        return history, audio, label, history

    if history is None:
        history = []

    # ─── Normalize FileData ─────────────────────
    if isinstance(audio_np, dict) and "path" in audio_np:
        audio_np = audio_np["path"]

    if not (isinstance(audio_np, str) and os.path.exists(audio_np)):
        print(f"❌ [AUDIO] Invalid input: {audio_np!r}")
        history.append({
            "role": "assistant",
            "content": "⚠️ No audio received. Please record again."
        })
        return safe_return(history, (None, None), "Unknown")

    try:
        # Gradio already saved the recording as a wav file, so use it directly
        asr_started_at = time.perf_counter()
        user_input, audio_file_path = transcribe_audio(audio_np, init_data['crisperwhisper_model'])
        record_timing("asr", asr_started_at)

        fluency_started_at = time.perf_counter()
        syll, sr_feats, ar, asd = calculate_all_features(user_input, audio_file_path)
        fl = classify_fluency(init_data['rf_model'], sr_feats, ar, asd)
        fluency_level = int(np.array(fl).item())
        record_timing("fluency_analysis", fluency_started_at)

        vector_collection = [
            init_data.get('beginner_collection'),
            init_data.get('intermediate_collection'),
            init_data.get('advanced_collection')
        ][min(fluency_level, 2)]

        model           = init_data['model']
        tokenizer       = init_data['tokenizer']
        embedding_model = init_data['embedding_model']
        boost_value     = init_data['boost_value']
        device          = init_data['device']

        retrieval_started_at = time.perf_counter()
        boost_words = knn_search(user_input, embedding_model, vector_collection)
        record_timing("retrieval", retrieval_started_at)

        prompt_started_at = time.perf_counter()
        logits_proc = create_boost_processor(tokenizer, boost_words, boost_value)

        fluency_labels = ["Beginner 🟢", "Intermediate 🟡", "Advanced 🔵"]
        label = fluency_labels[fluency_level] if 0 <= fluency_level < len(fluency_labels) else "Unknown"
        print(f"🏷️ [LABEL] {label}")

        system_msg = build_prompt(boost_words, user_input, label, include_turn=False)

        messages = (
            [{"role": "system", "content": system_msg}]
            + [
                {"role": msg['role'], "content": msg['content']}
                for msg in history
                if msg['role'] in ['user', 'assistant']
            ]
            + [{"role": "user", "content": user_input}]
        )
        record_timing("prompt_setup", prompt_started_at)

        generation_started_at = time.perf_counter()
        assistant_response = generate_chat_response(
            model, tokenizer, messages,
            logits_proc, stopping_criteria, device
        )
        record_timing("llm_generation", generation_started_at)
        print(f"🤖 [RESPONSE] {assistant_response[:300]}", flush=True)

        new_history = history + [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": assistant_response},
        ]

        tts_started_at = time.perf_counter()
        tts_bytes = generate_speech_openai(assistant_response, fluency_level, response_format="wav")
        record_timing("tts_request", tts_started_at)
        if not tts_bytes:
            print("❌ [TTS] No audio returned.")
            return safe_return(new_history, (None, None), label)

        tts_decode_started_at = time.perf_counter()
        out_data, out_sr = sf.read(io.BytesIO(tts_bytes), dtype='float32')
        record_timing("tts_decode", tts_decode_started_at)

        return safe_return(new_history, (out_sr, out_data), label)

    except Exception as e:
        print(f"❌ [PROCESS ERROR] {e}")
        history.append({
            "role": "assistant",
            "content": "⚠️ Something went wrong. Please try again."
        })
        return safe_return(history, (None, None), "Unknown")

custom_css = """
#custom-title {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 1rem;
    margin-bottom: 0;
    text-align: center;
}

#avatar-inline {
    width: 64px;
    height: 64px;
    border-radius: 50%;
    object-fit: cover;
}

#custom-title span {
    font-size: 2rem;
    font-weight: 700;
    font-family: 'Segoe UI', 'Inter', 'Helvetica Neue', sans-serif;
    color: white;
    line-height: 1.2;
}

#instructions {
    text-align: center;
    font-size: 1.5rem;
    color: white;
    margin-top: 0.5rem;
    margin-bottom: 2rem;
}

#instructions ol {
    display: inline-block;
    text-align: left;
    padding-left: 1.2rem;
    margin: 0;
}

#instructions li {
    margin-bottom: 0.4rem;
}

.gr-chatbot {
    background-color: #1e1e1e !important;
    color: white;
}

.gr-chat-message {
    background-color: rgba(255, 255, 255, 0.05);
    color: white;
    border-radius: 10px;
    padding: 0.6rem 1rem;
    margin: 0.25rem 0;
}

.gr-chat-message-user {
    font-weight: bold;
    background-color: rgba(0, 123, 255, 0.2);
}

.gr-chat-message-assistant {
    background-color: rgba(255, 255, 255, 0.08);
}
"""


# ------------------------------
# Gradio Interface
# ------------------------------
with gr.Blocks(css=custom_css) as demo:
    gr.HTML("""
        <div id="custom-title">
            <img src="/static/images/AdaptLingoAvatar.png" alt="Avatar" id="avatar-inline">
            <span>Talk with AdaptLingo!</span>
        </div>
        <div id="instructions">
            <ol>
                <li>In the left side of the top box, press "Record" and allow microphone access from your browser.</li>
                <li>Speak into your microphone, and press "Stop" when done.</li>
                <li>Press "Submit Speech," and the chatbot will generate a response in the box below. It may take several seconds.</li>
                <li>You can replay your speech in the top box and the chatbot's speech from the bottom box.</li>
                <li>To send a new message, click the "X" in the top right of the top box, then resume from step 1.</li>
            </ol>
        </div>
    """)

    def update_submit_button_visibility(audio):
        is_valid = bool(audio)
        return is_valid, gr.update(visible=is_valid)

    # States
    audio_valid = gr.State(False)
    history_state = gr.State([])

    # Components
    audio_input = gr.Audio(
        sources=["microphone"],
        label="🎤 Press & Speak",
        format="wav",
        type="filepath"
    )

    submit_button = gr.Button("🎙️ Submit Speech", visible=False)

    fluency_label = gr.Label(
        label="🧠 Fluency Level",
        visible=True,
        value="Waiting for speech..."
    )

    response_text = gr.Chatbot(
        label="💬 AdaptLingo Chat",
        elem_id="response-box",
        type="messages",
        autoscroll=True
    )

    audio_output = gr.Audio(
        label="🔊 AdaptLingo Voice",
        interactive=False,
        autoplay=True,
        type="numpy",
        visible=True
    )

    # Audio input change triggers button reveal and updates audio_valid
    audio_input.change(
        fn=update_submit_button_visibility,
        inputs=[audio_input],
        outputs=[audio_valid, submit_button],
        show_progress=False
    )

    # Submit button click runs the full pipeline
    submit_button.click(
        fn=process_user_audio_openai,
        inputs=[audio_input, history_state],
        outputs=[response_text, audio_output, fluency_label, history_state],
        queue=True
        )

# ------------------------------
# Server Launch
# ------------------------------
if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Launch the AdaptLingo gradio app")
    parser.add_argument("--share", action="store_true", help="Create a public shareable link using ngrok")
    args = parser.parse_args()

    # Mount the Gradio interface on the FastAPI server
    app = mount_gradio_app(app, demo, path="/")
    port = 7860

    if args.share:
        # Open an ngrok tunnel on the desired port
        public_url = ngrok.connect(port)
        print("Public URL:", public_url)
        # Launch the app using uvicorn
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    else:
        uvicorn.run(app, host="127.0.0.1", port=port)
