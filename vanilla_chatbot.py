import gradio as gr
import os
import tempfile
import soundfile as sf
import argparse
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn
from gradio import mount_gradio_app
from pyngrok import ngrok
import openai
from datetime import datetime
import numpy as np
import time

from logger import log_interaction  # same logger as before

# ------------------------------
# FastAPI + static
# ------------------------------
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=openai.api_key)

# ------------------------------
# Prolific ID capture
# ------------------------------
CURRENT_PROLIFIC_ID = "UNKNOWN"

def capture_prolific_id(request: gr.Request):
    global CURRENT_PROLIFIC_ID
    pid = request.query_params.get("PROLIFIC_PID", "UNKNOWN")
    CURRENT_PROLIFIC_ID = pid
    print(f"✅ Prolific ID captured: {pid}")
    return f"Prolific ID: {pid}"

def log_with_pid(user_input, assistant_response, audio_path):
    log_interaction(
        session_id=CURRENT_PROLIFIC_ID,
        user_input=user_input,
        assistant_response=assistant_response,
        fluency_level="N/A",   # no classifier here
        audio_path=audio_path
    )

# ------------------------------
# OpenAI Chat + TTS
# ------------------------------
def generate_chat_response(user_text, history):
    """
    Vanilla OpenAI chat completion.
    """
    try:
        messages = [{"role": "system", "content": "You are a helpful, safe, friendly chatbot."}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_text})

        response = client.chat.completions.create(
            model="gpt-5",   # lightweight but solid
            messages=messages,
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ Chat error: {e}")
        return "⚠️ Sorry, something went wrong."

def generate_speech_openai(text, voice="nova", model="gpt-4o-mini-tts", response_format="wav"):
    try:
        response = client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            response_format=response_format
        )
        return response.content
    except Exception as e:
        print(f"❌ TTS error: {e}")
        return None

# ------------------------------
# Processing pipeline
# ------------------------------
def process_user_audio_control(audio_np, history):
    global CURRENT_PROLIFIC_ID

    if isinstance(audio_np, dict) and "path" in audio_np:
        audio_np = audio_np["path"]

    if not (isinstance(audio_np, str) and os.path.exists(audio_np)):
        return history or [], (None, None), "Unknown", history or []

    try:
        data, sr = sf.read(audio_np)
    except Exception as e:
        print(f"❌ [AUDIO] Read error: {e}")
        return history or [], (None, None), "Unknown", history or []

    # Save raw user audio with Prolific ID prefix
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = "saved_user_audio"
    os.makedirs(save_dir, exist_ok=True)
    tmp_path = os.path.join(save_dir, f"{CURRENT_PROLIFIC_ID}_user_input_{timestamp}.wav")
    sf.write(tmp_path, data, sr)

    # Fake "transcription" since we don’t need Whisper here
    # If you want speech input only, plug Whisper in; otherwise assume it's text
    user_input = f"[Voice input saved: {os.path.basename(tmp_path)}]"

    # Get assistant reply
    assistant_response = generate_chat_response(user_input, history or [])

    # Update history
    new_history = (history or []) + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": assistant_response},
    ]

    # TTS
    tts_bytes = generate_speech_openai(assistant_response, response_format="wav")
    if not tts_bytes:
        return new_history, (None, None), "N/A", new_history

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tts_file:
        tts_file.write(tts_bytes)
        tts_file.flush()
        os.fsync(tts_file.fileno())
        tts_path = tts_file.name

    out_data, out_sr = sf.read(tts_path, dtype='float32')
    os.remove(tts_path)

    # Logging
    log_with_pid(user_input, assistant_response, tmp_path)

    return new_history, (out_sr, out_data), "N/A", new_history

# ------------------------------
# UI (same as your main bot)
# ------------------------------
custom_css = """/* keep your same CSS here */"""

with gr.Blocks(css=custom_css) as demo:
    gr.HTML("""
        <div id="custom-title">
            <img src="/static/images/AdaptLingoAvatar.png" alt="Avatar" id="avatar-inline">
            <span>Talk with Bot B</span>
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

    pid_banner = gr.Markdown("Waiting for Prolific ID...")

    demo.load(
        fn=capture_prolific_id,
        inputs=None,
        outputs=[pid_banner]
    )

    audio_valid = gr.State(False)
    history_state = gr.State([])

    audio_input = gr.Audio(
        sources=["microphone"],
        label="🎤 Press & Speak",
        format="wav",
        type="filepath"
    )

    submit_button = gr.Button("🎙️ Submit Speech", visible=False)

    fluency_label = gr.Label(
        label="🧠 Fluency Level",
        visible=False,
        value="N/A (control bot)"
    )

    response_text = gr.Chatbot(
        label="💬 Control Chatbot",
        elem_id="response-box",
        type="messages",
        autoscroll=True
    )

    audio_output = gr.Audio(
        label="🔊 Control Bot Voice",
        interactive=False,
        autoplay=True,
        type="numpy",
        visible=True
    )

    def update_submit_button_visibility(audio):
        is_valid = bool(audio)
        return is_valid, gr.update(visible=is_valid)

    audio_input.change(
        fn=update_submit_button_visibility,
        inputs=[audio_input],
        outputs=[audio_valid, submit_button],
        show_progress=False
    )

    submit_button.click(
        fn=process_user_audio_control,
        inputs=[audio_input, history_state],
        outputs=[response_text, audio_output, fluency_label, history_state],
        queue=True
    )

# ------------------------------
# Server Launch
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the control Gradio app")
    parser.add_argument("--share", action="store_true", help="Create a public shareable link using ngrok")
    args = parser.parse_args()

    app = mount_gradio_app(app, demo, path="/")
    port = 7861  # different port so it can run alongside the main bot

    if args.share:
        public_url = ngrok.connect(port)
        print("Public URL:", public_url)
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    else:
        uvicorn.run(app, host="127.0.0.1", port=port)
