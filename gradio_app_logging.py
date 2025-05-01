import gradio as gr
import os
import numpy as np
from threading import Lock
import tempfile
import soundfile as sf
import argparse
from helpers import (
    generate_speech_from_text,
    create_boost_processor,
    create_stopping_criteria,
    knn_search,
    build_prompt,
    generate_response,
    load_initial_data,
    transcribe_audio,
    classify_fluency
)
from initialize import initialize
from calculate_features import calculate_all_features
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn
import uuid
from logger import log_interaction
import os
import tempfile
import numpy as np
import soundfile as sf
import openai
from datetime import datetime
from gradio import mount_gradio_app
from pyngrok import ngrok

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'

# ------------------------------
# Global initialization and state
# ------------------------------
init_data = {}
config_lock = Lock()

def load_app_data():
    global init_data
    with config_lock:
        init_data = load_initial_data(initialize)
        print("Loaded Configuration")

load_app_data()

client = openai.OpenAI(api_key="")


def fluency_to_speed(fluency_level):
    if isinstance(fluency_level, int):
        if fluency_level == 0:
            return 0.8
        elif fluency_level == 1:
            return 0.9
    return 1.0
    
# Sage or shimmer are the best
def generate_speech_openai(text, fluency_level, voice="sage", model="tts-1", response_format="wav"):
    """
    Calls OpenAI TTS and returns raw bytes in WAV format.
    """
    try:
        response = client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            response_format=response_format,
            speed=fluency_to_speed(fluency_level)
        )
        return response.content
    except Exception as e:
        print(f"[TTS ERROR] {e}")
        return None

def process_user_audio_openai(audio_np, history):
    global init_data

    print("📥 [START] Submit clicked.")
    print(f"[DEBUG] raw audio_np: {audio_np!r}")

    def safe_return(h, audio, label):
        if not isinstance(audio, tuple) or len(audio) != 2:
            print("⚠️ [SAFE_RETURN] Audio output was not a tuple. Fixing.")
            audio = (None, None)

        sr, data = audio
        if sr is None or data is None:
            print("⚠️ [SAFE_RETURN] audio contained None, replacing with silent fallback.")
            sr = 24000
            data = np.zeros(1, dtype=np.float32)
            audio = (sr, data)

        print(f"🧪 [SAFE_RETURN] Returning: sample_rate={sr}, array_shape={data.shape}")
        return h, audio, label, h


    # ─── Normalize FileData ─────────────────────
    if isinstance(audio_np, dict) and "path" in audio_np:
        audio_np = audio_np["path"]

    if not (isinstance(audio_np, str) and os.path.exists(audio_np)):
        print(f"❌ [AUDIO] Invalid input: {audio_np!r}")
        history = history or []
        history.append({
            "role": "assistant",
            "content": "⚠️ No audio received. Please record again."
        })
        return safe_return(history, (None, None), "Unknown")

    try:
        data, sr = sf.read(audio_np)
    except Exception as e:
        print(f"❌ [AUDIO] Read error: {e}")
        history = history or []
        history.append({
            "role": "assistant",
            "content": "⚠️ Could not read your recording. Try again."
        })
        return safe_return(history, (None, None), "Unknown")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = "saved_user_audio"
    os.makedirs(save_dir, exist_ok=True)
    tmp_path = os.path.join(save_dir, f"user_input_{timestamp}.wav")

    try:
        sf.write(tmp_path, data, sr)
        print(f"💾 [SAVED] Audio saved to: {tmp_path}")

        cr_model = init_data['crisperwhisper_model']
        processor = init_data['crisperwhisper_processor']
        user_input, audio_file_path = transcribe_audio(tmp_path, cr_model, processor)
        print(f"📝 [TRANSCRIPTION] Text: '{user_input}'")

        syll, sr_feats, ar, asd = calculate_all_features(user_input, audio_file_path)
        fl = classify_fluency(init_data['rf_model'], sr_feats, ar, asd)
        fluency_level = int(np.array(fl).item())
        print(f"📊 [FLUENCY] Level: {fluency_level}")

        vector_collection = [
            init_data.get('beginner_collection'),
            init_data.get('intermediate_collection'),
            init_data.get('advanced_collection')
        ][min(fluency_level, 2)]

        model           = init_data['model']
        tokenizer       = init_data['tokenizer']
        embedding_model = init_data['embedding_model']
        essential_words = init_data['essential_words']
        boost_value     = init_data['boost_value']
        device          = init_data['device']

        boost_words = knn_search(user_input, embedding_model, vector_collection) + essential_words
        logits_proc   = create_boost_processor(tokenizer, boost_words, boost_value)
        stopping_crit = create_stopping_criteria(tokenizer)
        system_msg    = build_prompt(boost_words, user_input)

        text_history = "\n".join([
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in (history or [])
            if msg['role'] in ['user', 'assistant']
        ])
        prompt = (
            f"{system_msg}\n"
            f"{text_history}\n"
            f"User: {user_input}\n"
            "Assistant:"
        )

        assistant_response = generate_response(
            model, tokenizer, prompt,
            logits_proc, stopping_crit, device
        )
        print(f"🤖 [RESPONSE] {assistant_response[:80]}...")

        history = history or []
        history.append({"role": "user",      "content": user_input})
        history.append({"role": "assistant", "content": assistant_response})

        tts_bytes = generate_speech_openai(assistant_response, fluency_level, response_format="wav")
        if not tts_bytes:
            print("❌ [TTS] No audio returned.")
            return safe_return(history, (None, None), fluency_level)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tts_file:
            tts_file.write(tts_bytes)
            tts_file.flush()
            os.fsync(tts_file.fileno())
            tts_path = tts_file.name

        out_data, out_sr = sf.read(tts_path, dtype='float32')
        os.remove(tts_path)

        fluency_labels = ["Beginner 🟢", "Intermediate 🟡", "Advanced 🔵"]
        label = fluency_labels[fluency_level] if 0 <= fluency_level < len(fluency_labels) else "Unknown"
        print(f"🏷️ [LABEL] {label}")

        log_interaction(
            session_id="default-session",
            user_input=user_input,
            assistant_response=assistant_response,
            fluency_level=label,
            audio_path=tmp_path
        )

        print("✅ [DONE]")
        return safe_return(history, (out_sr, out_data), label)

    except Exception as e:
        print(f"❌ [PROCESS ERROR] {e}")
        history = history or []
        history.append({
            "role": "assistant",
            "content": "⚠️ Something went wrong. Please try again."
        })
        return safe_return(history, (None, None), "Unknown")

    finally:
        print("🧹 [CLEANUP] Done processing audio")


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
                <li>In the top box, press "record" and allow microphone access.</li>
                <li>Speak into the microphone.</li>
                <li>Press "Submit," and the chatbot will generate a response in the box below.</li>
                <li>You can replay the speech in the bottom box.</li>
            </ol>
        </div>
    """)


    audio_input = gr.Audio(
        sources=["microphone"],
        label="🎤 Press & Speak",
        format="wav",
        type="filepath",
    )
    submit_button = gr.Button("🎙️ Submit Speech")

    fluency_label = gr.Label(
        label="🧠 Fluency Level",
        visible=False,
        value="Waiting for speech..."
    )

    response_text = gr.Chatbot(
        label="💬 AdaptLingo Chat",
        elem_id="response-box",
        type="messages"
    )


    audio_output = gr.Audio(
        label="🔊 AdaptLingo Voice",
        interactive=False,
        autoplay=True,
        type="numpy",
        visible=True,
    )


    history_state = gr.State(value=[])

    audio_valid = gr.State(False)

    audio_input.change(
        fn=lambda audio: bool(audio),
        inputs=[audio_input],
        outputs=[audio_valid]
    )

    submit_button.click(
        fn=process_user_audio_openai,
        inputs=[audio_input, history_state],
        outputs=[response_text, audio_output, fluency_label, history_state],
        preprocess=False,
        queue=True,
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
