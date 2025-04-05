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

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

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

# ------------------------------
# Main processing function
# ------------------------------
def process_user_audio(audio_np, history):
    global init_data

    if audio_np is None or not isinstance(audio_np, tuple):
        return [("System", "No audio received.")], None, "Unknown", history

    sample_rate, audio_array = audio_np

    # Save audio temporarily for processing
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        sf.write(tmpfile.name, audio_array, sample_rate)
        tmp_path = tmpfile.name

    try:
        whisper_model = init_data['crisperwhisper_pipe']
        forest_classifier = init_data['rf_model']

        user_input, audio_file_path = transcribe_audio(tmp_path, whisper_model)

        syll, sr, ar, asd = calculate_all_features(user_input, audio_file_path)
        fluency_level = classify_fluency(forest_classifier, sr, ar, asd)

        if fluency_level == 0:
            vector_collection = init_data.get('beginner_collection', None)
        elif fluency_level == 1:
            vector_collection = init_data.get('intermediate_collection', None)
        else:
            vector_collection = init_data.get('advanced_collection', None)

        model = init_data['model']
        tokenizer = init_data['tokenizer']
        tts_model = init_data['tts_model']
        embedding_model = init_data['embedding_model']
        essential_words = init_data['essential_words']
        boost_value = init_data['boost_value']
        device = init_data['device']

        boost_words = knn_search(user_input, embedding_model, vector_collection) + essential_words
        logits_processor = create_boost_processor(tokenizer, boost_words, boost_value)
        stopping_criteria = create_stopping_criteria(tokenizer)

        system_message = build_prompt(boost_words, user_input)
        text_history = "\n".join([
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in history if msg['role'] in ['user', 'assistant']
        ])
        prompt = f"{system_message}\n{text_history}\nUser: {user_input}\nAssistant:"

        assistant_response = generate_response(
            model, tokenizer, prompt, logits_processor, stopping_criteria, device
        )

        # Append to history
        if not isinstance(history, list):
            history = []
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": assistant_response})

        # TTS output
        tts_path = generate_speech_from_text(tts_model, assistant_response)
        if not tts_path or not os.path.exists(tts_path):
            return history, None, fluency_level, history

        with open(tts_path, "rb") as f:
            audio_bytes = f.read()
        os.remove(tts_path)
        audio_output_np = np.frombuffer(audio_bytes, dtype=np.int16)

        fluency_levels = ["Beginner 🟢", "Intermediate 🟡", "Advanced 🔵"]
        if hasattr(fluency_level, "item"):
            fluency_level = fluency_level.item()
        label = fluency_levels[fluency_level] if isinstance(fluency_level, int) and fluency_level < len(fluency_levels) else "Unknown"

        return history, (44100, audio_output_np), label, history

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# ------------------------------
# CSS Styling
# ------------------------------
custom_css = """
#custom-title {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 1rem;
    margin-bottom: 2rem;
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

#response-box {
    font-size: 1.1rem;
    line-height: 1.6;
    max-height: 300px;
    overflow-y: auto;
    scroll-behavior: smooth;
    margin-bottom: 2rem;
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
    """)

    audio_input = gr.Audio(
        sources=["microphone"],
        label="🎤 Press & Speak",
        format="wav"
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
        type="filepath",
        visible=True,
        value="static/audio/example.wav"
    )

    history_state = gr.State(value=[])

    submit_button.click(
        fn=process_user_audio,
        inputs=[audio_input, history_state],
        outputs=[response_text, audio_output, fluency_label, history_state]
    ).then(
        fn=lambda: None,
        inputs=None,
        outputs=[audio_input]
    )

# ------------------------------
# Server Launch
# ------------------------------
if __name__ == "__main__":
    import argparse
    import uvicorn
    from gradio import mount_gradio_app
    from pyngrok import ngrok

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
        uvicorn.run(app, host="127.0.0.1", port=port)
    else:
        uvicorn.run(app, host="127.0.0.1", port=port)
