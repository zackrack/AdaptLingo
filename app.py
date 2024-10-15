from flask import Flask, render_template, request, jsonify, session, send_from_directory, url_for
from flask_session import Session
from flask_cors import CORS  # Enable CORS if needed
import os
from threading import Thread, Lock
from helpers import (
    generate_speech_from_text,
    create_boost_processor,
    create_stopping_criteria,
    knn_search,
    build_prompt,
    generate_response,
    load_config,
    update_config,
    load_initial_data,
    reinitialize_models
)
from initialize import initialize

app = Flask(__name__)
app.secret_key = 'your_secure_secret_key'  # Replace with a secure secret key in production
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)
# CORS(app)  # Uncomment this if CORS is needed for external API calls

# Global variables to hold models and configurations
init_data = {}
config_lock = Lock()  # To ensure thread safety during re-initialization

def load_app_data():
    global init_data
    with config_lock:
        # Reload the configuration from the file
        init_data = load_initial_data(initialize)
        # Print the configuration values for debugging
        print("Loaded Configuration:")
        # print(init_data)

# Load initial data
load_app_data()
print("If you are on a Columbia CS machine, remember to set CUDA_VISIBLE_DEVICES to a single GPU.")

@app.route('/')
def index():
    # Initialize conversation history in session if not already
    if 'conversation_history' not in session:
        session['conversation_history'] = ""
    return render_template('index.html')

# Combined chat endpoint for both HTML interface and API
@app.route('/chat', methods=['POST'])
def chat():
    # Check if the request contains JSON data
    if request.is_json:
        data = request.get_json()
        user_input = data.get('message', '').strip()
    else:
        # Assume form data
        user_input = request.form.get('message', '').strip()

    if not user_input:
        return jsonify({'message': 'No message provided.', 'audio_url': ''}), 400

    if user_input.lower() == "exit":
        session.pop('conversation_history', None)
        return jsonify({'message': 'Conversation ended.', 'audio_url': ''})

    # Use global init_data
    global init_data
    model = init_data['model']
    tokenizer = init_data['tokenizer']
    tts_model = init_data['tts_model']
    embedding_model = init_data['embedding_model']
    collection = init_data['collection']
    essential_words = init_data['essential_words']
    boost_value = init_data['boost_value']
    device = init_data['device']

    # Device configuration
    device = next(model.parameters()).device

    # Process the user input and generate a response
    boost_words = knn_search(user_input, embedding_model, collection) + essential_words
    logits_processor = create_boost_processor(tokenizer, boost_words, boost_value)
    stopping_criteria = create_stopping_criteria(tokenizer)

    # Build the prompt using conversation history
    system_message = build_prompt(boost_words, user_input)
    conversation_history = session.get('conversation_history', "")
    prompt = f"{system_message}\n{conversation_history}\nUser: {user_input}\nAssistant:"

    # Generate the assistant's response
    assistant_response = generate_response(
        model, tokenizer, prompt, logits_processor, stopping_criteria, device
    )

    # Append the user input and assistant response to the conversation history
    conversation_history += f"User: {user_input}\nAssistant: {assistant_response}\n"
    session['conversation_history'] = conversation_history

    # Generate speech from the assistant's response using the pre-loaded TTS model
    audio_file_path = generate_speech_from_text(tts_model, assistant_response)

    if audio_file_path:
        # Get the audio file name from the path
        audio_file_name = os.path.basename(audio_file_path)
        # Build the audio URL
        audio_url = url_for('audio', filename=audio_file_name)
    else:
        audio_url = ''

    # Return the assistant's response and the audio URL
    return jsonify({'message': assistant_response, 'audio_url': audio_url})

# Serve audio files
@app.route('/audio/<path:filename>')
def audio(filename):
    return send_from_directory('static/audio', filename)

# Get configuration
@app.route('/get_config', methods=['GET'])
def get_config_route():
    try:
        config = load_config()
        return jsonify(config)
    except Exception as e:
        print(f"Error reading config.json: {e}")
        return jsonify({'message': 'Error reading configuration.'}), 500

@app.route('/update_config', methods=['POST'])
def update_config_route():
    new_config = request.get_json()
    if not new_config:
        return jsonify({'message': 'No configuration data provided.'}), 400

    try:
        # Optional: Convert numeric strings to actual numbers if necessary
        if 'boost_value' in new_config:
            new_config['boost_value'] = float(new_config['boost_value'])

        # Update the configuration file
        update_config(new_config)

        # Reload the updated configuration from the file
        load_app_data()  # This will reload init_data with the updated configuration

        # Start reinitialization in a background thread
        reinit_thread = Thread(target=reinitialize_models, args=(initialize,))
        reinit_thread.start()

        return jsonify({'message': 'Settings updated successfully. Models are reinitializing.'})
    except Exception as e:
        print(f"Error updating configuration: {e}")
        return jsonify({'message': 'Error updating settings.'}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=55556, debug=True)
