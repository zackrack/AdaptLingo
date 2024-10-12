from flask import Flask, render_template, request, jsonify, session, send_from_directory, url_for
from flask_session import Session
import os
import time
import threading  # For thread safety
import json
from helpers import (
    generate_speech_from_text, 
    create_boost_processor, 
    create_stopping_criteria, 
    knn_search, 
    build_prompt, 
    generate_response
)
from initialize import initialize

app = Flask(__name__)
app.secret_key = 'your_secure_secret_key'  # Replace with a secure secret key in production
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Global variables to hold models and configurations
init_data = {}
config_lock = threading.Lock()  # To ensure thread safety during re-initialization

def load_initial_data():
    global init_data
    with config_lock:
        init_data = initialize()

# Load initial data
load_initial_data()

@app.route('/')
def index():
    # Initialize conversation history in session if not already
    if 'conversation_history' not in session:
        session['conversation_history'] = ""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['message']

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
    assistant_response = generate_response(model, tokenizer, prompt, logits_processor, stopping_criteria, device)

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

@app.route('/audio/<path:filename>')
def audio(filename):
    return send_from_directory('static/audio', filename)

@app.route('/get_config', methods=['GET'])
def get_config():
    try:
        with open('config.json', 'r') as config_file:
            config = json.load(config_file)
        return jsonify(config)
    except Exception as e:
        print(f"Error reading config.json: {e}")
        return jsonify({'message': 'Error reading configuration.'}), 500

@app.route('/update_config', methods=['POST'])
def update_config():
    new_config = request.get_json()
    if not new_config:
        return jsonify({'message': 'No configuration data provided.'}), 400

    # Validate new_config here as needed
    required_keys = ["model_name", "word_list_filename", "essential_words_filename", 
                     "boost_value", "sentencetransformer_model", "bert_models", 
                     "tts_model", "tts_device"]
    for key in required_keys:
        if key not in new_config:
            return jsonify({'message': f'Missing configuration key: {key}'}), 400

    try:
        with config_lock:
            # Update config.json
            with open('config.json', 'w') as config_file:
                json.dump(new_config, config_file, indent=4)
            # Re-initialize models and configurations
            load_initial_data()
        return jsonify({'message': 'Settings updated successfully. Models re-initialized.'})
    except Exception as e:
        print(f"Error updating configuration: {e}")
        return jsonify({'message': 'Error updating settings.'}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=55556, debug=True)