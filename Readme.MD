# Chatbot Interface and API

This repository contains a Flask-based chatbot with a web interface and API. The application supports conversational interactions, text-to-speech (TTS), and dynamic reloading of configurations, such as model settings and embedding modifications.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
  - [POST /chat](#post-chat)
  - [GET /get_config](#get-get_config)
  - [POST /update_config](#post-update_config)
- [Configuration](#configuration)
- [Run the Application](#run-the-application)

## Features
- Conversational chatbot with text-to-speech support.
- Dynamic configuration management with real-time reloading.
- Supports embedding-based search using K-Nearest Neighbors (KNN) with a user-defined boost value.
- Simple HTML-based frontend for chatting with the assistant.
- Speech recognition for inputs.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/chatbot-interface-api.git
    cd chatbot-interface-api
    ```

2. (Recommended) Use a Python virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Set up environment variables:
    - Make sure to set a secure Flask secret key.
    - Optionally, configure `CUDA_VISIBLE_DEVICES` if using a GPU:
      ```bash
      export CUDA_VISIBLE_DEVICES=0
      ```

4. Configure your models and embeddings in `config.json`.

## Usage

1. **Start the Flask application**:
    ```bash
    python app.py
    ```

2. **Access the chatbot**:
    - Visit `http://localhost:55556/` in your web browser to chat with the assistant via the web interface.

3. **Interact via API**:
    - You can also interact with the chatbot and update its configuration via the provided API endpoints (described below).

## API Endpoints

### POST /chat

This endpoint processes user input and returns a chatbot response, optionally generating an audio response using TTS.

- **URL**: `/chat`
- **Method**: `POST`
- **Request Format**: JSON or Form Data
- **Request Parameters**:
    - `message` (string): The user's input to the chatbot.
- **Response**:
    - `message` (string): The assistant's response.
    - `audio_url` (string): A URL to the generated audio file (if TTS is enabled).

#### Example Request (JSON):

```bash
curl -X POST http://localhost:55556/chat -H "Content-Type: application/json" -d '{"message": "Hello!"}'
```

## Common Error Solutions

### 1. Ensure Praat is Installed
The Praat application must be installed on your system. Update the `praat_executable` path in `audio.py` accordingly. If you are using a Columbia machine, you may need to build Praat from source.

#### Steps to Build Praat from Source:
1. Clone the Praat repository:
   ```bash
   git clone https://github.com/praat/praat.git
   ```
2. Install the required dependencies:
   ```bash
   conda install cmake gcc gxx gtk3 alsa-lib -c conda-forge
   conda install gcc gxx cmake -c conda-forge
   ```
3. Navigate to the cloned repository:
   ```bash
   cd praat
   ```
4. Create a build directory:
   ```bash
   mkdir build
   cd build
   ```
5. Configure the build:
   ```bash
   cmake ..
   ```
6. Compile the source code:
   ```bash
   make -j$(nproc)
   ```

### 2. Ensure FFmpeg is Installed
Make sure FFmpeg is installed on your system. You can install it using a package manager like `apt` (Linux), `brew` (macOS), or from the [FFmpeg official website](https://ffmpeg.org/download.html).

## Todo:
- Dynamically delete generated audio after next message (right now it's staying in the cache)
- Separate word lists into levels
- Retrieve words based on user level
- Judge user level with pratt
- Judge user level with grammar
- Train English TTS Model for Style-BERT-VITS2
- Test whether audio input features work -- Over browser HTTPS or other library implementation, we can test it temporarily with premade audio files.