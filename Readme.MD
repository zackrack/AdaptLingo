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

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/chatbot-interface-api.git
    cd chatbot-interface-api
    ```

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up environment variables:
    - Make sure to set a secure Flask secret key.
    - Optionally, configure `CUDA_VISIBLE_DEVICES` if using a GPU:
      ```bash
      export CUDA_VISIBLE_DEVICES=0  # Use GPU 0
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
