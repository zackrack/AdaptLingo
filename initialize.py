# initialize.py
import os
import json
import nltk
import chromadb
import torch  # Ensure torch is imported
from sentence_transformers import SentenceTransformer
from helpers import (
    load_llm_model, load_bert_model, load_tts_model,
    read_words_file, get_or_create_collection
)

def initialize():
    # Load configuration from config.json
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    print("Config loaded successfully.")
    
    # Extract configuration settings
    model_name = config.get('model_name')
    word_list_filename = config.get('word_list_filename')
    essential_words_filename = config.get('essential_words_filename')
    boost_value = config.get('boost_value')
    sentencetransformer_model_name = config.get('sentencetransformer_model')
    bert_models_config = config.get('bert_models')
    tts_model_config = config.get('tts_model')

    # Define a directory to download the NLTK data to
    nltk_data_path = os.path.expanduser('~/nltk_data')
    if not os.path.exists(nltk_data_path):
        os.makedirs(nltk_data_path)

    nltk.data.path.append(nltk_data_path)

    # Download the required NLTK resource
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_path)

    # Initialize ChromaDB client
    client = chromadb.Client()

    # Load the main chatbot model
    model, tokenizer = load_llm_model(model_name)

    # **Add device configuration here**
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the BERT models for NLP tasks
    load_bert_model(bert_models_config)

    # Load the TTS model
    tts_model = load_tts_model(tts_model_config)

    # Read the word lists
    words = read_words_file(word_list_filename)
    essential_words = read_words_file(essential_words_filename)

    # Embed words using the specified SentenceTransformer model
    try:
        embedding_model = SentenceTransformer(sentencetransformer_model_name, device=device)
    except Exception as e:
        print(f"Error loading SentenceTransformer model '{sentencetransformer_model_name}': {e}")
        raise e

    words = list(set(words))
    word_embeddings = embedding_model.encode(words, convert_to_tensor=True)

    # Create or get the collection for word embeddings
    collection = get_or_create_collection(client, "word_embeddings", words, word_embeddings)

    print("Initialized successfully.")
    # Return necessary variables as a dictionary
    return {
        'model': model,
        'tokenizer': tokenizer,
        'tts_model': tts_model,
        'embedding_model': embedding_model,
        'collection': collection,
        'essential_words': essential_words,
        'boost_value': boost_value,
        'device': device  # Include device if needed elsewhere
    }
