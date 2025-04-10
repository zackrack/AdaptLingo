# initialize.py
import os
import json
import nltk
import chromadb
import torch 
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download

from helpers import (
    load_llm_model, load_bert_model, load_tts_model,
    read_words_file, get_or_create_collection, load_crisper_model, load_rf_model
)

def initialize():
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    print("Config loaded.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_name = config.get('model_name')
    beginner_words_filename = config.get('beginner_words_filename')
    intermediate_words_filename = config.get('intermediate_words_filename')
    advanced_words_filename = config.get('advanced_words_filename')
    essential_words_filename = config.get('essential_words_filename')
    boost_value = config.get('boost_value')
    sentencetransformer_model_name = config.get('sentencetransformer_model')
    bert_models_config = config.get('bert_models')
    tts_model_config = config.get('tts_model')
    rf_model = config.get("classifier_model")

    nltk_data_path = os.path.expanduser('~/nltk_data')
    if not os.path.exists(nltk_data_path):
        os.makedirs(nltk_data_path)
        print(f"Created NLTK data directory at {nltk_data_path}")
    nltk.data.path.append(nltk_data_path)
    print("NLTK data path set.")

    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_path)
        print("Downloaded averaged_perceptron_tagger.")
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    except LookupError:
        nltk.download('averaged_perceptron_tagger_eng', download_dir=nltk_data_path)
        print("Downloaded averaged_perceptron_tagger_eng.")
    try:
        nltk.data.find('corpora/cmudict')
    except LookupError:
        nltk.download('cmudict', download_dir=nltk_data_path)
        print("Downloaded cmudict.")

    client = chromadb.PersistentClient(path="chroma_store")

    print("ChromaDB client initialized.")

    model, tokenizer = load_llm_model(model_name)
    model.to(device)
    print("Chatbot model loaded.")

    load_bert_model(bert_models_config)
    print("BERT models loaded.")

    tts_model = load_tts_model(tts_model_config)
    print("TTS model loaded.")

    beginner_words = read_words_file(beginner_words_filename)
    intermediate_words = read_words_file(intermediate_words_filename)
    advanced_words = read_words_file(advanced_words_filename)
    essential_words = read_words_file(essential_words_filename)
    print("Word files read for beginner, intermediate, and advanced.")

    try:
        embedding_model = SentenceTransformer(sentencetransformer_model_name, device=device)
        print("SentenceTransformer model loaded.")
    except Exception as e:
        print(f"Error loading SentenceTransformer model '{sentencetransformer_model_name}': {e}")
        raise e

    beginner_words = list(set(beginner_words))
    intermediate_words = list(set(intermediate_words))
    advanced_words = list(set(advanced_words))
    print("Duplicates removed from word lists.")

    beginner_embeddings = embedding_model.encode(beginner_words, convert_to_tensor=True)
    beginner_collection = get_or_create_collection(client, "beginner_embeddings", beginner_words, beginner_embeddings)
    print("Beginner collection created.")

    intermediate_embeddings = embedding_model.encode(intermediate_words, convert_to_tensor=True)
    intermediate_collection = get_or_create_collection(client, "intermediate_embeddings", intermediate_words, intermediate_embeddings)
    print("Intermediate collection created.")

    advanced_embeddings = embedding_model.encode(advanced_words, convert_to_tensor=True)
    advanced_collection = get_or_create_collection(client, "advanced_embeddings", advanced_words, advanced_embeddings)
    print("Advanced collection created.")

    crisperwhisper_model, crisperwhisper_processor = load_crisper_model()
    print("Crisper Whisper pipeline loaded.")

    clf = load_rf_model(rf_model)
    print("Classifier model loaded.")

    audio_dir = os.path.join("static", "audio")
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)
        print(f"Created audio directory at {audio_dir}")

    print("Initialization complete.")
    return {
        'model': model,
        'tokenizer': tokenizer,
        'tts_model': tts_model,
        'embedding_model': embedding_model,
        'boost_value': boost_value,
        'device': device,
        'crisperwhisper_model': crisperwhisper_model,
        'crisperwhisper_processor': crisperwhisper_processor,
        'rf_model': clf,
        'beginner_collection': beginner_collection,
        'intermediate_collection': intermediate_collection,
        'advanced_collection': advanced_collection,
        'essential_words': essential_words,
    }


if __name__ == "__main__":
    initialize()