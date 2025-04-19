# initialize.py

import os
import json
import nltk
import chromadb
import torch
from sentence_transformers import SentenceTransformer

from helpers import (
    load_llm_model,
    load_bert_model,
    load_tts_model,
    read_words_file,
    get_or_create_collection,
    load_crisper_model,
    load_rf_model,
)
from helpers.generation import RAGConversationSystem


def initialize():
    # ——— Load config
    with open("config.json", "r") as config_file:
        config = json.load(config_file)
    print("Config loaded.")

    # ——— Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ——— Config fields
    model_name = config.get("model_name")
    sentencetransformer_model_name = config.get("sentencetransformer_model")
    max_history = config.get("max_history", 20)
    beginner_words_filename = config.get("beginner_words_filename")
    intermediate_words_filename = config.get("intermediate_words_filename")
    advanced_words_filename = config.get("advanced_words_filename")
    essential_words_filename = config.get("essential_words_filename")
    boost_value = config.get("boost_value")
    bert_models_config = config.get("bert_models")
    tts_model_config = config.get("tts_model")
    rf_model = config.get("classifier_model")

    # ——— NLTK setup
    nltk_data_path = os.path.expanduser("~/nltk_data")
    os.makedirs(nltk_data_path, exist_ok=True)
    nltk.data.path.append(nltk_data_path)
    for resource in [
        "taggers/averaged_perceptron_tagger",
        "taggers/averaged_perceptron_tagger_eng",
        "corpora/cmudict",
    ]:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource.split("/")[-1], download_dir=nltk_data_path)

    # ——— ChromaDB client
    client = chromadb.PersistentClient(path="chroma_store")
    print("ChromaDB client initialized.")

    # ——— Load HF LLM + tokenizer
    hf_model, tokenizer = load_llm_model(model_name)
    hf_model.to(device)
    print("Chatbot HF model & tokenizer loaded.")

    # ——— Wrap in RAG system
    rag_system = RAGConversationSystem(
        embedding_model_name=sentencetransformer_model_name,
        tokenizer=tokenizer,
        llm_model=hf_model,
        max_history=max_history,
    )
    rag_system.load_history()
    print("RAGConversationSystem initialized.")

    # ——— Load BERT / TTS / RF
    load_bert_model(bert_models_config)
    print("BERT models loaded.")

    tts_model = load_tts_model(tts_model_config)
    print("TTS model loaded.")

    crisperwhisper_pipe, crisperwhisper_processor = load_crisper_model()
    print("Crisper Whisper pipeline loaded.")

    clf = load_rf_model(rf_model)
    print("Classifier model loaded.")

    # ——— Word lists + embeddings
    beginner_words = list(set(read_words_file(beginner_words_filename)))
    intermediate_words = list(set(read_words_file(intermediate_words_filename)))
    advanced_words = list(set(read_words_file(advanced_words_filename)))
    essential_words = read_words_file(essential_words_filename)
    print("Word files read and deduped.")

    embedding_model = SentenceTransformer(sentencetransformer_model_name, device=device)
    print("SentenceTransformer model loaded.")

    # ——— Create Chroma collections
    for name, words in [
        ("beginner", beginner_words),
        ("intermediate", intermediate_words),
        ("advanced", advanced_words),
    ]:
        embs = embedding_model.encode(words, convert_to_tensor=True)
        _ = get_or_create_collection(client, f"{name}_embeddings", words, embs)
        print(f"{name.capitalize()} collection created.")

    # ——— Ensure audio directory
    audio_dir = os.path.join("static", "audio")
    os.makedirs(audio_dir, exist_ok=True)
    print("Initialization complete.")

    return {
        "model": rag_system,
        "tokenizer": tokenizer,
        "tts_model": tts_model,
        "embedding_model": embedding_model,
        "boost_value": boost_value,
        "device": device,
        "crisperwhisper_pipe": crisperwhisper_pipe,
        "crisperwhisper_processor": crisperwhisper_processor,
        "rf_model": clf,
        "beginner_collection": client.get_collection("beginner_embeddings"),
        "intermediate_collection": client.get_collection("intermediate_embeddings"),
        "advanced_collection": client.get_collection("advanced_embeddings"),
        "essential_words": essential_words,
    }


if __name__ == "__main__":
    initialize()
