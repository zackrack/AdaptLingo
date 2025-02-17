# helpers/__init__.py

from .models import load_bert_model, load_llm_model, load_tts_model
from .generation import generate_speech_from_text, generate_response
from .processors import BoostLogitsProcessor, StopOnTokens, create_boost_processor, create_stopping_criteria
from .retrieval import knn_search, get_or_create_collection
from .text_handling import read_words_file, build_prompt
from .config_handling import load_config, update_config, load_initial_data, reinitialize_models
from .audio import transcribe_audio, classify_fluency

__all__ = [
    "load_bert_model",              # Load the BERT model (now set for English)
    "load_llm_model",               # Load the main LLM model (Mistral-7B or other)
    "load_tts_model",               # Load the TTS model for generating English speech
    "generate_speech_from_text",    # Generate speech from text
    "generate_response",            # Generate the chatbot response
    "BoostLogitsProcessor",         # Custom logits processor for boosting specific token probabilities
    "StopOnTokens",                 # Custom stopping criteria for token-based stopping
    "create_boost_processor",       # Helper for creating the boost processor
    "create_stopping_criteria",     # Helper for creating stopping criteria
    "knn_search",                   # Function for KNN search in the embeddings
    "get_or_create_collection",     # Create or get a ChromaDB collection
    "read_words_file",              # Function to read word lists
    "build_prompt",                  # Build the prompt for chatbot generation
    "load_config",
    "update_config",
    "load_initial_data",
    "reinitialize_models",
    "transcribe_audio",
    "classify_fluency",
]
