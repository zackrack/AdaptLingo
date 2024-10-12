from style_bert_vits2.nlp import bert_models
from style_bert_vits2.constants import Languages
from style_bert_vits2.tts_model import TTSModel
from pathlib import Path
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

def load_bert_model(bert_models_config):
    """
    Loads the BERT models and tokenizers for the specified languages.
    
    bert_models_config: dict, mapping language codes to model names
    """
    for lang_code, model_name in bert_models_config.items():
        try:
            bert_models.load_model(Languages[lang_code], model_name)
            bert_models.load_tokenizer(Languages[lang_code], model_name)
            print(f"Loaded BERT model and tokenizer for {lang_code}: {model_name}")
        except Exception as e:
            print(f"Error loading BERT model for {lang_code}: {model_name}. Error: {e}")
            raise e
    print("All BERT models and tokenizers loaded successfully.")

def load_llm_model(base_model_name):
    """
    Loads the main chatbot model (LLM) with optional quantization for memory optimization.
    """
    # Configure 4-bit quantization for the model to save memory
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load the base model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quantization_config,
        device_map="auto"
    )

    # Load the tokenizer for the model
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    return model, tokenizer

# Load the TTS model for speech synthesis (now moved to models.py)
def load_tts_model(tts_model_config):
    """
    Loads the Style-BERT-VITS2 TTS model from Hugging Face.
    Downloads the model files if not already present.
    """
    assets_root = Path("model_assets")

    # Ensure the assets directory exists
    if not assets_root.exists():
        assets_root.mkdir(parents=True, exist_ok=True)

    # Extract model file paths from config
    model_file = tts_model_config.get("model_file")
    config_file = tts_model_config.get("config_file")
    style_file = tts_model_config.get("style_file")
    hf_repo = tts_model_config.get("hf_repo")
    device = tts_model_config.get("device", "cuda")
    # List of files to download
    files_to_download = [model_file, config_file, style_file]

    # Download each file from Hugging Face if not present
    for file in files_to_download:
        file_path = assets_root / file
        if not file_path.exists():
            try:
                hf_hub_download(repo_id=hf_repo, filename=file, local_dir=assets_root)
                print(f"Downloaded {file} to {assets_root}")
            except Exception as e:
                print(f"Error downloading {file}: {e}")
                raise e
        else:
            print(f"Found existing {file} in {assets_root}")

    # Initialize the TTS model
    try:
        model = TTSModel(
            model_path=assets_root / model_file,
            config_path=assets_root / config_file,
            style_vec_path=assets_root / style_file,
            device=device  # Use the device specified in config
        )
        print(f"Loaded TTS model on {device}")
    except Exception as e:
        print(f"Error loading TTS model: {e}")
        raise e

    return model