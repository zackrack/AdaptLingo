import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

def initialize_crisperwhisper(
    model_id: str = "nyrahealth/CrisperWhisper",
    chunk_length_s: float = 15,
    batch_size: int = 4,
    return_timestamps: str = "word"
):
    """
    Initialize and return the CrisperWhisper ASR pipeline.

    Parameters:
        model_id (str): The Hugging Face model ID to load.
        chunk_length_s (float): The length (in seconds) for each audio chunk.
        batch_size (int): The batch size for audio processing.
        return_timestamps (str): Granularity of timestamps to return (e.g., 'word').

    Returns:
        A Hugging Face ASR pipeline instance configured for CrisperWhisper.
    """
    # Determine device and torch dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"Initializing CrisperWhisper on device: {device} with model ID: {model_id}")

    # Load the CrisperWhisper model with error handling
    try:
        crisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load CrisperWhisper model '{model_id}': {e}")

    crisper_model.to(device)

    # Load the corresponding processor
    try:
        crisper_processor = AutoProcessor.from_pretrained(model_id)
    except Exception as e:
        raise RuntimeError(f"Failed to load CrisperWhisper processor for '{model_id}': {e}")

    # Create the ASR pipeline with word-level timestamps
    try:
        crisperwhisper_pipe = pipeline(
            "automatic-speech-recognition",
            model=crisper_model,
            tokenizer=crisper_processor.tokenizer,
            feature_extractor=crisper_processor.feature_extractor,
            chunk_length_s=chunk_length_s,
            batch_size=batch_size,
            return_timestamps=return_timestamps,
            torch_dtype=torch_dtype,
            device=device,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to create the ASR pipeline: {e}")

    print("CrisperWhisper pipeline successfully initialized.")
    return crisperwhisper_pipe
