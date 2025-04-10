import soundfile as sf
import torch
import os
import time
import nltk
import re

def ensure_nltk_tagger_resource():
    """
    Ensures that the required NLTK resource is available.
    """
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        print("NLTK resource not found. Downloading 'averaged_perceptron_tagger'...")
        nltk.download('averaged_perceptron_tagger')
        print("Download complete.")
    else:
        print("NLTK resource already available.")
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    except LookupError:
        print("NLTK resource not found. Downloading 'averaged_perceptron_tagger_eng'...")
        nltk.download('averaged_perceptron_tagger_eng')
        print("Download complete.")
    else:
        print("NLTK resource already available.")

def generate_speech_from_text(model, text, filename=None, language="EN"):
    """
    Generates speech from the provided text using the Style-BERT-VITS2 model.
    Saves the audio to a WAV file and returns the file path.
    """
    if not os.path.exists('static/audio'):
        os.makedirs('static/audio')

    # Ensure NLTK resource is available before running inference
    ensure_nltk_tagger_resource()
 
    try:
        # Run the TTS inference process
        sr, audio = model.infer(text=text, language=language)
        
        # Use a unique filename if not provided
        if not filename:
            filename = f"assistant_response_{int(time.time())}.wav"
        
        # Save audio to "static/audio/filename"
        file_path = os.path.join('static', 'audio', filename)
        sf.write(file_path, audio, sr)
        print(f"Speech generated and saved to {file_path}")
        return file_path  # Return the file path instead of sr and audio
    except LookupError as e:
        print(f"Error generating speech due to missing resource: {e}")
        # Retry downloading and infer again if a LookupError occurs
        ensure_nltk_tagger_resource()
        return generate_speech_from_text(model, text, filename, language)  # Retry inference
    except Exception as e:
        print(f"Error generating speech: {e}")
        return None


# Function to generate the chatbot's response (moved to generation.py)
def generate_response(model, tokenizer, prompt, logits_processor, stopping_criteria, device):
    """
    Generates a response from the model using the provided prompt, logits processor,
    and stopping criteria.
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=inputs["attention_mask"],  # Add the attention mask here
            max_new_tokens=50,
            do_sample=True,
            temperature=0.2,
            pad_token_id=tokenizer.eos_token_id,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria
        )

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    assistant_response = output_text[len(prompt):].split("\nUser:")[0].strip()
    
    match = re.search(r'(User:|Assistant:)', assistant_response)
    if match:
        assistant_response = assistant_response[:match.start()]
    return assistant_response.strip()