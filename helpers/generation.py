import soundfile as sf
import torch
import os
import time

def generate_speech_from_text(model, text, filename=None, language="EN"):
    """
    Generates speech from the provided text using the Style-BERT-VITS2 model.
    Saves the audio to a WAV file and returns the file path.
    """
    if not os.path.exists('static/audio'):
        os.makedirs('static/audio')

    try:
        sr, audio = model.infer(text=text, language=language)
        
        # Use a unique filename if not provided
        if not filename:
            filename = f"assistant_response_{int(time.time())}.wav"
        
        # Save audio to "static/audio/filename"
        file_path = os.path.join('static', 'audio', filename)
        sf.write(file_path, audio, sr)
        print(f"Speech generated and saved to {file_path}")
        return file_path  # Return the file path instead of sr and audio
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
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria
        )

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    assistant_response = output_text[len(prompt):].split("\nUser:")[0].strip()
    
    return assistant_response
