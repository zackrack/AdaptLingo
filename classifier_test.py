from huggingface_hub import hf_hub_download
import joblib

# Download the private pickle file using your access token
model_path = hf_hub_download(
    repo_id="zrackau1/cefr_fluency_classifier", 
    filename="fluency_classifier.pkl",
    token="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"  # Replace with your actual token
)

# Load the model using joblib
clf = joblib.load(model_path)

import pandas as pd

# Create a dummy input with your three features
dummy_data = pd.DataFrame({
    "speechrate(nsyll/dur)": [5.5],                 # dummy value for speech rate
    "articulation_rate(nsyll/phonationtime)": [6.7],  # dummy value for articulation rate
    "ASD(speakingtime/nsyll)": [0.9]                  # dummy value for ASD ratio
})

# Get prediction from the loaded classifier
prediction = clf.predict(dummy_data)

# Map numeric predictions back to labels (based on your original mapping)
inverse_label_mapping = {0: "beginner", 1: "intermediate", 2: "advanced"}
predicted_label = inverse_label_mapping[prediction[0]]

print("Predicted Fluency:", predicted_label)
