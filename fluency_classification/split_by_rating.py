import os
import shutil
import pandas as pd

# Define paths
csv_file = 'csv/combined_ratings_for_both_sets.csv'  # Path to your stacked CSV
audio_folder = 'sampled_audio'  # The folder containing your audio files
output_folder = 'audio'  # Folder where the files will be copied to
processed_csv = 'audio/processed_audio_fluency.csv'  # File to save processed fluency ratings

# Create level mapping for CEFR levels
level_mapping = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}
reverse_mapping = {v: k for k, v in level_mapping.items()}

# Read the CSV file
df = pd.read_csv(csv_file)

# Strip leading/trailing spaces from 'Fluency' and 'Fluency2' columns
df["Fluency"] = df["Fluency"].str.strip()
df["Fluency2"] = df["Fluency2"].str.strip()

# Debugging: Check unique values in 'Fluency' and 'Fluency2'
print("Unique values in 'Fluency':", df['Fluency'].unique())
print("Unique values in 'Fluency2':", df['Fluency2'].unique())

# Map CEFR levels to numeric values

df["Fluency_Numeric"] = df["Fluency"].map(level_mapping)
df["Fluency2_Numeric"] = df["Fluency2"].map(level_mapping)

# Take the mean of the two ratings
df["Merged_Fluency_Numeric"] = df[["Fluency_Numeric", "Fluency2_Numeric"]].max(axis=1)

# Round the numeric mean to the nearest integer
df["Merged_Fluency_Numeric"] = df["Merged_Fluency_Numeric"].round()

# Convert back to CEFR levels using the rounded numeric mean
df["Merged_Fluency"] = df["Merged_Fluency_Numeric"].map(reverse_mapping).fillna("Unknown")

# Debugging: Check any rows where the merged fluency is still "Unknown"
print("Rows with 'Unknown' merged fluency after rounding:")
print(df[df['Merged_Fluency'] == 'Unknown'])

# Save the processed DataFrame to a CSV file for future use
df.to_csv(processed_csv, index=False)
print(f"Processed fluency ratings saved to: {processed_csv}")

# Ensure output directories exist
for level in reverse_mapping.values():
    os.makedirs(os.path.join(output_folder, level), exist_ok=True)
os.makedirs(os.path.join(output_folder, "Unknown"), exist_ok=True)

# Copy files into appropriate directories
for _, row in df.iterrows():
    audio_file = os.path.join(audio_folder, row["Audio"] + ".wav")  # Use the 'Audio' value from the CSV
    merged_fluency = row["Merged_Fluency"]

    # Check if the audio file exists
    if os.path.exists(audio_file):
        target_dir = os.path.join(output_folder, merged_fluency)
        shutil.copy(audio_file, target_dir)  # Use copy instead of move
        print(f"Copied: {audio_file} -> {target_dir}")
    else:
        print(f"Audio file not found: {audio_file}")

print("Files copied successfully.")
