import os
import pandas as pd

# List of proficiency levels (folder names)
# levels = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2', 'Unknown']
levels = ['Unknown']

# Define the path to your audio directory
audio_directory = 'audio'  # Replace with your audio directory path

# Initialize an empty list to collect file names and their corresponding proficiency levels
data = []

# Traverse through the folders in the audio directory
for level in levels:
    folder_path = os.path.join(audio_directory, level)
    
    # Check if the folder exists
    if os.path.isdir(folder_path):
        # Process each audio file in the folder
        for file in os.listdir(folder_path):
            if file.endswith('.wav'):  # Only process .wav files
                # Normalize file names for matching
                file_name = file.replace('.wav', '').strip()
                file_name = file_name.replace(' ', '_')
                
                # Add file name and proficiency level to the list
                data.append({'name': file_name, 'fluency': level})

# Create a DataFrame with the processed data (audio file names and proficiency levels)
fluency_data = pd.DataFrame(data)

# Load Fluency data (if you have an existing file with Fluency column, e.g., 'audio_file_fluency.csv')
# If you have no external fluency data, you can skip this part
fluency_file = 'audio/audio.csv'  # Replace with your actual file path (if needed)
fluency_data_external = pd.read_csv(fluency_file)

# Normalize audio names in the external fluency data
fluency_data_external['Audio'] = fluency_data_external['Audio'].str.replace('.wav', '', regex=False).str.strip()
fluency_data_external['Audio'] = fluency_data_external['Audio'].str.replace(' ', '_', regex=False)
fluency_data_external.rename(columns={'Audio': 'name'}, inplace=True)

# Initialize an empty list to collect merged DataFrames
merged_dataframes = []

# Process each level's SyllableNuclei file
for level in levels:
    file_path = f'SyllableNuclei_{level}.txt'  # Replace with your actual file paths
    syllable_data = pd.read_csv(file_path, delimiter=',')  # Adjust delimiter if necessary
    
    # Normalize audio names in syllable data
    syllable_data['name'] = syllable_data['name'].str.strip()
    syllable_data['name'] = syllable_data['name'].str.replace(' ', '_', regex=False)
    
    # Merge with fluency data (either external fluency or folder-based)
    merged = pd.merge(syllable_data, fluency_data[['name', 'fluency']], on='name', how='left')
    
    # Collect the merged DataFrame
    merged_dataframes.append(merged)

# Concatenate all levels into a single DataFrame
final_merged_data = pd.concat(merged_dataframes, ignore_index=True)

# Debugging: Inspect unmatched rows
unmatched = final_merged_data[final_merged_data['fluency'].isna()]
print(f"Number of unmatched rows: {len(unmatched)}")
print("Sample of unmatched rows:")
print(unmatched.head())

# Drop rows where the Fluency column is blank (NaN)
final_merged_data = final_merged_data.dropna(subset=['fluency'])

# Save the final merged data to a CSV file
output_file = 'audio/totrain.csv'
final_merged_data.to_csv(output_file, index=False)

print(f"Merged data has been saved to {output_file}")
