# Data Specification for Audio Processing Script

## Overview
This document outlines the necessary data structure and requirements for the script to successfully process and categorize audio files based on their fluency ratings.

## Folder Structure

The script expects the following directory structure:

```
project_root/
│── csv/
│   ├── combined_ratings_for_both_sets.csv  # Input CSV containing fluency ratings
│── sampled_audio/
│   ├── <audio_file_1>.wav
│   ├── <audio_file_2>.wav
│   ├── ...
│── audio/  # Output folder (created by the script)
│   ├── A1/
│   ├── A2/
│   ├── B1/
│   ├── B2/
│   ├── C1/
│   ├── C2/
│   ├── Unknown/
│   ├── processed_audio_fluency.csv  # Output CSV containing processed fluency ratings
```

## Input CSV (`csv/combined_ratings_for_both_sets.csv`)

The input CSV file should contain the following columns:

| Column Name | Description |
|-------------|------------|
| `Audio` | The filename (without `.wav` extension) of the audio sample. |
| `Fluency` | The first fluency rating (CEFR level: A1, A2, B1, B2, C1, C2). |
| `Fluency2` | The second fluency rating (CEFR level: A1, A2, B1, B2, C1, C2). |

### Example of `combined_ratings_for_both_sets.csv`
```
Audio,Fluency,Fluency2
sample_001,A1,A2
sample_002,B1,B2
sample_003,C1,C2
sample_004,A2,B1
sample_005,B2,C1
```

## Processing Steps

1. The script reads `csv/combined_ratings_for_both_sets.csv`.
2. It removes any leading/trailing spaces in the `Fluency` and `Fluency2` columns.
3. The CEFR fluency levels (`A1` to `C2`) are mapped to numeric values (`1` to `6`):
   ```
   A1 -> 1
   A2 -> 2
   B1 -> 3
   B2 -> 4
   C1 -> 5
   C2 -> 6
   ```
4. The highest fluency rating between `Fluency` and `Fluency2` is selected.
5. The numeric fluency rating is rounded and converted back to a CEFR level.
6. A new CSV (`audio/processed_audio_fluency.csv`) is generated with the processed fluency levels.
7. The script ensures that subdirectories (`A1` to `C2` and `Unknown`) exist in the `audio/` folder.
8. It copies `.wav` audio files from `sampled_audio/` into their corresponding fluency-level folders.
9. If an audio file is missing from `sampled_audio/`, it is logged as **not found**.

## Output CSV (`audio/processed_audio_fluency.csv`)

| Column Name | Description |
|-------------|------------|
| `Audio` | The filename (without `.wav` extension). |
| `Fluency` | Original first fluency rating. |
| `Fluency2` | Original second fluency rating. |
| `Fluency_Numeric` | First fluency rating converted to a number (1-6). |
| `Fluency2_Numeric` | Second fluency rating converted to a number (1-6). |
| `Merged_Fluency_Numeric` | The highest fluency rating. |
| `Merged_Fluency` | Final fluency rating after rounding. |

### Example of `processed_audio_fluency.csv`
```
Audio,Fluency,Fluency2,Fluency_Numeric,Fluency2_Numeric,Merged_Fluency_Numeric,Merged_Fluency
sample_001,A1,A2,1,2,2,A2
sample_002,B1,B2,3,4,4,B2
sample_003,C1,C2,5,6,6,C2
sample_004,A2,B1,2,3,3,B1
sample_005,B2,C1,4,5,5,C1
```

## Expected Output in `audio/` Folder

After processing, the `audio/` directory will contain:

- Audio files copied into their respective CEFR level folders.
- A `processed_audio_fluency.csv` file summarizing the processed fluency ratings.

Example after execution:
```
audio/
├── A1/
│   ├── sample_001.wav
├── A2/
│   ├── sample_002.wav
├── B1/
│   ├── sample_004.wav
├── B2/
│   ├── sample_003.wav
├── C1/
│   ├── sample_005.wav
├── C2/
│   ├── sample_006.wav
├── Unknown/  # Contains any files with missing or unrecognized fluency levels
├── processed_audio_fluency.csv
```

## Notes

- The script assumes that **all audio files are named exactly as listed in the CSV file** (without the `.wav` extension in the CSV).
- If an audio file is missing, it will **not be copied**, and a message will be printed to indicate that it was not found.
- The final `Merged_Fluency` rating ensures consistency in fluency categorization.
- The script uses **max()** between `Fluency_Numeric` and `Fluency2_Numeric` to select the higher fluency rating.

## Contact

For any issues or questions about the dataset format, please reach out to the maintainer.

