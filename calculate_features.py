import numpy as np
import parselmouth
import librosa


def quick_syllable_estimate(
    sound_path: str,
    silence_threshold_db: float = -25.0,
    pitch_floor: float = 75.0,
    voicing_threshold: float = 0.45,
    sample_step: float = 0.02,    # <-- bigger step = fewer samples = faster
    intensity_quantile: float = 0.99,
    min_syllable_gap_s: float = 0.15,
    downsample_hz: int = 16000,
    skip_pitch_check: bool = False
) -> int:
    """
    Return a quick 'syllable-like' peak count for a single audio file, with speed-ups:
      - Downsampling (16 kHz default)
      - Larger sample_step (0.02 default)
      - Optional pitch skipping (skip_pitch_check=True for max speed)
    """
    # 1) Load audio into parselmouth, measure total duration
    snd_original = parselmouth.Sound(sound_path)
    duration = snd_original.xmax - snd_original.xmin
    if duration <= 0:
        return 0

    # 2) Downsample audio (reduces data size => faster pitch/intensity)
    if downsample_hz > 0:
        snd = snd_original.resample(downsample_hz)
    else:
        snd = snd_original

    # 3) Compute intensity
    intensity = snd.to_intensity()

    # 4) Sample intensity at intervals = sample_step
    times = np.arange(snd.xmin, snd.xmax, sample_step)
    sampled_values = np.array([
        intensity.get_value(t) if intensity.get_value(t) is not None else np.nan
        for t in times
    ], dtype=float)

    # 5) Compute a dB threshold = 0.99 quantile + silence_threshold_db
    valid_values = sampled_values[~np.isnan(sampled_values)]
    if len(valid_values) == 0:
        return 0
    db_q99 = np.quantile(valid_values, intensity_quantile)
    threshold_db = db_q99 + silence_threshold_db

    # 6) Optionally compute pitch
    pitch = None
    if not skip_pitch_check:
        pitch = snd.to_pitch_ac(
            pitch_floor=pitch_floor,
            voicing_threshold=voicing_threshold
        )

    # 7) Identify local maxima
    local_max_indices = []
    for i in range(1, len(times) - 1):
        left_val = sampled_values[i - 1]
        mid_val  = sampled_values[i]
        right_val= sampled_values[i + 1]
        if not np.isnan(mid_val) and not np.isnan(left_val) and not np.isnan(right_val):
            if mid_val > left_val and mid_val > right_val:
                local_max_indices.append(i)

    # 8) Check threshold, (optional) pitch, and min gap
    count_syllables = 0
    last_accepted_time = -999
    for idx in local_max_indices:
        peak_time = times[idx]
        peak_db   = sampled_values[idx]
        if peak_db > threshold_db:
            # If skipping pitch => is_voiced=True by default
            is_voiced = True
            if pitch is not None:
                f0 = pitch.get_value_at_time(peak_time)
                is_voiced = (f0 is not None and not np.isnan(f0))

            if is_voiced and (peak_time - last_accepted_time) > min_syllable_gap_s:
                count_syllables += 1
                last_accepted_time = peak_time

    return count_syllables


def compute_tempo_metrics(
    audio_path: str,
    top_db: float = 20.0,
    sr_downsample: int = 16000,
    skip_pitch_check: bool = False
) -> dict:
    """
    Combine:
      - Syllable counting via quick_syllable_estimate()
      - Silence detection via librosa.effects.split()
    
    Returns a dict of:
       n_syllables, speech_rate, articulation_rate, asd
    """
    # 1) Count syllables (parselmouth) with speed-ups
    n_syllables = quick_syllable_estimate(
        sound_path=audio_path,
        sample_step=0.02,
        downsample_hz=sr_downsample,
        skip_pitch_check=skip_pitch_check
    )

    # 2) Librosa load & downsample for silence detection => speaking time
    y, sr = librosa.load(audio_path, sr=sr_downsample)
    total_duration = len(y) / sr
    if total_duration <= 0:
        return {
            "n_syllables": n_syllables,
            "speech_rate": 0.0,
            "articulation_rate": 0.0,
            "asd": 0.0
        }

    # 3) Identify non-silent intervals
    intervals = librosa.effects.split(y, top_db=top_db)
    speaking_time = sum((end - start) for start, end in intervals)
    speaking_time_s = speaking_time / sr
    pause_time_s = total_duration - speaking_time_s
    if pause_time_s < 0:
        pause_time_s = 0.0

    # 4) Compute SR, AR, ASD
    speech_rate = 0.0 if total_duration <= 0 else (n_syllables / total_duration)
    articulation_rate = (n_syllables / speaking_time_s) if speaking_time_s > 0 else 0.0
    asd = (speaking_time_s / n_syllables) if (n_syllables > 0 and speaking_time_s > 0) else 0.0

    return {
        "n_syllables": n_syllables,
        "speech_rate": speech_rate,
        "articulation_rate": articulation_rate,
        "asd": asd
    }


if __name__ == "__main__":
    # Example usage: single file
    audio_file = "ES_SP_C1_22_16_3_EGG.mp3"

    # If you want maximum speed, set skip_pitch_check=True
    # but watch out for possible extra overcounting
    metrics = compute_tempo_metrics(
        audio_file,
        top_db=20.0,
        sr_downsample=16000,
        skip_pitch_check=False
    )
    print("=== Tempo Metrics ===")
    print(f"  Syllables         = {metrics['n_syllables']}")
    print(f"  Speech Rate       = {metrics['speech_rate']:.2f} syll/sec")
    print(f"  Articulation Rate = {metrics['articulation_rate']:.2f} syll/sec")
    print(f"  ASD               = {metrics['asd']:.2f} sec/syll")
