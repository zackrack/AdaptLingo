def quick_syllable_estimate(
    sound_path: str,
    silence_threshold_db: float = -25.0,
    pitch_floor: float = 75.0,
    voicing_threshold: float = 0.45,
    sample_step: float = 0.01,
    intensity_quantile: float = 0.99,
    min_syllable_gap_s: float = 0.15,
    downsample_hz: int = 16000,
    skip_pitch_check: bool = False,
    smoothing_window_s: float = 0.05,  # ~50ms smoothing
) -> int:
    """
    Return a quick 'syllable-like' peak count for a single audio file.
    Uses Savitzky–Golay smoothing to reduce overcounting from minor fluctuations.
    """
    # 1) Load audio, measure total duration
    snd_original = parselmouth.Sound(sound_path)
    duration = snd_original.xmax - snd_original.xmin
    if duration <= 0:
        return 0

    # 2) Downsample to reduce data size => faster pitch/intensity
    if downsample_hz > 0:
        snd = snd_original.resample(downsample_hz)
    else:
        snd = snd_original

    # 3) Compute intensity
    intensity = snd.to_intensity()

    # 4) Sample intensity at regular intervals
    times = np.arange(snd.xmin, snd.xmax, sample_step)
    sampled_values = np.array(
        [intensity.get_value(t) if intensity.get_value(t) is not None else np.nan
         for t in times],
        dtype=float
    )

    valid_values = sampled_values[~np.isnan(sampled_values)]
    if len(valid_values) == 0:
        return 0

    # 5) Compute dB threshold based on quantile + offset
    db_q99 = np.quantile(valid_values, intensity_quantile)
    threshold_db = db_q99 + silence_threshold_db

    # 6) Optionally compute pitch (used to verify voicing)
    pitch = None
    if not skip_pitch_check:
        pitch = snd.to_pitch_ac(
            pitch_floor=pitch_floor,
            voicing_threshold=voicing_threshold
        )

    # 7) Savitzky–Golay SMOOTHING to reduce minor fluctuations
    window_size = int(np.round(smoothing_window_s / sample_step))
    # Savitzky–Golay filter requires an odd window length
    if window_size % 2 == 0:
        window_size += 1

    # If the window_size is too large or too small, clamp it:
    if window_size < 3:
        window_size = 3
    if window_size >= len(sampled_values):
        window_size = len(sampled_values) - 1
        if window_size % 2 == 0:
            window_size -= 1

    # polyorder must be < window_size; let's keep it conservative
    polyorder = min(3, window_size - 1)

    # Some edge-case handling: if data is too short for smoothing, skip it
    if len(sampled_values) > window_size and window_size >= 3:
        # Replace NaNs temporarily (S-G filter cannot handle NaNs)
        # We'll do a basic fill, or you can use your own interpolation logic
        nan_mask = np.isnan(sampled_values)
        if np.any(nan_mask):
            # Simple fill: replace NaNs with the mean of non-NaN samples
            fill_val = np.nanmean(sampled_values)
            sampled_values[nan_mask] = fill_val

        smoothed_values = savgol_filter(sampled_values, window_size, polyorder)
    else:
        smoothed_values = sampled_values

    # 8) Identify local maxima on the smoothed contour
    local_max_indices = []
    for i in range(1, len(times) - 1):
        left_val  = smoothed_values[i - 1]
        mid_val   = smoothed_values[i]
        right_val = smoothed_values[i + 1]
        if not np.isnan(mid_val) and not np.isnan(left_val) and not np.isnan(right_val):
            if mid_val > left_val and mid_val > right_val:
                local_max_indices.append(i)

    # 9) Check thresholds, pitch, and minimum gap for each peak
    count_syllables = 0
    last_accepted_time = -999
    for idx in local_max_indices:
        peak_time = times[idx]
        peak_db   = smoothed_values[idx]
        if peak_db > threshold_db:
            # If skipping pitch => is_voiced=True by default
            is_voiced = True
            if pitch is not None:
                f0 = pitch.get_value_at_time(peak_time)
                is_voiced = (f0 is not None and not np.isnan(f0))

            # Only count if voiced, above threshold, and not too close to last peak
            if is_voiced and (peak_time - last_accepted_time) > min_syllable_gap_s:
                count_syllables += 1
                last_accepted_time = peak_time

    return count_syllables