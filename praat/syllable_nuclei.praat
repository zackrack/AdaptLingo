###########################################################################
#                                                                         #
#  Praat Script Syllable Nuclei, version 3 (Syllable Detector)           #
#  Modified for Direct Single File Input Handling                         #
#                                                                         #
###########################################################################
#
# Original Copyright and License...
# ... [omitted for brevity]
#
# Modified to handle single file input and correct DataCollectionType handling

form Detect Syllables and Filled Pauses in Speech Utterances
  comment As INPUT, provide the path to the Sound file:
  sentence SoundFilePath C:/Users/zrack/chatbotproject/audio/C1/alice.wav
  
  comment ________________________________________________________________________________
  
  comment  Parameters Syllable Nuclei:
  optionmenu Pre_processing 1
    option None
    option Band pass (300..3300 Hz)
    option Reduce noise
  real Silence_threshold_(dB) -25
  real Minimum_dip_near_peak_(dB) 2
  real Minimum_pause_duration_(s) 0.3
# real Pitch_floor_(Hz) 30
# real Voicing_threshold 0.25
# optionmenu Parser 2
#   option peak-dip		; save code in case anybody requests backwards compatibility
#   option dip-peak-dip
  comment ________________________________________________________________________________
  
  comment  Parameters Filled Pauses:
  boolean Detect_Filled_Pauses yes
  optionmenu Language 1
    option English
#   option Mandarin (not yet implemented)
#   option Spanish  (not yet implemented)
    option Dutch
  real Filled_Pause_threshold 1.00
  comment ________________________________________________________________________________
  
  comment Destination of OUTPUT:
  optionmenu Data 1
    option TextGrid(s) only
    option Praat Info window
    option Save as text file
    option Table
  choice DataCollectionType 2
    button OverWriteData
    button AppendData
  boolean Keep_Objects_(when_processing_files) yes
endform

# Hidden arguments for internal use
pitch_floor       = 30
voicing_threshold =  0.25
parser$           = "dip-peak-dip"

# Procedure to validate and process arguments
procedure processArgs_single
  sound_file$ = SoundFilePath$
  if not fileReadable(sound_file$)
    exit: "Cannot open file ", sound_file$
  endif
  
  # Map DataCollectionType index to string
  if DataCollectionType$ == "1"
    DataCollectionType = "OverWriteData"
  elif DataCollectionType$ == "2"
    DataCollectionType = "AppendData"
  else
    exit: "Unknown DataCollectionType: ", DataCollectionType$
  endif
endproc

processArgs_single

# Read the sound file directly
Read from file: sound_file$

# Get Sound object after reading the file
idSnd = selected("Sound")

# Procedure to find syllable nuclei
procedure findSyllableNuclei
    name$ = selected$("Sound")

    # Apply pre-processing if selected
    if Pre_processing$ == "None"
        idSnd_processed = selected ("Sound")
    elif Pre_processing$ == "Band pass (300..3300 Hz)"
        idSnd_processed = Filter (pass Hann band): 300, 3300, 100
        Scale peak: 0.99
    elif Pre_processing$ == "Reduce noise"
        idSnd_processed = noprogress Reduce noise: 0, 0, 0.025, 80, 10000, 40, -20, "spectral-subtraction"
        Scale peak: 0.99
    endif

    # Extract timing information
    tsSnd  = Get start time
    teSnd  = Get end time
    dur    = Get total duration

    # Convert to Intensity
    idInt  = To Intensity: 50, 0, "yes"		
    dbMin  = Get minimum: 0, 0, "Parabolic"
    dbMax  = Get maximum: 0, 0, "Parabolic"
    dbQ99  = Get quantile: 0, 0, 0.99		

    # Estimate Intensity threshold
    threshold  = dbQ99 + Silence_threshold_(dB)
    threshold2 = dbMax - dbQ99
    threshold3 = Silence_threshold_(dB) - threshold2
    if threshold < dbMin
        threshold = dbMin
    endif

    # Detect silences and create TextGrid
    idTG = To TextGrid (silences): threshold3, Minimum_pause_duration_(s), 0.1, "", "sound"
    Set tier name: 1, "Phrases"
    nrIntervals = Get number of intervals: 1
    nsounding   = 0
    speakingtot = 0

    # Iterate over intervals to count sounding segments
    for interval to nrIntervals
        lbl$ = Get label of interval: 1, interval
        if lbl$ <> ""
            ts = Get start time of interval: 1, interval
            te = Get end time of interval: 1, interval
            nsounding   += 1
            speakingtot += te - ts
            Set interval text: 1, interval, string$(nsounding)
        endif
    endfor

    # Convert Intensity to Peaks
    selectObject: idInt
    idPeak = To IntensityTier (peaks)
    
    # Convert Sound to Pitch
    selectObject: idSnd_processed
    idP = noprogress To Pitch (ac): 0.02, pitch_floor, 4, "no", 0.03, voicing_threshold, 0.01, 0.35, 0.25, 450

    # Fill arrays with intensity values
    peakcount = 0
    selectObject: idPeak
    nrPeaks = Get number of points
    for peak to nrPeaks
        selectObject: idPeak
        time  = Get time from index: peak
        dbMax = Get value at index: peak
        selectObject: idP
        voiced = Get value at time: time, "Hertz", "Linear"
        if dbMax > threshold and (voiced <> undefined)
            peakcount += 1
            t [peakcount*2] = time		; peaks at EVEN indices (base 2)
            db[peakcount*2] = dbMax
        endif
    endfor

    # Get Minima between peaks
    t[0]               = Get start time
    t[2*(peakcount+1)] = Get end time
    selectObject: idInt
    for valley to peakcount+1
        t [2*valley-1] = Get time of minimum: t[2*(valley-1)], t[2*valley], "Parabolic"
        db[2*valley-1] = Get minimum:         t[2*(valley-1)], t[2*valley], "Parabolic"
    endfor

    # Insert Nuclei into TextGrid
    selectObject: idTG
    Insert point tier: 1, "Nuclei"

    # Initialize variables for voiced counts
    voicedcount = 0	
    tp[voicedcount] = t[0]
    tRise = t[0]
    tFall = t[0]
    tMax = t[0]
    tMin = t[0]
    dbMax = db[1]
    dbMin = db[1]
    nrPoints = 2*peakcount+1
    selectObject: idTG

    # Iterate through points to detect nuclei
    for point to nrPoints
        if db[point] > dbMax
            tMax  =  t[point]
            dbMax = db[point]
            if db[point] - dbMin > Minimum_dip_near_peak_(dB)
                tRise =  t[point]
                dbMin = db[point]
            endif
        elif db[point] < dbMin
            tMin  =  t[point]
            dbMin = db[point]
            if dbMax - db[point] > Minimum_dip_near_peak_(dB)
                tFall =  t[point]
                dbMax = db[point]
            endif
        endif

        # Insert voiced peaks based on parser type
        if parser$ == "dip-peak-dip" and tRise <> t[0] and tRise < tFall and tFall <> t[0]
            i  = Get interval at time: 1, tMax  # Changed tier number to 1 (Phrases)
            l$ = Get label of interval: 1, i
            if l$ <> ""
                voicedcount += 1
                tp[voicedcount] = tMax
                Insert point: 1, tMax, string$(voicedcount)
                tMax  = t[point]
                tMin  = t[point]
                dbMin = db[point]
                dbMax = db[point]
                tRise = t[0]
                tFall = t[0]
            endif
        endif
    endfor

    # Summarize results in Info window or other output
    speakingrate = voicedcount / dur
    articulationrate = voicedcount / speakingtot
    npause = nsounding - 1
    asd = speakingtot / voicedcount

    if Data$ == "Praat Info window"
        appendInfo: "'name$', 'voicedcount', 'npause', 'dur(s)', 'speakingtot(s)', 'speakingrate(nsyll/dur)', 'articulationrate(nsyll/phonationtime)', 'asd(speakingtime/nsyll)'"
    elif Data$ == "Save as text file"
        appendFile: "SyllableNuclei.txt", "'name$', 'voicedcount', 'npause', 'dur(s)', 'speakingtot(s)', 'speakingrate(nsyll/dur)', 'articulationrate(nsyll/phonationtime)', 'asd(speakingtime/nsyll)'"
    elif Data$ == "Table"
        appendFile: temporaryDirectory$ + "/SyllableNuclei.tmp", "'name$', 'voicedcount', 'npause', 'dur(s)', 'speakingtot(s)', 'speakingrate(nsyll/dur)', 'articulationrate(nsyll/phonationtime)', 'asd(speakingtime/nsyll)'"
    endif
endproc

# Procedure to count filled pauses
procedure countFilledPauses: .id
    selectObject: .id
    .nrInt = Get number of intervals: 3  # Assuming tier 3 exists for filled pauses
    .nrFP  = 0
    .tFP   = 0
    for .int to .nrInt
        .lbl$ = Get label of interval: 3, .int
        if .lbl$ == "fp"
            .nrFP += 1
            .ts = Get start time of interval: 3, .int
            .te = Get end time of interval: 3, .int
            .tFP += (.te - .ts)
        endif
    endfor
    if Data$ == "Praat Info window"
        appendInfoLine: ", ', .nrFP, ', .tFP:3'"
    elif Data$ == "Save as text file"
        appendFileLine: "SyllableNuclei.txt", ", ', .nrFP, ', .tFP:3'"
    elif Data$ == "Table"
        appendFileLine: temporaryDirectory$ + "/SyllableNuclei.tmp", ", ', .nrFP, ', .tFP:3'"
    endif
endproc

# Procedure to terminate lines in output
procedure terminateLines
    if Data$ == "Praat Info window"
        appendInfoLine: ""
    elif Data$ == "Save as text file"
        appendFileLine: "SyllableNuclei.txt", ""
    elif Data$ == "Table"
        appendFileLine: temporaryDirectory$ + "/SyllableNuclei.tmp", ""
    endif
endproc

# Procedure for final output handling
procedure coda
    if Data$ == "Table"
        Read Table from comma-separated file: temporaryDirectory$ + "/SyllableNuclei.tmp"
        deleteFile: temporaryDirectory$ + "/SyllableNuclei.tmp"
    endif
    # Note: For single file, object management is simplified
endproc

# Start processing
findSyllableNuclei

if Detect_Filled_Pauses
    # Run the FilledPauses.praat script, ensure it's in the same directory or provide full path
    runScript: "FilledPauses.praat", Language$, Filled_Pause_threshold, (Data$ == "Table")
    
    if Data$ == "Table"
        idTbl = selected("Table")
    endif
    countFilledPauses idTbl
else
    terminateLines
endif

# Save output based on user selection
selectObject: idTG
if Data$ == "TextGrid(s) only"
    # Replace .wav extension with .TextGrid
    textgrid_file$ = replace$(sound_file$, ".wav", ".TextGrid")
    Save as TextGrid: textgrid_file$
elif Data$ == "Praat Info window"
    # Output is already in Info window
elif Data$ == "Save as text file"
    # Save as TextGrid file and append to text file
    textgrid_file$ = replace$(sound_file$, ".wav", ".TextGrid")
    Save as TextGrid: textgrid_file$
    # Save additional info to text file
    Save as text file: "SyllableNuclei.txt", textgrid_file$
elif Data$ == "Table"
    # Replace .wav extension with .auto.Table and save to temporary file
    table_file$ = temporaryDirectory$ + "/SyllableNuclei.tmp"
    Save as tab-separated file: table_file$, replace$(sound_file$, ".wav", ".auto.Table")
endif

# Clean up objects if necessary
if not Keep_Objects_(when_processing_files)
    removeObject: idSnd, idTG
    if Detect_Filled_Pauses and Data$ == "Table"
        removeObject: idTbl
    endif
endif

# Final output handling
coda
