import librosa
import numpy as np
import warnings

def analyze_prosody(video_file_path: str, sr: int = 16000):
    """
    Analyzes the prosody (pitch and intensity) of an audio/video file.
    
    Requires ffmpeg to be installed, or the 'moviepy' package (added to requirements.txt).
    `librosa.load` will use one of these to extract the audio track.
    """
    print(f"Loading audio from {video_file_path} for prosody analysis...")
    
    # Suppress warnings if the file is an mp3 (e.g., "audioread warning")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            # Librosa uses moviepy (if installed) or ffmpeg (if in path)
            # to extract audio from video containers.
            # We resample to 16kHz, a common rate for speech processing.
            y, sr = librosa.load(video_file_path, sr=sr)
        except Exception as e:
            print(f"Error loading audio: {e}")
            print("This often happens if ffmpeg is not installed, or 'moviepy' is missing.")
            raise e

    print("Audio loaded, analyzing pitch (YIN)...")
    # Get pitch (f0) using YIN algorithm
    # fmin/fmax are typical human speech ranges
    f0 = librosa.yin(y,
                     fmin=librosa.note_to_hz('C2'), # ~65 Hz
                     fmax=librosa.note_to_hz('C7')) # ~2093 Hz

    print("Analyzing intensity (RMS)...")
    # Get intensity (Root-Mean-Square energy)
    # The result is a 2D array (n_rms, n_frames), so we take [0]
    rms = librosa.feature.rms(y=y)[0]

    print("Aligning timestamps...")
    # Get the timestamps for each frame.
    # f0 and rms will have the same number of frames by default.
    times = librosa.times_like(f0, sr=sr)

    prosody_log = []
    
    # Iterate and create the log
    for i in range(len(times)):
        timestamp = times[i]
        pitch = f0[i]
        intensity = rms[i]
        
        prosody_log.append({
            "timestamp": float(timestamp),
            # Convert numpy.nan to None for JSON serialization
            "pitch": float(pitch) if not np.isnan(pitch) else None,
            "intensity": float(intensity)
        })

    print(f"Prosody analysis complete. {len(prosody_log)} frames processed.")
    return prosody_log

# --- Local Test ---
if __name__ == "__main__":
    """
    To test this script directly:
    1. Make sure you have 'moviepy' installed: pip install moviepy
    2. Run: python prosody_processor.py <path_to_your_video.mp4>
    """
    import json
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python prosody_processor.py <path_to_video.mp4>")
        sys.exit(1)
        
    file_path = sys.argv[1]
    
    try:
        log = analyze_prosody(file_path)
        
        output_file = "prosody_output.json"
        with open(output_file, "w") as f:
            json.dump(log, f, indent=2)
        
        print(f"\n✅ Prosody analysis successful.")
        print(f"📄 Results saved to {output_file}")
        
        if log:
            print("\n-- Sample (first 5 frames) --")
            print(json.dumps(log[:5], indent=2))
            
    except Exception as e:
        print(f"\n❌ Prosody analysis failed: {e}")
        print("Please ensure 'moviepy' is installed (`pip install moviepy`) or 'ffmpeg' is in your system PATH.")