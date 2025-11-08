"""
enricher.py

Master module for enriching transcript data with multimodal analysis.
Combines transcription, vision (pose), and prosody data to create a 
comprehensive multimodal timeline with insights and tags.
"""

import numpy as np
from heuristics import apply_all_heuristics, analyze_sentence_patterns


def calculate_wrist_velocity(vision_frames):
    """
    Calculates average wrist velocity over a sequence of frames.
    Uses landmarks 15 (Left Wrist) and 16 (Right Wrist).
    
    Args:
        vision_frames: list of vision frame dicts with pose data
        
    Returns:
        float: Average wrist velocity (Euclidean distance per frame)
    """
    if not vision_frames or len(vision_frames) < 2:
        return 0.0
        
    total_movement = 0.0
    frame_count = 0
    
    for i in range(1, len(vision_frames)):
        # Safely access poses
        prev_poses = vision_frames[i-1].get("poses", [])
        curr_poses = vision_frames[i].get("poses", [])
        
        if not prev_poses or not curr_poses:
            continue
            
        prev = prev_poses[0] if len(prev_poses) > 0 else []
        curr = curr_poses[0] if len(curr_poses) > 0 else []
        
        # Helper to safely get landmark by ID
        def get_lm(pose, lm_id):
            if not pose:
                return None
            return next((lm for lm in pose if lm.get("id") == lm_id), None)

        prev_lw = get_lm(prev, 15)
        prev_rw = get_lm(prev, 16)
        curr_lw = get_lm(curr, 15)
        curr_rw = get_lm(curr, 16)

        if prev_lw and curr_lw and prev_rw and curr_rw:
            # Simple 2D Euclidean distance (ignoring Z for simplicity)
            l_dist = np.sqrt((curr_lw["x"] - prev_lw["x"])**2 + (curr_lw["y"] - prev_lw["y"])**2)
            r_dist = np.sqrt((curr_rw["x"] - prev_rw["x"])**2 + (curr_rw["y"] - prev_rw["y"])**2)
            
            # Average movement of both hands between these two frames
            total_movement += (l_dist + r_dist) / 2.0
            frame_count += 1

    return total_movement / frame_count if frame_count > 0 else 0.0


def calculate_prosody_metrics(prosody_frames):
    """
    Calculate average prosody metrics from a list of prosody frames.
    
    Args:
        prosody_frames: list of prosody dicts with 'pitch' and 'intensity'
        
    Returns:
        tuple: (avg_intensity, avg_pitch)
    """
    if not prosody_frames:
        return 0.0, 0.0
    
    # Filter out invalid/outlier pitch values (e.g., > 2000 Hz or < 50 Hz)
    intensities = [p.get("intensity", 0) for p in prosody_frames 
                   if p.get("intensity") is not None]
    pitches = [p.get("pitch", 0) for p in prosody_frames 
               if p.get("pitch") is not None and 50 < p.get("pitch", 0) < 2000]
    
    avg_intensity = np.mean(intensities) if intensities else 0.0
    avg_pitch = np.mean(pitches) if pitches else 0.0
    
    return avg_intensity, avg_pitch


def slice_data_for_word(word, vision_data, prosody_data):
    """
    Extract vision and prosody frames that fall within a word's time range.
    
    Args:
        word: dict with 'start' and 'end' timestamps
        vision_data: list of vision frames
        prosody_data: list of prosody frames
        
    Returns:
        tuple: (word_vision_frames, word_prosody_frames)
    """
    start = word.get("start", 0)
    end = word.get("end", 0)
    
    word_vision = [f for f in vision_data 
                   if start <= f.get("timestamp", -1) <= end]
    word_prosody = [p for p in prosody_data 
                    if start <= p.get("timestamp", -1) <= end]
    
    return word_vision, word_prosody


def enrich_transcript(transcript_data, vision_data, prosody_data):
    """
    Master function to merge all data streams onto the transcript timeline.
    
    This function:
    1. Slices vision and prosody data for each word's timespan
    2. Calculates raw metrics (wrist velocity, intensity, pitch)
    3. Applies heuristics to generate tags and insights (including stutter detection)
    4. Performs sentence-level analysis
    
    Args:
        transcript_data: dict with 'words' list from STT
        vision_data: list of vision/pose frames
        prosody_data: list of prosody frames
        
    Returns:
        dict: {
            "words": [...],  # enriched words with metrics, tags, confidence
            "sentence_analysis": {...}  # sentence-level insights including fluency
        }
    """
    words = transcript_data.get("words", [])
    
    if not words:
        return {
            "words": [],
            "sentence_analysis": {
                "avg_confidence": 0,
                "fluency_score": 0,
                "disfluency_count": 0,
                "tag_distribution": {},
                "patterns": [],
                "word_count": 0,
                "total_duration": 0
            }
        }
    
    enriched_words = []
    
    for i, word in enumerate(words):
        try:
            # --- 1. SLICING: Get relevant frames for this word ---
            word_vision, word_prosody = slice_data_for_word(word, vision_data, prosody_data)
            
            # --- 2. ANALYSIS: Calculate raw metrics for this word ---
            
            # Vision Metric: Gesture Intensity
            avg_velocity = calculate_wrist_velocity(word_vision)
            
            # Prosody Metrics: Audio Intensity and Pitch
            avg_intensity, avg_pitch = calculate_prosody_metrics(word_prosody)
            
            # Attach raw metrics to the word
            word["metrics"] = {
                "wrist_velocity": round(avg_velocity, 4),
                "audio_intensity": round(avg_intensity, 4),
                "pitch": round(avg_pitch, 2)
            }
            
            # --- 3. HEURISTICS: Apply all heuristic rules ---
            # Pass context for stutter/false-start detection
            previous_words = enriched_words[-3:] if enriched_words else []
            next_words = words[i+1:i+4] if i+1 < len(words) else []
            
            enriched_word = apply_all_heuristics(
                word=word,
                metrics=word["metrics"],
                prosody_frames=word_prosody,
                word_index=i,
                previous_words=previous_words,
                next_words=next_words
            )
            enriched_words.append(enriched_word)
            
        except Exception as e:
            # If enrichment fails for a word, still include it with empty tags
            print(f"Error enriching word '{word.get('text', '')}': {e}")
            word["metrics"] = {
                "wrist_velocity": 0.0,
                "audio_intensity": 0.0,
                "pitch": 0.0
            }
            word["tags"] = ["error"]
            word["confidence_score"] = 50
            enriched_words.append(word)
    
    # --- 4. SENTENCE-LEVEL ANALYSIS ---
    sentence_analysis = analyze_sentence_patterns(enriched_words)
    
    # --- 5. STRIP DOWN TO ESSENTIAL FIELDS ---
    stripped_words = []
    for word in enriched_words:
        stripped_words.append({
            "text": word.get("text", ""),
            "tags": word.get("tags", []),
            "confidence_score": word.get("confidence_score", 50)
        })
    
    return {
        "words": stripped_words,
        "sentence_analysis": sentence_analysis
    }


def format_enriched_output(transcript_data, vision_data, prosody_data):
    """
    Convenience function that returns a complete enriched result
    matching the expected output format with all metadata.
    
    Args:
        transcript_data: Original transcript dict with 'words' and 'full_text'
        vision_data: List of vision frames
        prosody_data: List of prosody frames
        
    Returns:
        dict: Complete enriched output with all original data plus enrichment
    """
    enriched = enrich_transcript(transcript_data, vision_data, prosody_data)
    
    # Calculate data quality metrics
    def calculate_coverage(data, duration):
        if duration == 0 or not data:
            return 0.0
        covered_time = len(data) * (data[1].get("timestamp", 0) - data[0].get("timestamp", 0)) if len(data) > 1 else 0
        return min(100.0, (covered_time / duration) * 100)
    
    total_duration = words[-1]["end"] if (words := transcript_data.get("words", [])) else 0
    
    return {
        "enriched_transcript": enriched,
        "original_transcript": {
            "full_text": transcript_data.get("full_text", ""),
            "word_count": len(transcript_data.get("words", [])),
            "duration": total_duration
        },
        "data_quality": {
            "vision_coverage": round(calculate_coverage(vision_data, total_duration), 1),
            "prosody_coverage": round(calculate_coverage(prosody_data, total_duration), 1),
            "vision_frame_count": len(vision_data),
            "prosody_frame_count": len(prosody_data)
        }
    }