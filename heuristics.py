"""
heuristics.py

This module contains all the heuristic rules and thresholds for analyzing
multimodal data (speech, prosody, vision) to tag meaningful communication patterns.

These heuristics are designed to identify:
- Gesture emphasis and energy
- Vocal patterns (pitch, intensity, emphasis)
- Hand movement patterns
- Coordination between modalities
- Speech disfluencies (stutters, fillers, hesitations)

Usage:
    from heuristics import apply_all_heuristics, analyze_sentence_patterns
"""

import numpy as np
from collections import Counter

# ============================================================================
# THRESHOLD CONSTANTS
# ============================================================================

class Thresholds:
    """
    Centralized threshold values for all heuristics.
    These can be tuned based on real-world testing and user feedback.
    """
    
    # --- Gesture/Movement Thresholds ---
    HIGH_GESTURE_ENERGY = 0.02          # Wrist velocity indicating active gesturing
    STATIC_HANDS = 0.005                # Velocity below this = minimal movement
    MINIMAL_WORD_DURATION = 0.5         # Seconds - ignore short words for static analysis
    VERY_SHORT_WORD = 0.15              # Words under this are likely stutters/fragments
    
    # --- Audio Intensity Thresholds ---
    VOCAL_EMPHASIS = 0.065              # Audio intensity indicating strong emphasis
    MODERATE_EMPHASIS = 0.05            # Moderate emphasis
    LOW_INTENSITY = 0.035               # Quiet/soft speaking
    VERY_LOW_INTENSITY = 0.02           # Very quiet (hesitation, trail-off)
    
    # --- Pitch Thresholds ---
    HIGH_PITCH = 350.0                  # Hz - elevated pitch
    VERY_HIGH_PITCH = 450.0             # Hz - very elevated (excitement/stress)
    LOW_PITCH = 120.0                   # Hz - low pitch (confidence/finality) - lowered from 180
    PITCH_RISE = 60.0                   # Hz change indicating rising intonation
    PITCH_FALL = 60.0                   # Hz change indicating falling intonation
    PITCH_INSTABILITY = 150.0           # Hz variance indicating pitch wobble - increased from 80
    
    # --- Combined/Advanced Thresholds ---
    EMPHASIS_GESTURE_COMBO = 0.015      # Lower gesture threshold when combined with vocal emphasis
    FILLER_WORD_INTENSITY = 0.045       # Typical intensity of filler words
    
    # --- Pacing Thresholds ---
    VERY_FAST_WORD = 0.15               # Words under this duration = rushed
    FAST_WORD = 0.25                    # Quick delivery
    SLOW_WORD = 0.8                     # Deliberate/slow delivery
    VERY_SLOW_WORD = 1.2                # Very deliberate (emphasis or difficulty)
    
    # --- Pause Thresholds ---
    SHORT_PAUSE = 0.3                   # Brief pause between words
    LONG_PAUSE = 0.8                    # Significant pause (thinking/emphasis)
    VERY_LONG_PAUSE = 1.5               # Very long pause (major break)
    
    # --- Intensity Change Thresholds ---
    INTENSITY_SPIKE = 0.03              # Sudden increase in loudness
    INTENSITY_DROP = 0.02               # Sudden decrease in loudness


# ============================================================================
# FILLER WORDS AND SPEECH PATTERNS
# ============================================================================

# Common filler words and hesitation markers
FILLER_WORDS = {
    # Classic fillers
    "um", "uh", "er", "ah", "eh", "hmm", "mm", "mhm", "uh-huh",
    
    # Discourse markers (often fillers)
    "like", "you know", "i mean", "sort of", "kind of", 
    "basically", "actually", "literally", "honestly",
    "right", "okay", "ok", "alright", "yeah", "yep",
    
    # Thinking markers
    "well", "so", "now", "let's see", "let me think",
    
    # False starts
    "i", "the", "a", "and", "but", "or"  # When very short, often false starts
}

# Words that when repeated indicate stuttering
STUTTER_PRONE_WORDS = {
    "i", "the", "a", "and", "but", "to", "is", "was", "be",
    "it", "that", "you", "he", "she", "we", "they"
}

# Question words that typically have rising intonation
QUESTION_WORDS = {
    "what", "when", "where", "who", "why", "how", "which",
    "is", "are", "can", "could", "would", "should", "do", "does", "did"
}

# Strong assertion words
ASSERTION_WORDS = {
    "definitely", "absolutely", "certainly", "surely", "clearly",
    "obviously", "never", "always", "must", "will", "really"
}

# Uncertainty markers
UNCERTAINTY_WORDS = {
    "maybe", "perhaps", "possibly", "probably", "might", "may",
    "seem", "seems", "appears", "guess", "suppose", "think"
}


def is_filler_word(word_text):
    """Check if a word is a common filler/hesitation word."""
    cleaned = word_text.lower().strip(".,!?;:'\"")
    return cleaned in FILLER_WORDS


def is_question_word(word_text):
    """Check if a word typically starts questions."""
    cleaned = word_text.lower().strip(".,!?;:'\"")
    return cleaned in QUESTION_WORDS


def is_assertion_word(word_text):
    """Check if a word indicates strong assertion."""
    cleaned = word_text.lower().strip(".,!?;:'\"")
    return cleaned in ASSERTION_WORDS


def is_uncertainty_word(word_text):
    """Check if a word indicates uncertainty."""
    cleaned = word_text.lower().strip(".,!?;:'\"")
    return cleaned in UNCERTAINTY_WORDS


def detect_stutter_pattern(current_word, previous_words, word_index):
    """
    Detect if current word is part of a stutter pattern.
    
    Looks for:
    - Repeated words: "I I think"
    - Repeated syllables: "th-th-the"
    - Very short repeated fragments
    
    Args:
        current_word: dict with current word data
        previous_words: list of previous word dicts
        word_index: current word index in full transcript
        
    Returns:
        bool: True if stutter pattern detected
    """
    if word_index == 0:
        return False
    
    current_text = current_word["text"].lower().strip(".,!?;:'\"")
    current_duration = current_word["end"] - current_word["start"]
    
    # Very short words followed by similar words = stutter
    if current_duration < Thresholds.VERY_SHORT_WORD:
        if word_index > 0:
            prev_text = previous_words[-1]["text"].lower().strip(".,!?;:'\"")
            
            # Exact repetition
            if current_text == prev_text:
                return True
            
            # Partial repetition (syllable stutter)
            if len(current_text) <= 3 and prev_text.startswith(current_text):
                return True
    
    # Check for repeated small words (I I, the the)
    if current_text in STUTTER_PRONE_WORDS and word_index > 0:
        prev_text = previous_words[-1]["text"].lower().strip(".,!?;:'\"")
        if current_text == prev_text:
            return True
    
    return False


def detect_false_start(current_word, next_words, word_index):
    """
    Detect false starts: speaker starts a word/phrase then abandons it.
    
    Examples:
    - "I uh I mean"
    - "The the situation"
    - "We we're going"
    
    Args:
        current_word: current word dict
        next_words: list of upcoming words
        word_index: current index
        
    Returns:
        bool: True if false start detected
    """
    if not next_words:
        return False
    
    current_text = current_word["text"].lower().strip(".,!?;:'\"")
    current_duration = current_word["end"] - current_word["start"]
    
    # Short word followed by filler followed by restart
    if current_duration < 0.3 and current_text in STUTTER_PRONE_WORDS:
        if len(next_words) >= 2:
            next_is_filler = is_filler_word(next_words[0]["text"])
            if next_is_filler:
                return True
    
    return False


# ============================================================================
# CORE HEURISTIC FUNCTIONS
# ============================================================================

def analyze_pause_before_word(word, previous_word):
    """
    Analyze the pause/gap between this word and the previous word.
    
    Args:
        word: Current word dict with 'start'
        previous_word: Previous word dict with 'end'
        
    Returns:
        list: Tags related to pausing
    """
    tags = []
    
    if not previous_word:
        return tags
    
    pause_duration = word["start"] - previous_word["end"]
    
    if pause_duration < 0:  # Overlapping (shouldn't happen but handle it)
        return tags
    
    if pause_duration > Thresholds.VERY_LONG_PAUSE:
        tags.append("very_long_pause_before")
    elif pause_duration > Thresholds.LONG_PAUSE:
        tags.append("long_pause_before")
    elif pause_duration > Thresholds.SHORT_PAUSE:
        tags.append("short_pause_before")
    
    return tags


def analyze_intensity_change(prosody_frames):
    """
    Detect sudden changes in intensity within a word (spikes or drops).
    
    Returns:
        list: Tags related to intensity dynamics
    """
    tags = []
    
    if not prosody_frames or len(prosody_frames) < 3:
        return tags
    
    intensities = [p["intensity"] for p in prosody_frames 
                   if p["intensity"] is not None]
    
    if len(intensities) < 3:
        return tags
    
    # Calculate frame-to-frame changes
    changes = [abs(intensities[i] - intensities[i-1]) 
               for i in range(1, len(intensities))]
    
    max_change = max(changes) if changes else 0
    
    # Detect sudden spikes
    if max_change > Thresholds.INTENSITY_SPIKE:
        tags.append("intensity_spike")
    
    # Detect if intensity is increasing or decreasing overall
    if len(intensities) >= 4:
        first_half_avg = np.mean(intensities[:len(intensities)//2])
        second_half_avg = np.mean(intensities[len(intensities)//2:])
        
        if second_half_avg - first_half_avg > Thresholds.INTENSITY_SPIKE:
            tags.append("crescendo")
        elif first_half_avg - second_half_avg > Thresholds.INTENSITY_DROP:
            tags.append("decrescendo")
    
    return tags


def analyze_gesture_energy(wrist_velocity, word_duration):
    """
    Analyze hand/wrist movement patterns.
    
    Returns:
        list: Tags related to gesture energy
    """
    tags = []
    
    # High energy gesturing
    if wrist_velocity > Thresholds.HIGH_GESTURE_ENERGY:
        tags.append("high_gesture_energy")
    elif wrist_velocity > Thresholds.EMPHASIS_GESTURE_COMBO:
        tags.append("moderate_gesture")
    
    # Static hands (only for words long enough to be meaningful)
    if word_duration > Thresholds.MINIMAL_WORD_DURATION and wrist_velocity < Thresholds.STATIC_HANDS:
        tags.append("static_hands")
    
    return tags


def analyze_vocal_patterns(avg_intensity, avg_pitch, word_text, word_duration):
    """
    Analyze vocal/prosodic features.
    
    Returns:
        list: Tags related to vocal patterns
    """
    tags = []
    
    # Vocal emphasis levels
    if avg_intensity > Thresholds.VOCAL_EMPHASIS:
        tags.append("strong_vocal_emphasis")
    elif avg_intensity > Thresholds.MODERATE_EMPHASIS:
        tags.append("vocal_emphasis")
    
    # Soft/quiet speaking
    if avg_intensity < Thresholds.VERY_LOW_INTENSITY:
        tags.append("very_soft_spoken")
    elif avg_intensity < Thresholds.LOW_INTENSITY:
        tags.append("soft_spoken")
    
    # Pitch analysis
    if avg_pitch > Thresholds.VERY_HIGH_PITCH:
        tags.append("very_high_pitch")
    elif avg_pitch > Thresholds.HIGH_PITCH:
        tags.append("high_pitch")
    elif avg_pitch > 50 and avg_pitch < Thresholds.LOW_PITCH:  # Only tag if meaningfully low
        tags.append("low_pitch")
    
    # Pacing analysis based on word duration
    if word_duration < Thresholds.VERY_FAST_WORD:
        tags.append("very_fast")
    elif word_duration < Thresholds.FAST_WORD:
        tags.append("fast_paced")
    elif word_duration > Thresholds.VERY_SLOW_WORD:
        tags.append("very_slow")
    elif word_duration > Thresholds.SLOW_WORD:
        tags.append("slow_deliberate")
    
    # Filler word detection with acoustic features
    if is_filler_word(word_text):
        tags.append("filler_word")
        
        # Hesitant filler (low intensity + duration)
        if avg_intensity < Thresholds.FILLER_WORD_INTENSITY and word_duration > 0.3:
            tags.append("hesitation")
    
    # Semantic tags for word meaning
    if is_question_word(word_text):
        tags.append("question_word")
    
    if is_assertion_word(word_text):
        tags.append("assertion")
    
    if is_uncertainty_word(word_text):
        tags.append("uncertainty_marker")
    
    return tags


def analyze_pitch_contour(prosody_frames):
    """
    Analyze pitch changes over time within a word.
    
    Returns:
        list: Tags related to pitch movement and stability
    """
    tags = []
    
    if not prosody_frames or len(prosody_frames) < 4:  # Require more frames for stability
        return tags
    
    # Filter out outliers - be more aggressive
    pitches = [p["pitch"] for p in prosody_frames 
               if p["pitch"] is not None and 70 < p["pitch"] < 800]
    
    if len(pitches) < 4:  # Need enough data points
        return tags
    
    # Calculate pitch statistics
    pitch_change = pitches[-1] - pitches[0]
    pitch_std = np.std(pitches)
    pitch_variance = pitch_std ** 2
    
    # Rising intonation (questions, uncertainty, continuation)
    if pitch_change > Thresholds.PITCH_RISE:
        tags.append("rising_intonation")
    
    # Falling intonation (statements, finality, confidence)
    if pitch_change < -Thresholds.PITCH_FALL:
        tags.append("falling_intonation")
    
    # Pitch instability (nervousness, uncertainty) - use standard deviation instead
    # Only flag if there's significant wobbling AND the word is long enough
    if pitch_std > 40 and len(pitches) > 5:
        tags.append("pitch_wobble")
    
    return tags


def analyze_multimodal_emphasis(wrist_velocity, avg_intensity, avg_pitch):
    """
    Detect emphasis through coordination of multiple modalities.
    Strong emphasis often involves BOTH gesture and vocal energy.
    
    Returns:
        list: Tags for multimodal patterns
    """
    tags = []
    
    # Combined gesture + vocal emphasis = strong emphasis
    if (wrist_velocity > Thresholds.EMPHASIS_GESTURE_COMBO and 
        avg_intensity > Thresholds.VOCAL_EMPHASIS):
        tags.append("strong_emphasis")
    elif (wrist_velocity > Thresholds.EMPHASIS_GESTURE_COMBO and 
          avg_intensity > Thresholds.MODERATE_EMPHASIS):
        tags.append("moderate_emphasis")
    
    # High pitch + gesture = excitement/animated speech
    if (wrist_velocity > Thresholds.EMPHASIS_GESTURE_COMBO and 
        avg_pitch > Thresholds.HIGH_PITCH):
        tags.append("animated")
    
    # High pitch + high intensity = excitement/passion
    if (avg_pitch > Thresholds.HIGH_PITCH and 
        avg_intensity > Thresholds.VOCAL_EMPHASIS):
        tags.append("passionate")
    
    # Low energy across all modalities = low confidence/trail-off
    if (wrist_velocity < Thresholds.STATIC_HANDS and 
        avg_intensity < Thresholds.LOW_INTENSITY):
        tags.append("low_energy")
    
    # Gesture without voice = trying to find words
    if (wrist_velocity > Thresholds.EMPHASIS_GESTURE_COMBO and 
        avg_intensity < Thresholds.LOW_INTENSITY):
        tags.append("searching_for_words")
    
    return tags


def calculate_confidence_score(metrics, tags, word_duration):
    """
    Calculate a confidence score (0-100) for how "confident" the delivery seems.
    
    Factors that increase confidence:
    - Steady intensity (not too soft)
    - Falling intonation (declarative)
    - Moderate gesture energy (not static, not frantic)
    - No hesitation markers
    - Longer word duration (deliberate speech)
    - Assertion words
    
    Returns:
        int: Confidence score 0-100
    """
    score = 50  # Start neutral
    
    # Positive factors
    if "falling_intonation" in tags:
        score += 15
    if "strong_vocal_emphasis" in tags or "vocal_emphasis" in tags:
        score += 12
    if "low_pitch" in tags:
        score += 8  # Lower pitch often = confidence
    if "assertion" in tags:
        score += 10
    if "passionate" in tags:
        score += 8
    if metrics["wrist_velocity"] > 0.01 and metrics["wrist_velocity"] < 0.03:
        score += 10  # Moderate, purposeful gestures
    if metrics["audio_intensity"] > 0.045:
        score += 10
    if word_duration > 0.5:
        score += 5  # Longer words = more deliberate
    if "slow_deliberate" in tags:
        score += 8
    if "crescendo" in tags:
        score += 6  # Building intensity = emphasis
    
    # Negative factors
    if "stutter" in tags or "false_start" in tags:
        score -= 25
    if "hesitation" in tags or "filler_word" in tags:
        score -= 15
    if "uncertainty_marker" in tags:
        score -= 12
    if "static_hands" in tags and word_duration > 0.5:
        score -= 8
    if "soft_spoken" in tags or "very_soft_spoken" in tags:
        score -= 12
    if "rising_intonation" in tags and "question_word" not in tags:
        score -= 7  # Rising intonation on non-questions = uncertainty
    if "pitch_wobble" in tags:
        score -= 10
    if "low_energy" in tags:
        score -= 15
    if "searching_for_words" in tags:
        score -= 12
    if "very_fast" in tags:
        score -= 8  # Rushed speech can indicate nervousness
    if "long_pause_before" in tags or "very_long_pause_before" in tags:
        score -= 10
    if "decrescendo" in tags:
        score -= 5  # Trailing off
    
    # Clamp to 0-100
    return max(0, min(100, score))


# ============================================================================
# MASTER HEURISTIC APPLICATION
# ============================================================================

def apply_all_heuristics(word, metrics, prosody_frames, word_index=0, 
                        previous_words=None, next_words=None):
    """
    Apply all heuristic rules to a single word.
    
    Args:
        word: dict with 'text', 'start', 'end'
        metrics: dict with 'wrist_velocity', 'audio_intensity', 'pitch'
        prosody_frames: list of prosody data points for this word's timespan
        word_index: position in full transcript
        previous_words: list of previous enriched words
        next_words: list of upcoming words (not yet enriched)
    
    Returns:
        dict: Updated word with 'tags' and 'confidence_score'
    """
    previous_words = previous_words or []
    next_words = next_words or []
    
    word_duration = word["end"] - word["start"]
    word_text = word["text"]
    
    # Extract metrics
    wrist_velocity = metrics["wrist_velocity"]
    avg_intensity = metrics["audio_intensity"]
    avg_pitch = metrics["pitch"]
    
    # Apply all heuristic categories
    tags = []
    
    # Gesture analysis
    tags.extend(analyze_gesture_energy(wrist_velocity, word_duration))
    
    # Vocal/prosody analysis
    tags.extend(analyze_vocal_patterns(avg_intensity, avg_pitch, word_text, word_duration))
    
    # Pitch contour analysis
    tags.extend(analyze_pitch_contour(prosody_frames))
    
    # Intensity dynamics
    tags.extend(analyze_intensity_change(prosody_frames))
    
    # Multimodal coordination
    tags.extend(analyze_multimodal_emphasis(wrist_velocity, avg_intensity, avg_pitch))
    
    # Pause analysis (needs previous word)
    if previous_words:
        tags.extend(analyze_pause_before_word(word, previous_words[-1]))
    
    # Speech disfluency detection
    if detect_stutter_pattern(word, previous_words, word_index):
        tags.append("stutter")
    
    if detect_false_start(word, next_words, word_index):
        tags.append("false_start")
    
    # Remove duplicates while preserving order
    tags = list(dict.fromkeys(tags))
    
    # Calculate confidence score
    confidence = calculate_confidence_score(metrics, tags, word_duration)
    
    return {
        **word,
        "tags": tags,
        "confidence_score": confidence
    }


# ============================================================================
# SENTENCE-LEVEL ANALYSIS
# ============================================================================

def analyze_sentence_patterns(words):
    """
    Analyze patterns across multiple words (sentence-level features).
    
    Args:
        words: list of enriched word dicts
    
    Returns:
        dict: Sentence-level insights
    """
    if not words:
        return {}
    
    # Calculate sentence-level metrics
    avg_confidence = np.mean([w.get("confidence_score", 50) for w in words])
    
    # Count tag frequencies
    all_tags = []
    for w in words:
        all_tags.extend(w.get("tags", []))
    
    tag_counts = Counter(all_tags)
    
    # Detect overall patterns
    patterns = []
    
    # Disfluency patterns
    disfluency_count = (tag_counts.get("hesitation", 0) + 
                        tag_counts.get("filler_word", 0) +
                        tag_counts.get("stutter", 0) +
                        tag_counts.get("false_start", 0))
    
    if disfluency_count >= 3:
        patterns.append("frequent_disfluencies")
    elif disfluency_count >= len(words) * 0.3:
        patterns.append("some_disfluencies")
    
    # Gesture patterns
    if tag_counts.get("high_gesture_energy", 0) >= len(words) * 0.5:
        patterns.append("highly_animated")
    elif tag_counts.get("moderate_gesture", 0) >= len(words) * 0.4:
        patterns.append("moderately_animated")
    
    if tag_counts.get("static_hands", 0) >= len(words) * 0.6:
        patterns.append("minimal_gesture")
    
    # Energy patterns
    if tag_counts.get("low_energy", 0) >= len(words) * 0.4:
        patterns.append("low_energy_delivery")
    
    # Confidence patterns
    if avg_confidence >= 70:
        patterns.append("confident_delivery")
    elif avg_confidence <= 40:
        patterns.append("uncertain_delivery")
    
    # Vocal patterns
    emphasis_count = tag_counts.get("strong_vocal_emphasis", 0) + tag_counts.get("vocal_emphasis", 0)
    if emphasis_count >= len(words) * 0.4:
        patterns.append("emphatic_speech")
    
    # Pacing patterns
    fast_count = tag_counts.get("very_fast", 0) + tag_counts.get("fast_paced", 0)
    slow_count = tag_counts.get("very_slow", 0) + tag_counts.get("slow_deliberate", 0)
    
    if fast_count >= len(words) * 0.5:
        patterns.append("rapid_speech")
    elif slow_count >= len(words) * 0.5:
        patterns.append("deliberate_speech")
    
    # Pause patterns
    pause_count = (tag_counts.get("long_pause_before", 0) + 
                   tag_counts.get("very_long_pause_before", 0))
    if pause_count >= 3:
        patterns.append("frequent_pauses")
    
    # Emotional/intensity patterns
    if tag_counts.get("passionate", 0) >= len(words) * 0.3:
        patterns.append("passionate_delivery")
    
    if tag_counts.get("animated", 0) >= len(words) * 0.3:
        patterns.append("animated_delivery")
    
    # Uncertainty patterns
    uncertainty_count = tag_counts.get("uncertainty_marker", 0)
    if uncertainty_count >= 2:
        patterns.append("shows_uncertainty")
    
    # Calculate fluency score (0-100, higher = more fluent)
    fluency_score = max(0, 100 - (disfluency_count * 15) - (pause_count * 5))
    
    # Calculate average speaking rate (words per second)
    total_duration = words[-1]["end"] - words[0]["start"] if words else 0
    speaking_rate = len(words) / total_duration if total_duration > 0 else 0
    
    return {
        "avg_confidence": round(avg_confidence, 1),
        "fluency_score": fluency_score,
        "disfluency_count": disfluency_count,
        "speaking_rate": round(speaking_rate, 2),
        "tag_distribution": dict(tag_counts),
        "patterns": patterns,
        "word_count": len(words),
        "total_duration": round(total_duration, 2)
    }