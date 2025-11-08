from openai import OpenAI
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List
import json

class FeedbackItem(BaseModel):
    timestamp: float
    feedback: str

class FeedbackList(BaseModel):
    feedback: List[FeedbackItem]


load_dotenv()

SYSTEM_PROMPT = """You are an expert AI public speaking analyst trained to evaluate human presentation performance from noisy multimodal data (transcripts, timestamps, gesture metrics, pitch, intensity, etc.).

Your goal:
Given unstructured or messy input data describing a speaker's performance, extract and summarize the most meaningful **public speaking feedback**. You must identify specific moments worth commenting on, using the available timing information.

Your output must be a **strict JSON array**, where each element has this exact structure:

[
  {
    "timestamp": <float | string>,     // approximate time in seconds or formatted like "0:46"
    "feedback": <string>               // clear, actionable, and specific advice for that moment
  }
]

Formatting rules:
- Output only valid JSON (no explanations, no markdown, no prose).
- Use **2–8 feedback entries** maximum per minute of input.
- Each feedback entry should be concise (≤180 characters) but actionable (avoid generic praise).
- Prefer timestamps from the data if present; otherwise, estimate or use "unknown".
- Combine nearby repeated filler words or gestures into one aggregated feedback item.
- If the input is incomplete or inconsistent, still produce the best possible feedback using reasonable inference.
- Do not hallucinate metrics or events that contradict the data.
- Avoid repetition. Every feedback message must be unique and refer to a distinct issue or strength.
- Include both constructive criticism (e.g., pacing, clarity, gesture usage) and positive reinforcement (e.g., confidence, good emphasis) where appropriate.
- Note: the object in the end of the JSON has basic information about the speech, which can be used to generate general feedback. 




Evaluation priorities:
1. Speaking clarity and filler words
2. Vocal delivery (pace, pitch, volume variation)
3. Gestures and physical presence
4. Engagement and confidence

Example output:
[
  { "timestamp": 0.6, "feedback": "Avoid starting with filler words like 'Uh'—pause before beginning." },
  { "timestamp": 7.4, "feedback": "Good emphasis on 'important,' but slow down slightly to let the idea land." },
  { "timestamp": 9.2, "feedback": "End confidently—avoid filler endings like 'so' or 'right'." }
]
"""

content_mock = """Here is the messy data:"enriched_transcript": {
        "words": [
            {
                "text": "Uh,",
                "start": 0.46,
                "end": 0.64,
                "start_formatted": "0:460",
                "end_formatted": "0:640",
                "tags": [
                    "moderate_gesture",
                    "strong_vocal_emphasis",
                    "fast_paced",
                    "filler_word",
                    "decrescendo",
                    "strong_emphasis"
                ],
                "confidence_score": 62
            },
            {
                "text": "good",
                "start": 0.96,
                "end": 1.08,
                "start_formatted": "0:960",
                "end_formatted": "1:080",
                "tags": [
                    "strong_vocal_emphasis",
                    "very_fast",
                    "intensity_spike",
                    "crescendo",
                    "short_pause_before"
                ],
                "confidence_score": 70
            },
            {
                "text": "evening,",
                "start": 1.08,
                "end": 1.32,
                "start_formatted": "1:080",
                "end_formatted": "1:320",
                "tags": [
                    "high_gesture_energy",
                    "strong_vocal_emphasis",
                    "fast_paced",
                    "strong_emphasis"
                ],
                "confidence_score": 82
            },
            {
                "text": "everyone.",
                "start": 1.32,
                "end": 1.6,
                "start_formatted": "1:320",
                "end_formatted": "1:600",
                "tags": [
                    "strong_vocal_emphasis"
                ],
                "confidence_score": 72
            },
            {
                "text": "Um,",
                "start": 2.2,
                "end": 2.43,
                "start_formatted": "2:200",
                "end_formatted": "2:430",
                "tags": [
                    "strong_vocal_emphasis",
                    "fast_paced",
                    "filler_word",
                    "decrescendo",
                    "short_pause_before"
                ],
                "confidence_score": 52
            },
            {
                "text": "I",
                "start": 3.04,
                "end": 3.32,
                "start_formatted": "3:040",
                "end_formatted": "3:319",
                "tags": [
                    "moderate_gesture",
                    "filler_word",
                    "falling_intonation",
                    "pitch_wobble",
                    "intensity_spike",
                    "crescendo",
                    "short_pause_before"
                ],
                "confidence_score": 56
            },
            {
                "text": "want",
                "start": 3.32,
                "end": 3.48,
                "start_formatted": "3:319",
                "end_formatted": "3:480",
                "tags": [
                    "strong_vocal_emphasis",
                    "fast_paced",
                    "intensity_spike",
                    "decrescendo"
                ],
                "confidence_score": 67
            },
            {
                "text": "to",
                "start": 3.48,
                "end": 3.6,
                "start_formatted": "3:480",
                "end_formatted": "3:600",
                "tags": [
                    "high_gesture_energy",
                    "strong_vocal_emphasis",
                    "very_fast",
                    "intensity_spike",
                    "strong_emphasis"
                ],
                "confidence_score": 64
            },
            {
                "text": "talk",
                "start": 3.6,
                "end": 3.72,
                "start_formatted": "3:600",
                "end_formatted": "3:720",
                "tags": [
                    "high_gesture_energy",
                    "strong_vocal_emphasis",
                    "very_fast",
                    "intensity_spike",
                    "strong_emphasis"
                ],
                "confidence_score": 64
            },
            {
                "text": "about.",
                "start": 3.72,
                "end": 4.0,
                "start_formatted": "3:720",
                "end_formatted": "4:000",
                "tags": [
                    "high_gesture_energy",
                    "strong_vocal_emphasis",
                    "intensity_spike",
                    "decrescendo",
                    "strong_emphasis"
                ],
                "confidence_score": 67
            },
            {
                "text": "Uh,",
                "start": 5.16,
                "end": 5.36,
                "start_formatted": "5:160",
                "end_formatted": "5:360",
                "tags": [
                    "vocal_emphasis",
                    "fast_paced",
                    "filler_word",
                    "decrescendo",
                    "long_pause_before"
                ],
                "confidence_score": 42
            },
            {
                "text": "sorry.",
                "start": 5.36,
                "end": 5.84,
                "start_formatted": "5:360",
                "end_formatted": "5:839",
                "tags": [
                    "high_gesture_energy",
                    "strong_vocal_emphasis",
                    "very_high_pitch",
                    "pitch_wobble",
                    "intensity_spike",
                    "decrescendo",
                    "strong_emphasis",
                    "animated",
                    "passionate"
                ],
                "confidence_score": 75
            },
            {
                "text": "Uh,",
                "start": 6.26,
                "end": 6.44,
                "start_formatted": "6:259",
                "end_formatted": "6:440",
                "tags": [
                    "strong_vocal_emphasis",
                    "fast_paced",
                    "filler_word",
                    "intensity_spike",
                    "short_pause_before"
                ],
                "confidence_score": 57
            },
            {
                "text": "the",
                "start": 6.44,
                "end": 6.6,
                "start_formatted": "6:440",
                "end_formatted": "6:599",
                "tags": [
                    "high_gesture_energy",
                    "strong_vocal_emphasis",
                    "fast_paced",
                    "filler_word",
                    "rising_intonation",
                    "intensity_spike",
                    "strong_emphasis"
                ],
                "confidence_score": 50
            },
            {
                "text": "first",
                "start": 6.6,
                "end": 6.76,
                "start_formatted": "6:599",
                "end_formatted": "6:759",
                "tags": [
                    "high_gesture_energy",
                    "strong_vocal_emphasis",
                    "fast_paced",
                    "intensity_spike",
                    "decrescendo",
                    "strong_emphasis"
                ],
                "confidence_score": 67
            },
            {
                "text": "thing",
                "start": 6.76,
                "end": 7.0,
                "start_formatted": "6:759",
                "end_formatted": "7:000",
                "tags": [
                    "strong_vocal_emphasis",
                    "fast_paced",
                    "pitch_wobble",
                    "intensity_spike",
                    "decrescendo"
                ],
                "confidence_score": 57
            },
            {
                "text": "is",
                "start": 7.0,
                "end": 7.24,
                "start_formatted": "7:000",
                "end_formatted": "7:240",
                "tags": [
                    "strong_vocal_emphasis",
                    "fast_paced",
                    "question_word",
                    "intensity_spike"
                ],
                "confidence_score": 82
            },
            {
                "text": "really",
                "start": 7.24,
                "end": 7.36,
                "start_formatted": "7:240",
                "end_formatted": "7:360",
                "tags": [
                    "strong_vocal_emphasis",
                    "very_fast",
                    "assertion"
                ],
                "confidence_score": 74
            },
            {
                "text": "important.",
                "start": 7.36,
                "end": 7.6,
                "start_formatted": "7:360",
                "end_formatted": "7:599",
                "tags": [
                    "strong_vocal_emphasis",
                    "fast_paced",
                    "intensity_spike",
                    "crescendo"
                ],
                "confidence_score": 78
            },
            {
                "text": "It's",
                "start": 7.68,
                "end": 8.04,
                "start_formatted": "7:679",
                "end_formatted": "8:039",
                "tags": [
                    "moderate_gesture",
                    "vocal_emphasis",
                    "intensity_spike",
                    "moderate_emphasis"
                ],
                "confidence_score": 82
            },
            {
                "text": "innovation",
                "start": 8.04,
                "end": 8.52,
                "start_formatted": "8:039",
                "end_formatted": "8:519",
                "tags": [
                    "strong_vocal_emphasis",
                    "intensity_spike",
                    "decrescendo"
                ],
                "confidence_score": 67
            },
            {
                "text": "and.",
                "start": 8.52,
                "end": 8.84,
                "start_formatted": "8:519",
                "end_formatted": "8:839",
                "tags": [
                    "strong_vocal_emphasis",
                    "filler_word",
                    "rising_intonation",
                    "pitch_wobble"
                ],
                "confidence_score": 40
            },
            {
                "text": "Right.",
                "start": 8.84,
                "end": 9.12,
                "start_formatted": "8:839",
                "end_formatted": "9:119",
                "tags": [
                    "strong_vocal_emphasis",
                    "high_pitch",
                    "filler_word",
                    "pitch_wobble",
                    "intensity_spike",
                    "crescendo",
                    "passionate"
                ],
                "confidence_score": 71
            },
            {
                "text": "Okay,",
                "start": 9.12,
                "end": 9.44,
                "start_formatted": "9:119",
                "end_formatted": "9:439",
                "tags": [
                    "vocal_emphasis",
                    "filler_word",
                    "falling_intonation",
                    "pitch_wobble",
                    "intensity_spike"
                ],
                "confidence_score": 62
            },
            {
                "text": "so.",
                "start": 9.44,
                "end": 9.76,
                "start_formatted": "9:439",
                "end_formatted": "9:759",
                "tags": [
                    "strong_vocal_emphasis",
                    "filler_word",
                    "intensity_spike",
                    "decrescendo"
                ],
                "confidence_score": 52
            }
        ],
"""

def generate_feedback(enriched_data: dict) -> FeedbackList:
    """
    Generate AI feedback from enriched transcript data.
    
    Args:
        enriched_data: Dictionary containing enriched transcript with words, tags, timestamps, etc.
        
    Returns:
        FeedbackList object containing list of feedback items with timestamps
    """
    # Initialize OpenAI client
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )
    
    # Format the input data as a string
    # content = f"Here is the messy data: {json.dumps(enriched_data)}"
    content = content_mock
    
    # Call the AI to generate feedback
    response = client.responses.parse(
        model="gpt-5",
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        text_format=FeedbackList,
    )
    
    return response.output_parsed
