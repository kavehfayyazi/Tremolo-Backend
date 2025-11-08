import assemblyai as aai
import os
from dotenv import load_dotenv

load_dotenv()

def format_time(seconds):
    """Convert seconds to secs:millis format"""
    secs = int(seconds)
    millis = int((seconds % 1) * 1000)
    return f"{secs}:{millis:03d}"

def transcribe_video(video_url: str) -> dict:
    """
    Transcribe a video from a URL using AssemblyAI.
    
    Args:
        video_url: Public URL of the video file
        
    Returns:
        dict with transcription data including:
        - full_text: Complete transcription
        - words: List of word objects with text, start, and end times
        - status: 'completed' or 'error'
    """
    # Set API key from environment variable
    api_key = os.environ.get("ASSEMBLYAI_API_KEY")
    if not api_key:
        raise ValueError("ASSEMBLYAI_API_KEY environment variable not set")
    
    aai.settings.api_key = api_key
    
    # Configure transcription with universal model and disfluencies
    config = aai.TranscriptionConfig(
        speech_model=aai.SpeechModel.universal, 
        disfluencies=True
    )
    
    # Transcribe the video
    transcript = aai.Transcriber(config=config).transcribe(video_url)
    
    # Check for errors
    if transcript.status == "error":
        return {
            "status": "error",
            "error": transcript.error,
            "full_text": None,
            "words": []
        }
    
    # Format word data with timestamps
    words_data = []
    for word in transcript.words:
        words_data.append({
            "text": word.text,
            "start": word.start / 1000,  # Convert to seconds
            "end": word.end / 1000,      # Convert to seconds
            "start_formatted": format_time(word.start / 1000),
            "end_formatted": format_time(word.end / 1000)
        })
    
    return {
        "status": "completed",
        "full_text": transcript.text,
        "words": words_data,
        "error": None
    }


# Test code - only runs when script is executed directly
if __name__ == "__main__":
    test_url = "https://noahdev-tremolo-19.s3.us-east-2.amazonaws.com/Kevin+Surace+1+Minute+Ted+Talk+-+Eagles+Talent+Speakers+Bureau+(720p%2C+h264).mp4"
    result = transcribe_video(test_url)
    
    if result["status"] == "completed":
        print("Transcription:", result["full_text"])
        print("\nWords with timestamps:")
        for word in result["words"]:
            print(f"{word['text']} ({word['start_formatted']})")
    else:
        print(f"Error: {result['error']}")
