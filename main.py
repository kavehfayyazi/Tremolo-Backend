import uvicorn
import shutil
import uuid
import os
import json
import modal # Import Modal
from fastapi import FastAPI, UploadFile, File, Request
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import boto3
from dotenv import load_dotenv
from ai import generate_feedback, FeedbackList

# Optional deps
try:
    import modal  # optional; may not exist in Railway image
except Exception:
    modal = None

# Your local modules
from STT import transcribe_video
from prosody_processor import analyze_prosody
# Import the new enricher
from enricher import enrich_transcript
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables from .env file
load_dotenv()

s3_client = boto3.client(
    's3',
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    region_name=os.environ.get("AWS_REGION", "us-east-2")  # Specify your bucket's region
)
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")

JOB_STATUS_DB = {}

def upload_to_public_storage(file_path: str, job_id: str) -> str:
    """
    Uploads a file to S3 and makes it public-read.
    Returns the public URL for Modal.
    """
    if not S3_BUCKET_NAME:
        raise Exception("S3_BUCKET_NAME environment variable not set.")
        
    s3_key = f"uploads/{job_id}.mp4" # The "folder" and "filename" in S3

    try:
        print(f"Uploading {file_path} to S3 bucket {S3_BUCKET_NAME} as {s3_key}...")
        s3_client.upload_file(
            file_path,
            S3_BUCKET_NAME,
            s3_key,
            ExtraArgs={
                'ACL': 'public-read', # CRITICAL: This makes the file public
                'ContentType': 'video/mp4'
            }
        )
        
        # Get the public URL with the correct region
        region = s3_client.meta.region_name
        if not region:
            try:
                location = s3_client.get_bucket_location(Bucket=S3_BUCKET_NAME)
                region = location['LocationConstraint'] or 'us-east-1'
            except:
                region = 'us-east-1'
        
        public_url = f"https://{S3_BUCKET_NAME}.s3.{region}.amazonaws.com/{s3_key}"
        print(f"Upload complete. Public URL: {public_url}")
        return public_url

    except Exception as e:
        print(f"S3 upload failed: {e}")
        raise
    

app = FastAPI()

# --- App Configuration
origins = [
    "http://localhost:3000",
    "http://localhost:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class JobStatus(BaseModel):
    status: str
    job_id: str
    data: dict | None = None

try:
    VisionProcessor = modal.Cls.from_name("Tremolo-Vision", "VisionProcessor")
    VisionAgent = VisionProcessor()
except modal.exception.NotFoundError:
    print("Error: Modal function not found. Did you deploy `modal_processor.py`?")
    VisionAgent = None

# --- API Endpoints ---

@app.post("/api/upload-video", response_model=JobStatus)
async def upload_video(file: UploadFile = File(...)):
    """
    This endpoint uploads a video to S3 and starts a Modal job to process it.
    """
    if not VisionAgent:
        return JSONResponse(status_code=500, content={"status":"error", "detail":"Vision agent is not deployed."})

    try:
        job_id = str(uuid.uuid4())
        
        # 1. Save file locally (temporarily)
        temp_dir = "/tmp"
        os.makedirs(temp_dir, exist_ok=True)
        video_path = os.path.join(temp_dir, f"{job_id}.mp4")

        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 2. Upload the file to S3 FIRST so Modal can start
        public_video_url = upload_to_public_storage(video_path, job_id)
        
        # 3. Start the Modal vision job ASYNCHRONOUSLY
        print(f"Spawning Modal job for {job_id} with URL: {public_video_url}")
        call = VisionAgent.analyze.spawn(public_video_url)
        modal_call_id = call.object_id

        # 4. Run STT and Prosody in parallel locally
        print(f"Starting parallel processing (STT + Prosody) for {job_id}...")
        transcription_data = None
        prosody_data = None
        
        # Use ThreadPoolExecutor to run these IO-bound/CPU-bound tasks without blocking
        with ThreadPoolExecutor(max_workers=2) as executor:
            stt_future = executor.submit(transcribe_video, public_video_url)
            # We can run prosody on the local file since we still have it
            prosody_future = executor.submit(analyze_prosody, video_path)
            
            for future in as_completed([stt_future, prosody_future]):
                try:
                    if future == stt_future:
                        transcription_data = future.result()
                        print(f"Transcription completed for {job_id}")
                    elif future == prosody_future:
                        prosody_data = future.result()
                        print(f"Prosody analysis completed for {job_id}")
                except Exception as e:
                    print(f"Subtask failed for {job_id}: {e}")
                    # We'll handle errors by having None data, which enricher will have to handle

        # 5. Clean up local file
        os.remove(video_path)

        # 6. Create job record
        JOB_STATUS_DB[job_id] = {
            "status": "processing", 
            "modal_id": modal_call_id, 
            "transcription": transcription_data,
            "prosody": prosody_data,
            "vision": None,
            "enriched_transcript": None # Placeholder for final result
        }

        return JSONResponse(status_code=202, content={"status": "processing", "job_id": job_id})
        
    except Exception as e:
        print(f"Upload failed: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "job_id": None, "detail": str(e)})

@app.get("/api/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    job = JOB_STATUS_DB.get(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"status": "error", "detail": "Job not found."})
    
    if job["status"] == "complete":
        # Return the full job object, which is now clean
        return JSONResponse(status_code=200, content=job)

    # Poll Modal
    try:
        modal_call = modal.FunctionCall.from_id(job["modal_id"])
        try:
            vision_data = modal_call.get(timeout=0)
            
            # JOB COMPLETE! Time to enrich.
            print(f"Job {job_id} completed! Running enrichment...")
            job["vision"] = vision_data
            
            # --- THE ORCHESTRATION STEP ---
            if job["transcription"] and job["transcription"].get("status") == "completed":
                 job["enriched_transcript"] = enrich_transcript(
                     job["transcription"],
                     job["vision"],
                     job["prosody"]
                 )
            else:
                job["enriched_transcript"] = {"error": "Transcription failed, cannot enrich."}
            # -----------------------------

            job["status"] = "complete"
            safe = {k: v for k, v in job.items() if k not in ("prosody", "vision", "modal_id", "job_id")}
            return JSONResponse(status_code=200, content=safe)

            # --- CLEANUP ---
            # Remove raw data now that it's in the enriched_transcript
            # This cleans the in-memory DB entry and the response payload
            if "prosody" in job:
                del job["prosody"]
            if "vision" in job:
                del job["vision"]
            if "modal_id" in job: # No longer needed by client
                del job["modal_id"]
            # ---------------

            return JSONResponse(status_code=200, content=job)
            
        except TimeoutError:
            return JSONResponse(status_code=200, content={"status": "processing", "job_id": job_id})

    except Exception as e:
        print(f"Status check failed: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})

@app.post("/api/ai", response_model=FeedbackList)
async def generate_ai_feedback(request: Request):
    """
    This endpoint accepts enriched transcript data and returns AI-generated feedback.
    
    Expected input format:
    {
        "enriched_transcript": {
            "words": [
                {
                    "text": "word",
                    "start": 0.0,
                    "end": 0.5,
                    "tags": [...],
                    "confidence_score": 85
                },
                ...
            ]
        }
    }
    """
    print("Generating AI feedback...")
    try:
        # Parse the request body
        data = await request.json()
        
        # Generate feedback using the AI function
        feedback = generate_feedback(data)
        print("Returning feedback...", feedback)
        return feedback
        
    except Exception as e:
        print(f"AI feedback generation failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": str(e)}
        )

# ----------------------------------
# Entrypoint
# ----------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)