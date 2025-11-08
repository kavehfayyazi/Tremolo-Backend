import uvicorn
import shutil
import uuid
import os
import json
import modal  # Import Modal
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import boto3
from dotenv import load_dotenv
from STT import transcribe_video
from prosody_processor import analyze_prosody
# Import the new enricher
from enricher import enrich_transcript
from concurrent.futures import ThreadPoolExecutor, as_completed

# ----------------------------------
# Environment & AWS client setup
# ----------------------------------

# Load environment variables from .env file
load_dotenv()

s3_client = boto3.client(
    's3',
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    region_name=os.environ.get("AWS_REGION", "us-east-2")  # Specify your bucket's region
)
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")

# ----------------------------------
# App & CORS configuration
# ----------------------------------

# Read allowed origins from env or use sane defaults (Vercel + localhost)
RAW_ALLOWED = os.getenv(
    "ALLOWED_ORIGINS",
    "https://tremolo-frontend-one.vercel.app,"
    "http://localhost:3000,http://127.0.0.1:3000,"
    "http://localhost:5173,http://127.0.0.1:5173"
)
ALLOWED_ORIGINS = [o.strip() for o in RAW_ALLOWED.split(",") if o.strip()]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,   # list exact origins (best practice)
    allow_credentials=False,         # set True only if you send cookies/auth
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)

@app.get("/healthz")
def healthz():
    return {"ok": True, "allowed_origins": ALLOWED_ORIGINS}

# Explicit preflight (Starlette CORS already handles this; this keeps logs clean)
@app.options("/api/upload-video")
async def options_upload_video():
    return JSONResponse(status_code=204, content=None)

# ----------------------------------
# In-memory job state
# ----------------------------------

JOB_STATUS_DB = {}

# ----------------------------------
# Helpers
# ----------------------------------

def upload_to_public_storage(file_path: str, job_id: str) -> str:
    """
    Uploads a file to S3 and makes it public-read.
    Returns the public URL for the uploaded video.
    """
    if not S3_BUCKET_NAME:
        raise Exception("S3_BUCKET_NAME environment variable not set.")
        
    s3_key = f"uploads/{job_id}.mp4"  # The "folder" and "filename" in S3

    try:
        print(f"Uploading {file_path} to S3 bucket {S3_BUCKET_NAME} as {s3_key}...")
        s3_client.upload_file(
            file_path,
            S3_BUCKET_NAME,
            s3_key,
            ExtraArgs={
                'ACL': 'public-read',  # CRITICAL: This makes the file public
                'ContentType': 'video/mp4'
            }
        )
        
        # Get the public URL with the correct region
        region = s3_client.meta.region_name
        if not region:
            try:
                location = s3_client.get_bucket_location(Bucket=S3_BUCKET_NAME)
                region = location['LocationConstraint'] or 'us-east-1'
            except Exception:
                region = 'us-east-1'
        
        public_url = f"https://{S3_BUCKET_NAME}.s3.{region}.amazonaws.com/{s3_key}"
        print(f"Upload complete. Public URL: {public_url}")
        return public_url

    except Exception as e:
        print(f"S3 upload failed: {e}")
        raise

# ----------------------------------
# Models
# ----------------------------------

class JobStatus(BaseModel):
    status: str
    job_id: str
    data: dict | None = None

# ----------------------------------
# Modal setup
# ----------------------------------

try:
    VisionProcessor = modal.Cls.from_name("Tremolo-Vision", "VisionProcessor")
    VisionAgent = VisionProcessor()
except modal.exception.NotFoundError:
    print("Error: Modal function not found. Did you deploy `modal_processor.py`?")
    VisionAgent = None

# ----------------------------------
# Endpoints
# ----------------------------------

@app.post("/api/upload-video", response_model=JobStatus)
async def upload_video(file: UploadFile = File(...)):
    """
    Upload a video to S3 and kick off a Modal vision job.
    In parallel, run local STT and Prosody analyses.
    """
    if not VisionAgent:
        return JSONResponse(status_code=500, content={"status": "error", "detail": "Vision agent is not deployed."})

    # Validate the multipart field and type early
    if file is None:
        raise HTTPException(status_code=400, detail="Missing file. Send as multipart/form-data with field name 'file'.")
    if file.content_type and not file.content_type.startswith(("video/", "application/octet-stream")):
        raise HTTPException(status_code=415, detail=f"Unsupported content type: {file.content_type}")

    temp_dir = "/tmp"
    os.makedirs(temp_dir, exist_ok=True)
    job_id = str(uuid.uuid4())
    video_path = os.path.join(temp_dir, f"{job_id}.mp4")

    try:
        # 1) Save upload to temp file
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2) Upload to S3 FIRST so Modal can access the file via public URL
        public_video_url = upload_to_public_storage(video_path, job_id)

        # 3) Start the Modal vision job ASYNCHRONOUSLY
        print(f"Spawning Modal job for {job_id} with URL: {public_video_url}")
        call = VisionAgent.analyze.spawn(public_video_url)
        modal_call_id = call.object_id

        # 4) Run STT and Prosody in parallel locally
        print(f"Starting parallel processing (STT + Prosody) for {job_id}...")
        transcription_data = None
        prosody_data = None

        with ThreadPoolExecutor(max_workers=2) as executor:
            stt_future = executor.submit(transcribe_video, public_video_url)
            # Prosody can use the local file path for faster access
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
                    # Keep going; enricher will handle partial data

        # 5) Create job record (keep raw pieces until enrichment happens)
        JOB_STATUS_DB[job_id] = {
            "status": "processing",
            "job_id": job_id,
            "modal_id": modal_call_id,
            "transcription": transcription_data,
            "prosody": prosody_data,
            "vision": None,
            "enriched_transcript": None  # Placeholder for final result
        }

        # 6) Respond with 202 Accepted while background work continues
        return JSONResponse(status_code=202, content={"status": "processing", "job_id": job_id})

    except Exception as e:
        print(f"Upload failed: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "job_id": None, "detail": str(e)})

    finally:
        # 7) Clean up local file regardless of success/failure
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
        except Exception as cleanup_err:
            print(f"Temp file cleanup error for {video_path}: {cleanup_err}")

@app.get("/api/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    job = JOB_STATUS_DB.get(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"status": "error", "detail": "Job not found."})

    if job["status"] == "complete":
        # Return the full job object, already cleaned
        return JSONResponse(status_code=200, content=job)

    # Poll Modal for completion
    try:
        modal_call = modal.FunctionCall.from_id(job["modal_id"])
        try:
            # Non-blocking check; returns immediately if ready
            vision_data = modal_call.get(timeout=0)

            # JOB COMPLETE! Time to enrich.
            print(f"Job {job_id} completed! Running enrichment...")
            job["vision"] = vision_data

            # --- THE ORCHESTRATION STEP ---
            if job.get("transcription") and job["transcription"].get("status") == "completed":
                job["enriched_transcript"] = enrich_transcript(
                    job["transcription"],
                    job["vision"],
                    job.get("prosody")
                )
            else:
                job["enriched_transcript"] = {"error": "Transcription failed or incomplete; cannot enrich."}
            # -----------------------------

            job["status"] = "complete"

            # --- CLEANUP: remove raw bits we don't want to leak to clients ---
            for k in ("prosody", "vision", "modal_id"):
                if k in job:
                    del job[k]
            # -----------------------------------------------------------------

            return JSONResponse(status_code=200, content=job)

        except TimeoutError:
            # Still processing
            return JSONResponse(status_code=200, content={"status": "processing", "job_id": job_id})

    except Exception as e:
        print(f"Status check failed: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})

# ----------------------------------
# Entrypoint
# ----------------------------------

if __name__ == "__main__":
    # 0.0.0.0 for container platforms (Railway); port can be overridden by env
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)