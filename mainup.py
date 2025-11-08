import uvicorn
import shutil
import uuid
import os
import json
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import boto3
import prosody_processor
from dotenv import load_dotenv

load_dotenv();

# --- AWS / Storage ---

s3_client = boto3.client(
    's3',
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY")
)
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")

def upload_to_public_storage(file_path: str, job_id: str) -> str:
    """
    Uploads a file to S3 and makes it public-read.
    Returns the public URL.
    """
    if not S3_BUCKET_NAME:
        raise Exception("S3_BUCKET_NAME environment variable not set.")
        
    s3_key = f"uploads/{job_id}.mp4"

    try:
        print(f"Uploading {file_path} to S3 bucket {S3_BUCKET_NAME} as {s3_key}...")
        s3_client.upload_file(
            file_path,
            S3_BUCKET_NAME,
            s3_key,
            ExtraArgs={
                'ACL': 'public-read',
                'ContentType': 'video/mp4'
            }
        )
        print("Upload successful.")
    except Exception as e:
        print(f"S3 upload failed: {e}")
        raise e

    # Construct the public URL (adjust if you need region-specific)
    s3_url = f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{s3_key}"
    print(f"File available at public URL: {s3_url}")
    return s3_url


# --- In-memory "DB" for results (no background jobs now) ---

JOB_DB = {}  # job_id -> {"status": "...", "prosody_data": ..., "s3_url": ...}


# --- FastAPI App Setup ---

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "Tremolo Backend is running (no Modal)!"}


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    1) Receives video file
    2) Saves it to /tmp
    3) Runs local prosody analysis
    4) Uploads the video to S3 (public)
    5) Stores results in memory and returns job_id + data
    """
    job_id = str(uuid.uuid4())
    file_path = f"/tmp/{job_id}.mp4"

    try:
        # Save the uploaded file
        print(f"Saving uploaded file to {file_path}...")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print("File saved locally.")

        # Prosody analysis
        print(f"Analyzing prosody for {file_path}...")
        try:
            prosody_data = prosody_processor.analyze_prosody(file_path)
            print(f"Prosody analysis complete. {len(prosody_data)} frames.")
        except Exception as e:
            print(f"Prosody analysis failed: {e}")
            prosody_data = {"error": str(e)}

        # Upload to S3
        s3_url = upload_to_public_storage(file_path, job_id)

        # Remove local file
        try:
            os.remove(file_path)
            print(f"Removed local file: {file_path}")
        except FileNotFoundError:
            pass

        # Store results
        JOB_DB[job_id] = {
            "status": "complete",          # everything is synchronous now
            "prosody_data": prosody_data,
            "s3_url": s3_url
        }

        # Return result immediately
        return JSONResponse(
            status_code=200,
            content={
                "status": "complete",
                "job_id": job_id,
                "s3_url": s3_url,
                "prosody_data": prosody_data
            },
        )

    except Exception as e:
        # Clean up local file if it still exists
        if os.path.exists(file_path):
            os.remove(file_path)
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})


@app.get("/status/{job_id}")
def get_job_status(job_id: str):
    """
    With Modal removed, jobs are synchronous. This endpoint just returns
    whatever we stored for the given job_id (useful for clients that poll).
    """
    job = JOB_DB.get(job_id)
    if not job:
        return JSONResponse(
            status_code=404,
            content={"status": "error", "job_id": job_id, "detail": "Job ID not found."}
        )
    return JSONResponse(status_code=200, content=job)


# --- Run the Server ---
if __name__ == "__main__":
    print("Starting FastAPI server on http://127.0.0.1:8000")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)