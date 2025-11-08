import uvicorn
import shutil
import uuid
import os
import json
import modal # Import Modal
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import boto3

s3_client = boto3.client(
    's3',
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY")
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
        
        # Get the public URL
        # URL format is: https://[bucket-name].s3.[region].amazonaws.com/[key]
        # You may need to update this if your region isn't us-east-1
        region = s3_client.meta.region_name
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
    VisionAgent = modal.Function.lookup(
        "Tremolo-Vision", # name
        "VisionProcessor.analyze"      # The class and method to call
    )
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
            
        # 2. Upload the file to S3
        public_video_url = upload_to_public_storage(video_path, job_id)
        
        # 3. Clean up the local temp file
        os.remove(video_path)

        # 4. Start the Modal job ASYNCHRONOUSLY
        print(f"Spawning Modal job for {job_id} with URL: {public_video_url}")
        call = VisionAgent.spawn(public_video_url)
        modal_call_id = call.call_id

        # 5. Create the job in our mock DB (mapping our ID to Modal's ID)
        JOB_STATUS_DB[job_id] = {
            "status": "processing", 
            "modal_id": modal_call_id, 
            "data": None
        }

        # 6. Return IMMEDIATELY to the client
        return JSONResponse(
            status_code=202,
            content={"status": "processing", "job_id": job_id}
        )
        
    except Exception as e:
        print(f"Upload failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "job_id": None, "detail": str(e)}
        )

@app.get("/api/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """
    This endpoint now actively polls Modal for the job status.
    """
    
    # 1. Check our DB for the job
    job = JOB_STATUS_DB.get(job_id)

    if not job:
        return JSONResponse(
            status_code=404,
            content={"status": "error", "job_id": job_id, "detail": "Job not found."}
        )
    
    # 2. If it's already complete, just return the data
    if job["status"] == "complete":
        return JSONResponse(status_code=200, content=job)

    # 3. If still processing, let's check Modal
    try:
        modal_call_id = job["modal_id"]
        modal_call = modal.FunctionCall.get(modal_call_id)
        
        if modal_call.is_running():
            print(f"Job {job_id} is still running on Modal.")
            return JSONResponse(status_code=200, content={"status": "processing", "job_id": job_id})
        
        if modal_call.is_completed():
            print(f"Job {job_id} just completed!")
            vision_data = modal_call.result()
            
            # Save the result to our DB and return it
            job["status"] = "complete"
            job["data"] = vision_data
            return JSONResponse(status_code=200, content=job)

    except modal.exception.NotFoundError:
         return JSONResponse(status_code=404, content={"status": "error", "job_id": job_id, "detail": "Modal call ID not found."})
    except Exception as e:
        # This can happen if the Modal job failed
        print(f"Modal job {job_id} failed: {e}")
        try:
            # Try to get the specific exception from Modal
            modal_call = modal.FunctionCall.get(job["modal_id"])
            error_detail = str(modal_call.exception())
        except:
            error_detail = str(e)
            
        job["status"] = "error"
        job["data"] = {"detail": error_detail}
        return JSONResponse(status_code=200, content=job) # Return 200 so client can parse it

    # Default case (still processing)
    return JSONResponse(status_code=200, content={"status": "processing", "job_id": job_id})

# --- Run the Server ---
if __name__ == "__main__":
    print("Starting FastAPI server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)