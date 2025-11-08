import os
import uuid
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv
import boto3

# Optional deps
try:
    import modal  # optional; may not exist in Railway image
except Exception:
    modal = None

# Your local modules
from STT import transcribe_video
from prosody_processor import analyze_prosody
from enricher import enrich_transcript

# ----------------------------------
# Environment & AWS
# ----------------------------------
load_dotenv()

S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

def get_s3_client():
    if not (S3_BUCKET_NAME and AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY):
        raise RuntimeError("Missing one or more S3 env vars (S3_BUCKET_NAME, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)")
    return boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION,
    )

# ----------------------------------
# App & CORS
# ----------------------------------
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
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)

# ----------------------------------
# Health
# ----------------------------------
@app.get("/healthz")
def healthz():
    return {"ok": True, "allowed_origins": ALLOWED_ORIGINS}

@app.options("/api/upload-video")
async def options_upload_video():
    return JSONResponse(status_code=204, content=None)

# ----------------------------------
# In-memory job state
# ----------------------------------
JOB_STATUS_DB = {}

# ----------------------------------
# Vision / Modal (optional)
# ----------------------------------
VisionAgent = None

@app.on_event("startup")
def _maybe_init_modal():
    global VisionAgent
    if not modal:
        print("[startup] modal not installed; vision will be disabled.")
        return
    try:
        # Lazily resolve the class only if modal exists in this environment
        VisionProcessor = modal.Cls.from_name("Tremolo-Vision", "VisionProcessor")
        VisionAgent = VisionProcessor()
        print("[startup] Modal VisionAgent initialized.")
    except Exception as e:
        print(f"[startup] Modal unavailable: {e}. Vision will be disabled.")
        VisionAgent = None

# ----------------------------------
# Helpers
# ----------------------------------
def upload_to_public_storage(file_path: str, job_id: str) -> str:
    """
    Uploads a file to S3 and makes it public-read.
    Returns the public URL for the uploaded video.
    """
    s3_client = get_s3_client()
    s3_key = f"uploads/{job_id}.mp4"

    print(f"Uploading {file_path} to s3://{S3_BUCKET_NAME}/{s3_key} ...")
    s3_client.upload_file(
        file_path,
        S3_BUCKET_NAME,
        s3_key,
        ExtraArgs={"ACL": "public-read", "ContentType": "video/mp4"},
    )

    region = s3_client.meta.region_name or "us-east-1"
    # some buckets in us-east-1 omit the region in URL; this generic form works with explicit region:
    return f"https://{S3_BUCKET_NAME}.s3.{region}.amazonaws.com/{s3_key}"

# ----------------------------------
# Endpoints
# ----------------------------------
@app.post("/api/upload-video")
async def upload_video(file: UploadFile = File(...)):
    """
    Upload a video to S3.
    If Modal vision is available, spawn a vision job.
    In parallel, run STT and Prosody locally.
    """
    if file is None:
        raise HTTPException(status_code=400, detail="Missing file. Send as multipart/form-data with field name 'file'.")
    if file.content_type and not file.content_type.startswith(("video/", "application/octet-stream")):
        raise HTTPException(status_code=415, detail=f"Unsupported content type: {file.content_type}")

    os.makedirs("/tmp", exist_ok=True)
    job_id = str(uuid.uuid4())
    tmp_path = os.path.join("/tmp", f"{job_id}.mp4")

    try:
        # Save upload to temp file
        with open(tmp_path, "wb") as out:
            shutil.copyfileobj(file.file, out)

        # Upload to S3 first
        public_video_url = upload_to_public_storage(tmp_path, job_id)

        # If Modal is present, spawn vision; otherwise skip gracefully
        modal_call_id = None
        if VisionAgent:
            try:
                call = VisionAgent.analyze.spawn(public_video_url)
                modal_call_id = getattr(call, "object_id", None)
                print(f"[{job_id}] spawned Modal vision job: {modal_call_id}")
            except Exception as e:
                print(f"[{job_id}] Modal spawn failed (continuing without vision): {e}")

        # Run STT + Prosody in parallel
        transcription_data = None
        prosody_data = None
        with ThreadPoolExecutor(max_workers=2) as pool:
            stt_future = pool.submit(transcribe_video, public_video_url)
            prosody_future = pool.submit(analyze_prosody, tmp_path)
            for fut in as_completed([stt_future, prosody_future]):
                try:
                    res = fut.result()
                    if fut is stt_future:
                        transcription_data = res
                        print(f"[{job_id}] STT done")
                    else:
                        prosody_data = res
                        print(f"[{job_id}] Prosody done")
                except Exception as sub_e:
                    print(f"[{job_id}] subtask failed: {sub_e}")

        JOB_STATUS_DB[job_id] = {
            "status": "processing" if modal_call_id else "enriching",
            "job_id": job_id,
            "modal_id": modal_call_id,
            "transcription": transcription_data,
            "prosody": prosody_data,
            "vision": None,
            "enriched_transcript": None,
        }

        # If no modal vision, we can enrich immediately with what we have
        if not modal_call_id:
            try:
                if transcription_data and transcription_data.get("status") == "completed":
                    JOB_STATUS_DB[job_id]["enriched_transcript"] = enrich_transcript(
                        transcription_data, None, prosody_data
                    )
                else:
                    JOB_STATUS_DB[job_id]["enriched_transcript"] = {
                        "error": "Transcription missing or incomplete; cannot enrich."
                    }
                JOB_STATUS_DB[job_id]["status"] = "complete"
            except Exception as e:
                print(f"[{job_id}] enrichment failed: {e}")

        return JSONResponse(status_code=202, content={"status": JOB_STATUS_DB[job_id]["status"], "job_id": job_id})

    except Exception as e:
        print(f"[{job_id}] upload failed: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "job_id": None, "detail": str(e)})

    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception as cleanup_err:
            print(f"[{job_id}] temp cleanup error: {cleanup_err}")

@app.get("/api/status/{job_id}")
async def get_job_status(job_id: str):
    job = JOB_STATUS_DB.get(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"status": "error", "detail": "Job not found."})

    # If already complete or no modal job, return as-is (with raw bits scrubbed)
    if job["status"] == "complete" or not job.get("modal_id"):
        safe = {k: v for k, v in job.items() if k not in ("prosody", "vision", "modal_id")}
        return JSONResponse(status_code=200, content=safe)

    # Poll Modal non-blocking
    try:
        if not modal:
            return JSONResponse(status_code=200, content={"status": job["status"], "job_id": job_id})

        modal_call = modal.FunctionCall.from_id(job["modal_id"])
        try:
            vision_data = modal_call.get(timeout=0)  # immediate return if ready

            job["vision"] = vision_data
            # Enrich if we have transcription
            if job.get("transcription") and job["transcription"].get("status") == "completed":
                job["enriched_transcript"] = enrich_transcript(
                    job["transcription"], job["vision"], job.get("prosody")
                )
            else:
                job["enriched_transcript"] = {"error": "Transcription failed or incomplete; cannot enrich."}

            job["status"] = "complete"
            safe = {k: v for k, v in job.items() if k not in ("prosody", "vision", "modal_id")}
            return JSONResponse(status_code=200, content=safe)

        except TimeoutError:
            return JSONResponse(status_code=200, content={"status": "processing", "job_id": job_id})

    except Exception as e:
        print(f"[{job_id}] status check failed: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})

# ----------------------------------
# Entrypoint
# ----------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, proxy_headers=True, forwarded_allow_ips="*")