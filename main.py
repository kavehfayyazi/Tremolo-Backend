import os
import uuid
import json
import shutil
import boto3
import uvicorn
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ==== Your app-specific imports ====
from STT import transcribe_video
from prosody_processor import analyze_prosody
from enricher import enrich_transcript

# ==== Optional Modal (best-effort) ====
try:
    import modal
    _HAS_MODAL = True
except Exception:
    _HAS_MODAL = False

load_dotenv()

# ---- AWS S3 client (optional) ----
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")
_AWS_REGION = os.environ.get("AWS_REGION", "us-east-2")
_AWS_KEY = os.environ.get("AWS_ACCESS_KEY_ID")
_AWS_SECRET = os.environ.get("AWS_SECRET_ACCESS_KEY")

s3_client = None
if S3_BUCKET_NAME and _AWS_KEY and _AWS_SECRET:
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=_AWS_KEY,
        aws_secret_access_key=_AWS_SECRET,
        region_name=_AWS_REGION
    )

def upload_to_public_storage(file_path: str, job_id: str) -> str | None:
    """
    Uploads a file to S3 with public-read ACL, returns a public URL.
    Returns None if S3 is not configured.
    """
    if not s3_client or not S3_BUCKET_NAME:
        return None

    s3_key = f"uploads/{job_id}.mp4"
    try:
        s3_client.upload_file(
            file_path,
            S3_BUCKET_NAME,
            s3_key,
            ExtraArgs={"ACL": "public-read", "ContentType": "video/mp4"}
        )
        # Build region-aware URL
        region = s3_client.meta.region_name or "us-east-1"
        if region == "us-east-1":
            return f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{s3_key}"
        return f"https://{S3_BUCKET_NAME}.s3.{region}.amazonaws.com/{s3_key}"
    except Exception as e:
        print(f"S3 upload failed: {e}")
        return None

# ---- Optional Modal function wiring ----
VisionAgent = None
if _HAS_MODAL:
    try:
        VisionProcessor = modal.Cls.from_name("Tremolo-Vision", "VisionProcessor")
        VisionAgent = VisionProcessor()
    except Exception as e:
        print(f"Modal not available: {e}")
        VisionAgent = None

# ---- In-memory job store ----
# job = {
#   "status": "processing"|"complete"|"error",
#   "job_id": str,
#   "modal_id": Optional[str],
#   "transcription": {...}|None,
#   "prosody": {...}|None,
#   "vision": {...}|None,
#   "enriched_transcript": {...}|None
# }
JOB_STATUS_DB: dict[str, dict] = {}

app = FastAPI()

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        os.environ.get("FRONTEND_ORIGIN", ""),  # optional prod origin
        "*",  # relax for hackathon/demo
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
#        ENDPOINTS
# =========================

@app.post("/api/upload-video")
async def upload_video(file: UploadFile = File(...)):
    """
    Accepts multipart/form-data with a key named 'file' containing the MP4.
    Uploads to S3 (if configured), kicks off local STT+prosody in parallel.
    If Modal Vision is available, spawns a remote vision job; otherwise skip.
    Returns 202 with job_id while processing; polling via /api/status/{job_id}.
    """
    job_id = str(uuid.uuid4())
    temp_dir = "/tmp"
    os.makedirs(temp_dir, exist_ok=True)
    video_path = os.path.join(temp_dir, f"{job_id}.mp4")

    try:
        # Save upload
        with open(video_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Optional S3
        public_video_url = upload_to_public_storage(video_path, job_id)

        # Optional Modal Vision spawn
        modal_call_id = None
        if VisionAgent and public_video_url:
            try:
                call = VisionAgent.analyze.spawn(public_video_url)
                modal_call_id = getattr(call, "object_id", None)
            except Exception as e:
                print(f"Modal spawn failed (continuing without vision): {e}")

        # Run STT + Prosody in parallel (local)
        transcription_data = None
        prosody_data = None
        with ThreadPoolExecutor(max_workers=2) as executor:
            stt_future = executor.submit(
                transcribe_video,
                public_video_url if public_video_url else video_path
            )
            prosody_future = executor.submit(analyze_prosody, video_path)

            for fut in as_completed([stt_future, prosody_future]):
                try:
                    if fut is stt_future:
                        transcription_data = fut.result()
                    else:
                        prosody_data = fut.result()
                except Exception as sub_e:
                    print(f"Subtask error: {sub_e}")

        # If no vision (Modal off or still running), we can enrich now using what we have.
        # Otherwise enrichment will happen in the status poll once vision completes.
        vision_data = None
        enriched = None
        status = "processing"

        if modal_call_id is None:
            # No Modal path → enrich immediately with what we have
            try:
                if transcription_data and transcription_data.get("status") == "completed":
                    enriched = enrich_transcript(transcription_data, vision_data, prosody_data)
                    status = "complete"
                else:
                    enriched = {"error": "Transcription failed, cannot enrich."}
                    status = "complete"
            except Exception as e:
                print(f"Enrichment failed: {e}")
                enriched = {"error": f"Enrichment failed: {e}"}
                status = "complete"

        JOB_STATUS_DB[job_id] = {
            "status": status,
            "job_id": job_id,
            "modal_id": modal_call_id,
            "transcription": transcription_data,
            "prosody": prosody_data,
            "vision": vision_data,
            "enriched_transcript": enriched
        }

        # If status is already complete (no Modal), return full result now
        if status == "complete":
            # cleanup raw fields we don’t want to expose if you prefer minimal output:
            job = JOB_STATUS_DB[job_id]
            # keep everything for now so frontend can inspect; trim if desired
            return JSONResponse(status_code=200, content=job)

        # Otherwise it’s processing (waiting for Modal), tell client to poll
        return JSONResponse(status_code=202, content={"status": "processing", "job_id": job_id})

    except Exception as e:
        print(f"Upload failed: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})
    finally:
        # Best-effort local cleanup; comment out during debugging if desired
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
        except Exception:
            pass


@app.get("/api/status/{job_id}")
async def get_job_status(job_id: str):
    """
    Poll job status. If Modal vision is running, try to collect it.
    When all available data is ready, enrich and mark complete.
    """
    job = JOB_STATUS_DB.get(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"status": "error", "detail": "Job not found."})

    # Already done? return as-is
    if job["status"] == "complete":
        return JSONResponse(status_code=200, content=job)

    # If we have a pending Modal call, try to fetch it non-blocking.
    modal_id = job.get("modal_id")
    if _HAS_MODAL and modal_id:
        try:
            modal_call = modal.FunctionCall.from_id(modal_id)
            try:
                # timeout=0 → immediate poll; raises on not ready
                vision_data = modal_call.get(timeout=0)
                job["vision"] = vision_data
                # Attempt enrichment now that vision is in
                try:
                    if job["transcription"] and job["transcription"].get("status") == "completed":
                        job["enriched_transcript"] = enrich_transcript(
                            job["transcription"], job["vision"], job["prosody"]
                        )
                    else:
                        job["enriched_transcript"] = {"error": "Transcription failed, cannot enrich."}
                    job["status"] = "complete"
                    # Trim internals if you want a clean payload
                    job.pop("modal_id", None)
                    return JSONResponse(status_code=200, content=job)
                except Exception as e_enrich:
                    job["enriched_transcript"] = {"error": f"Enrichment failed: {e_enrich}"}
                    job["status"] = "complete"
                    job.pop("modal_id", None)
                    return JSONResponse(status_code=200, content=job)

            except Exception:
                # Not ready yet (Modal timeout/empty); keep processing
                return JSONResponse(status_code=200, content={"status": "processing", "job_id": job_id})
        except Exception as e:
            # If Modal lookup itself failed, degrade gracefully and finalize without vision
            print(f"Modal status check failed: {e}")
            try:
                if job["transcription"] and job["transcription"].get("status") == "completed":
                    job["enriched_transcript"] = enrich_transcript(
                        job["transcription"], None, job["prosody"]
                    )
                else:
                    job["enriched_transcript"] = {"error": "Transcription failed, cannot enrich."}
            except Exception as e2:
                job["enriched_transcript"] = {"error": f"Enrichment failed: {e2}"}
            job["status"] = "complete"
            job.pop("modal_id", None)
            return JSONResponse(status_code=200, content=job)

    # No Modal in play → if we somehow got here still “processing”, finalize from what we have
    try:
        if job["transcription"] and job["transcription"].get("status") == "completed":
            job["enriched_transcript"] = enrich_transcript(
                job["transcription"], job.get("vision"), job.get("prosody")
            )
        else:
            job["enriched_transcript"] = {"error": "Transcription failed, cannot enrich."}
    except Exception as e:
        job["enriched_transcript"] = {"error": f"Enrichment failed: {e}"}

    job["status"] = "complete"
    job.pop("modal_id", None)
    return JSONResponse(status_code=200, content=job)


if __name__ == "__main__":
    # For local dev; in prod use your ASGI server (e.g., uvicorn/gunicorn)
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))