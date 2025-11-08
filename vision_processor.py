import modal
import cv2
import httpx
import os
import numpy as np

# 1. Setup modal environment
modal_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "opencv-python-headless==4.8.0.76",
        "mediapipe==0.10.9",
        "numpy==1.26.4",
        "httpx"
    )
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .add_local_file("pose_landmarker_full.task", "/root/pose_landmarker_full.task")
)

app = modal.App(name="Tremolo-Vision", image=modal_image)

@app.cls(cpu=2.0, memory=2048)
class VisionProcessor:
    @modal.enter()
    def setup(self):
        import mediapipe as mp
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='/root/pose_landmarker_full.task'),
            running_mode=VisionRunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = PoseLandmarker.create_from_options(options)
        
        # 7: Left Ear, 8: Right Ear
        # 11: Left Shoulder, 12: Right Shoulder
        # 13: Left Elbow, 14: Right Elbow
        # 15: Left Wrist, 16: Right Wrist
        self.REQUIRED_LANDMARKS = [7, 8, 11, 12, 13, 14, 15, 16]

    @modal.method()
    def analyze(self, video_url: str):
        import mediapipe as mp
        
        print(f"Starting analysis for: {video_url}")
        video_path = "/tmp/video_to_analyze.mp4"
        
        try:
            with httpx.Client(follow_redirects=True, timeout=60.0) as client:
                response = client.get(video_url)
                response.raise_for_status()
                with open(video_path, "wb") as f:
                    f.write(response.content)
            print("Video downloaded successfully.")
        except Exception as e:
            return {"error": f"Failed to download video: {e}"}

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
             return {"error": "Could not open video file."}

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        # Target ~10 FPS
        TARGET_FPS = 10.0
        frame_interval = max(1, int(round(fps / TARGET_FPS)))
        print(f"Video FPS: {fps}. Processing every {frame_interval} frames.")
        
        vision_log = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                try:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    
                    # KEY FIX: Use frame_count directly for monotonic increasing timestamp
                    timestamp_ms = int((frame_count / fps) * 1000)
                    
                    detection_result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

                    frame_data = {
                        "timestamp": round(frame_count / fps, 3),
                        "poses": []
                    }

                    if detection_result.pose_landmarks:
                        for pose in detection_result.pose_landmarks:
                            filtered_pose = []
                            for idx in self.REQUIRED_LANDMARKS:
                                landmark = pose[idx]
                                filtered_pose.append({
                                    "id": idx,
                                    "x": round(landmark.x, 4),
                                    "y": round(landmark.y, 4),
                                    "z": round(landmark.z, 4),
                                    "visibility": round(landmark.visibility, 2)
                                })
                            frame_data["poses"].append(filtered_pose)
                    
                    vision_log.append(frame_data)
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {e}")
                    # Continue to next frame instead of crashing

            frame_count += 1

        cap.release()
        if os.path.exists(video_path):
            os.remove(video_path)

        print(f"Analysis complete. Processed {len(vision_log)} frames.")
        return vision_log

# Local entrypoint for testing
@app.local_entrypoint()
def main(video_url: str):
    import json
    processor = VisionProcessor()
    result = processor.analyze.remote(video_url)
    print(f"\n✅ Analysis complete! {len(result)} frames processed.")
    with open("vision_output.json", "w") as f:
        json.dump(result, f, indent=2)
    print("📄 Results saved to vision_output.json")