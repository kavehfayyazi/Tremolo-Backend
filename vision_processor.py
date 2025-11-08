import modal
import cv2
import httpx  # To download the video
import os   # To remove temp file

# 1. Setup modal environment
modal_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "opencv-python-headless==4.8.0.76",
        "mediapipe",
        "httpx",  # To fetch the video from the S3/public URL
    )
    .apt_install(
        "libgl1-mesa-glx",  # System dependencies for OpenCV
        "libglib2.0-0",
    )
    .add_local_file("pose_landmarker_full.task", "/root/pose_landmarker_full.task")
)

# 2. Create the App
# We apply the image and name the app here.
app = modal.App(name="Tremolo-Vision", image=modal_image)


# --- The Modal "Class" ---
# This class will be loaded onto a worker.
@app.cls()  # Using CPU processing
class VisionProcessor:
    @modal.enter()
    def setup(self):
        """
        This runs once when the container starts.
        We create the landmarker here so it's "warm" and ready.
        """
        import mediapipe as mp
        
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path='/root/pose_landmarker_full.task',
                delegate=mp.tasks.BaseOptions.Delegate.CPU
            ),
            running_mode=VisionRunningMode.VIDEO)
        
        print("Warming up container, creating PoseLandmarker...")
        self.landmarker = PoseLandmarker.create_from_options(options)
        print("Landmarker created.")

    @modal.method()
    def analyze(self, video_url: str) -> list:
        """
        This is the main function our FastAPI server will call.
        It takes a URL, downloads the video, processes it,
        and returns the *entire* JSON log directly.
        """
        print("Entered function")
        # Validate URL format
        if not video_url or not isinstance(video_url, str):
            raise ValueError("Invalid video_url provided")
        if not (video_url.startswith("http://") or video_url.startswith("https://")):
            raise ValueError(f"Video URL must be HTTP/HTTPS: {video_url}")
        
        vision_log = []
        
        # 1. Download the video from the public URL
        # We need to save it to a temporary local file for OpenCV
        temp_video_path = "/tmp/input_video.mp4"
        try:
            with httpx.stream("GET", video_url, timeout=30.0, follow_redirects=True) as response:
                response.raise_for_status()  # Check for download errors
                with open(temp_video_path, "wb") as f:
                    for chunk in response.iter_bytes():
                        f.write(chunk)
            print(f"Video downloaded from {video_url}")
        except Exception as e:
            print(f"Error downloading video: {e}")
            raise

        # 2. Open the downloaded video file
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            raise Exception(f"Error: Could not open video file {temp_video_path}")

        print("Starting frame analysis...")
        # 3. Loop through every frame
        # Get FPS for more accurate timestamp calculation
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # Default to 30 if unavailable
        
        # Configuration: How many frames per second to process
        TARGET_FPS = 3  # Process 3 frames per second
        frame_skip = int(fps / TARGET_FPS)  # Skip N frames to achieve target FPS
        
        frame_number = 0
        processed_count = 0
        
        print(f"Video FPS: {fps}, Processing every {frame_skip} frames (target: {TARGET_FPS} FPS)")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Only process every Nth frame
            if frame_number % frame_skip != 0:
                frame_number += 1
                continue
            
            # Calculate timestamp more reliably
            timestamp_ms = int((frame_number / fps) * 1000)
            frame_number += 1
            processed_count += 1
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Import mediapipe for image processing
            import mediapipe as mp
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            try:
                # 4. Detect pose landmarks
                pose_landmarker_result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
                
                # 5. Extract and store results
                landmarks_list = []
                if pose_landmarker_result.pose_landmarks:
                    for landmark_list in pose_landmarker_result.pose_landmarks:
                        frame_landmarks = []
                        for landmark in landmark_list:
                            frame_landmarks.append({
                                'x': landmark.x, 'y': landmark.y, 'z': landmark.z,
                                'visibility': landmark.visibility
                            })
                        landmarks_list.append(frame_landmarks)

                vision_log.append({
                    "timestamp_sec": timestamp_ms / 1000.0,
                    "frame_number": frame_number - 1,
                    "pose_landmarks": landmarks_list
                })
            except Exception as e:
                print(f"Error processing frame {frame_number} at {timestamp_ms}ms: {e}")
                vision_log.append({
                    "timestamp_sec": timestamp_ms / 1000.0,
                    "frame_number": frame_number - 1,
                    "pose_landmarks": [],
                    "error": str(e)
                })
        
        cap.release()
        print(f"Analysis complete. Processed {processed_count} frames out of {frame_number} total frames.")
        
        # 6. Clean up the temp video file
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
            
        # 7. Return the final data (Modal handles JSON serialization)
        return vision_log


# Use the new decorator to define a local test entrypoint
@app.local_entrypoint()
def main(video_url: str):
    """
    This lets you test the function from your local command line:
    `modal run vision_processor.py --video-url "..."`
    """
    import json
    
    processor = VisionProcessor()
    result = processor.analyze.remote(video_url)
    
    print(f"\n✅ Analysis complete! {len(result)} frames processed.\n")
    
    # Save to a JSON file
    output_file = "vision_output.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"📄 Full results saved to: {output_file}")
    
    # Show a sample of the first frame with landmarks
    if result and len(result) > 0:
        first_frame = result[0]
        print(f"\n📊 Sample - First frame:")
        print(f"  Timestamp: {first_frame.get('timestamp_sec')}s")
        print(f"  Frame number: {first_frame.get('frame_number')}")
        if first_frame.get('pose_landmarks'):
            num_people = len(first_frame['pose_landmarks'])
            num_landmarks = len(first_frame['pose_landmarks'][0]) if num_people > 0 else 0
            print(f"  People detected: {num_people}")
            print(f"  Landmarks per person: {num_landmarks}")
            if num_landmarks > 0:
                print(f"\n  First landmark example:")
                print(f"    {json.dumps(first_frame['pose_landmarks'][0][0], indent=6)}")
        else:
            print("  No poses detected in this frame")
    
    print(f"\n💡 Open '{output_file}' to see all {len(result)} frames of landmark data!")