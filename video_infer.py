import streamlit as st
from ultralytics import YOLO
import cv2
import os
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

def run():
    # load model
    model = YOLO('./model/best.pt')

    def process_video_with_stats(video_path, output_path, conf=0.25, max_frames=500):
        """
        Process video file with emotion detection and collect statistics
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video
            conf: Confidence threshold
            max_frames: Maximum number of frames to process (default: 500)
        """
        import cv2
        from collections import defaultdict
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Limit frames to process
        frames_to_process = min(total_frames, max_frames)
        
        st.write(f"Video properties:")
        st.write(f"  FPS: {fps}")
        st.write(f"  Resolution: {width}x{height}")
        st.write(f"  Total frames: {total_frames}")
        st.write(f"  Processing: {frames_to_process} frames")
        st.write(f"  Duration: {frames_to_process/fps:.2f} seconds")
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Statistics
        emotion_counts = defaultdict(int)
        frame_count = 0
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run inference on frame with optimized settings
            results = model(frame, conf=conf, verbose=False, device='cpu')
            
            # Count emotions
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                class_name = results[0].names[class_id]
                emotion_counts[class_name] += 1
            
            # Get annotated frame
            annotated_frame = results[0].plot()
            
            # Write frame
            out.write(annotated_frame)
            
            frame_count += 1
            
            # Update progress
            progress = frame_count / frames_to_process
            progress_bar.progress(progress)
            status_text.text(f'Processing frame {frame_count}/{frames_to_process}')
        
        # Release resources
        cap.release()
        out.release()
        
        progress_bar.empty()
        status_text.empty()
        
        st.success(f"Video processing complete! Processed {frame_count} frames.")
        st.write(f"\nEmotion detections in video:")
        for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
            st.write(f"  {emotion:15} : {count:5} detections")
        
        return emotion_counts

    with st.form("video_infer_form"):
        st.write("## Upload a Video for Emotion Detection")
        uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
        submit_button = st.form_submit_button(label='Detect Video')

    if submit_button and uploaded_file is not None:
        # Save uploaded video to a temporary location
        temp_dir = Path("temp_videos")
        temp_dir.mkdir(exist_ok=True)
        VIDEO_PATH = temp_dir / uploaded_file.name
        with open(VIDEO_PATH, "wb") as f:
            f.write(uploaded_file.getbuffer())

        OUTPUT_PATH = temp_dir / f"output_{uploaded_file.name}"

        # Run video processing with stats (limited to 500 frames)
        emotion_stats = process_video_with_stats(str(VIDEO_PATH), str(OUTPUT_PATH), conf=0.25, max_frames=500)

        # Display output video
        st.write("### Output Video with Emotion Annotations")
        st.video(str(OUTPUT_PATH), format="video/mp4")

        # Download button for processed video
        with open(OUTPUT_PATH, 'rb') as f:
            st.download_button(
                label="Download Result Video",
                data=f,
                file_name=f"emotion_detected_{uploaded_file.name}",
                mime="video/mp4"
            )

if __name__ == "__main__":
    run()