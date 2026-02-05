import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from collections import defaultdict
import time

def run():
    # Load model
    model = YOLO('./model/best.pt')

    st.write("## Webcam Emotion Detection")
    st.write("Capture frames from your webcam and detect emotions in real-time.")

    # Option 1: Simple camera input (single snapshot)
    st.write("### Option 1: Capture Single Frame")
    camera_photo = None
    photo_button = st.button("Open Camera")
    if photo_button:
        camera_photo = st.camera_input("Take a picture")

    if camera_photo is not None:
        # Convert the uploaded image to OpenCV format
        image = Image.open(camera_photo)
        img_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Run inference
        with st.spinner('Detecting emotions...'):
            results = model(img_bgr, conf=0.25, verbose=False)
            
            # Get annotated image
            annotated_img = results[0].plot()
            
            # Convert back to RGB for display
            annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Original Image**")
            st.image(img_array, use_container_width=True)
        
        with col2:
            st.write("**Detected Emotions**")
            st.image(annotated_img_rgb, use_container_width=True)
        
        # Display detection details
        st.write("### Detection Results:")
        if len(results[0].boxes) > 0:
            for i, box in enumerate(results[0].boxes):
                class_id = int(box.cls[0])
                class_name = results[0].names[class_id]
                confidence = float(box.conf[0])
                st.write(f"**Detection {i+1}:** {class_name} (confidence: {confidence:.2f})")
        else:
            st.write("No emotions detected in the image.")

    # Option 2: Live webcam feed with continuous inference
    st.write("---")
    st.write("### Option 2: Live Webcam Feed")
    st.write("Process live webcam feed with emotion detection.")

    # Session state for webcam control
    if 'webcam_running' not in st.session_state:
        st.session_state.webcam_running = False

    col1, col2 = st.columns(2)
    with col1:
        start_button = st.button("Start Webcam", disabled=st.session_state.webcam_running)
    with col2:
        stop_button = st.button("Stop Webcam", disabled=not st.session_state.webcam_running)

    if start_button:
        st.session_state.webcam_running = True
        st.rerun()

    if stop_button:
        st.session_state.webcam_running = False
        st.rerun()

    if st.session_state.webcam_running:
        # Placeholders for live feed
        frame_placeholder = st.empty()
        stats_placeholder = st.empty()
        stop_placeholder = st.empty()
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Error: Could not open webcam. Please check your camera permissions.")
            st.session_state.webcam_running = False
        else:
            # Set camera properties for better performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            emotion_counts = defaultdict(int)
            frame_count = 0
            max_frames = 300  # Limit to 300 frames (about 10 seconds at 30fps)
            
            # Process frames
            while st.session_state.webcam_running and frame_count < max_frames:
                ret, frame = cap.read()
                
                if not ret:
                    st.error("Error reading from webcam.")
                    break
                
                # Run inference
                results = model(frame, conf=0.25, verbose=False)
                
                # Count emotions
                for box in results[0].boxes:
                    class_id = int(box.cls[0])
                    class_name = results[0].names[class_id]
                    emotion_counts[class_name] += 1
                
                # Get annotated frame
                annotated_frame = results[0].plot()
                
                # Convert BGR to RGB for Streamlit
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Display frame
                frame_placeholder.image(annotated_frame_rgb, channels="RGB", use_container_width=True)
                
                # Display stats
                frame_count += 1
                stats_text = f"**Frame:** {frame_count}/{max_frames}\n\n**Emotion Detections:**\n"
                if emotion_counts:
                    for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
                        stats_text += f"- {emotion}: {count}\n"
                else:
                    stats_text += "No emotions detected yet."
                
                stats_placeholder.markdown(stats_text)
                
                # Add small delay to control frame rate
                time.sleep(0.033)  # ~30 fps
            
            # Release webcam
            cap.release()
            st.session_state.webcam_running = False
            
            if frame_count >= max_frames:
                st.info(f"Reached maximum frame limit ({max_frames} frames). Click 'Start Webcam' to continue.")

    st.write("---")
    st.write("**Note:** Live webcam feed will automatically stop after 300 frames (~10 seconds). Click 'Start Webcam' again to continue.")

if __name__ == "__main__":
    run()