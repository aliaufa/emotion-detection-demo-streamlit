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

    with st.form("image_infer_form"):
        st.write("## Upload an Image for Emotion Detection")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        submit_button = st.form_submit_button(label='Detect Image')

    if submit_button and uploaded_file is not None:
        # Save uploaded image to a temporary location
        temp_dir = Path("temp_images")
        temp_dir.mkdir(exist_ok=True)
        IMAGE_PATH = temp_dir / uploaded_file.name
        with open(IMAGE_PATH, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Run inference
        results = model(IMAGE_PATH, conf=0.25)

        # Display results
        for result in results:
            # Get the annotated image
            annotated_img = result.plot()
            
            # Convert BGR to RGB for matplotlib
            annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            
            # Display
            fig = plt.figure(figsize=(12, 8))
            plt.imshow(annotated_img_rgb)
            plt.axis('off')
            plt.title('Emotion Detection Results')
            st.pyplot(fig)
            
            # Print detection details
            boxes = result.boxes
            print(f"\nDetected {len(boxes)} objects:")
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = result.names[class_id]
                st.write(f"  - {class_name}: {confidence:.2%}")

if __name__ == "__main__":
    run()