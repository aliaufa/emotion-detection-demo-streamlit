import streamlit as st
import landing, image_infer, video_infer, camera_infer

with st.sidebar:
    st.title("Emotion Detection App")
    app_mode = st.radio("Choose the app mode",
                            ["Home", "Image Inference", "Video Inference", "Camera Inference"])
    
    st.write("---")
    st.write("# About")
    st.write("""
    This is a demo application for Emotion Detection using YOLOv26.
             
## Dataset:

Custom dataset with 7 emotion classes: angry, happy, relaxed, rock (sunglasses), romantic, sad, surprised.

url: https://universe.roboflow.com/spacex-bmpib/emotion-gyup3/dataset/8
""")

if app_mode == "Home":
    landing.run()

elif app_mode == "Image Inference":
    image_infer.run()

elif app_mode == "Video Inference":
    video_infer.run()

elif app_mode == "Camera Inference":
    camera_infer.run()