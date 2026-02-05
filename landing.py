import streamlit as st

def run():
    # title
    st.write("# Emotion Detection")

    # description
    st.write("""
    This application utilizes an object detection model to detect and classify human emotions from images and videos.
    Emotions detected include angry, happy, relaxed, rock (sunglasses), romantic, sad, and surprised.
    The model is built using the YOLOv8 architecture and trained on a custom dataset.
    """)

    st.write('## How to Use')
    st.write('''There are 3 modes:
        
- Infer from image
- Infer from Video
- Live camera feed
        
Go to navigation sidebar to chose mode.''')

    st.write("## Result Example")

    col1, col2 = st.columns(2, border=True)

    with col1:
        # make the title centered
        st.markdown("<h2 style='text-align: center;'>Input</h2>", unsafe_allow_html=True)
        st.video("sample_video.mp4", format="video/mp4",
                muted=True, autoplay=True)

    with col2:
        st.markdown("<h2 style='text-align: center;'>Output</h2>", unsafe_allow_html=True)
        st.video("output_video_annotated.mp4", format="video/mp4",
                muted=True, autoplay=True)
        
if __name__ == "__main__":
    run()