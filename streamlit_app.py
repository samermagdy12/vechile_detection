import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile
import os
import numpy as np

# --- CONFIGURATION ---
st.set_page_config(
    page_title="YOLOv8 Traffic Detection & Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "A professional object detection application built with Streamlit and YOLOv8."
    }
)

# --- HEADER AND INTRODUCTION (Modern UI) ---
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    .header-title {
        font-size: 2.5em;
        font-weight: 700;
        color: #1f77b4; /* Streamlit primary color */
        margin-bottom: 0.2em;
    }
    .header-subtitle {
        font-size: 1.1em;
        color: #555;
        margin-bottom: 1.5em;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<p class="header-title">Intelligent Traffic Analysis with YOLOv8</p>', unsafe_allow_html=True)
st.markdown(
    """
    <p class="header-subtitle">
    Upload an **image or video** to perform real-time object detection on vehicles, pedestrians, and traffic signs. 
    The system leverages a pre-trained YOLOv8 model for high-accuracy analysis.
    </p>
    """,
    unsafe_allow_html=True
)

# --- MODEL LOADING ---
# NOTE: The original model path 'D:\deploy\best (11).pt' is a Windows path and will not work here.
# I will use a standard pre-trained model for demonstration.
# The user must replace 'yolov8n.pt' with their actual model path if they want to use their custom model.
@st.cache_resource
def load_model():
    # Using a standard YOLOv8 nano model for demonstration.
    # User should replace 'yolov8n.pt' with their actual model path.
    try:
        model = YOLO(r'D:\deploy\best (11).pt')
        return model
    except Exception as e:
        st.error(f"Error loading model. Please ensure 'yolov8n.pt' or your custom model is accessible. Details: {e}")
        return None

model = load_model()

if model is None:
    st.stop()

# --- UPLOAD AND PROCESSING ---
uploaded_file = st.file_uploader(
    "Upload Media File (Image or Video)",
    type=["jpg", "jpeg", "png", "mp4", "avi"],
    help="Supported formats: JPG, PNG, MP4, AVI"
)

if uploaded_file:
    # Use st.spinner for better UX during processing
    with st.spinner(f"Processing {uploaded_file.name}..."):
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False) as tfile:
            tfile.write(uploaded_file.read())
            tfile_path = tfile.name

        file_ext = uploaded_file.name.split('.')[-1].lower()

        if file_ext in ["jpg", "jpeg", "png"]:
            # --- IMAGE PROCESSING ---
            col1, col2 = st.columns(2)

            # Display Original Image
            with col1:
                st.subheader("Original Image")
                img = Image.open(tfile_path)
                st.image(img, use_container_width=True)

            # Predict and display Annotated Image
            with col2:
                st.subheader("Annotated Image")
                
                # Predict
                results = model.predict(img)
                
                # Get annotated image (numpy array in BGR format by default from plot())
                annotated_img_bgr = results[0].plot()
                
                # FIX: Convert BGR to RGB for correct display in Streamlit
                annotated_img_rgb = cv2.cvtColor(annotated_img_bgr, cv2.COLOR_BGR2RGB)
                
                # Display the corrected image
                st.image(annotated_img_rgb, use_container_width=True)

                # Save the annotated image for download (saving as RGB is fine, but we'll use PIL for consistency)
                annotated_img_pil = Image.fromarray(annotated_img_rgb)
                out_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
                annotated_img_pil.save(out_file)

                # Download Button
                with open(out_file, "rb") as f:
                    st.download_button(
                        "Download Annotated Image",
                        data=f.read(),
                        file_name="annotated_image.png",
                        mime="image/png",
                        key="download_image"
                    )
            
            # Clean up temporary file
            os.unlink(tfile_path)
            os.unlink(out_file)


        elif file_ext in ["mp4", "avi"]:
            # --- VIDEO PROCESSING ---
            st.subheader("Video Analysis")
            
            # Display Original Video in a column
            col_orig, col_proc = st.columns(2)
            with col_orig:
                st.markdown("#### Original Video")
                st.video(tfile_path)

            # Process Video
            out_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

            cap = cv2.VideoCapture(tfile_path)
            
            # Check if video opened successfully
            if not cap.isOpened():
                st.error("Error: Could not open video file.")
                os.unlink(tfile_path)
                st.stop()

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Ensure width and height are valid
            if width == 0 or height == 0:
                st.error("Error: Video dimensions are invalid.")
                cap.release()
                os.unlink(tfile_path)
                st.stop()

            out = cv2.VideoWriter(out_file, fourcc, fps, (width, height))

            # Use a progress bar for better UX
            progress_bar = st.progress(0)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            current_frame = 0
            
            st.info("Processing video frames... This may take a moment.")

            # Placeholder for live display of processed video frame
            with col_proc:
                st.markdown("#### Live Detection Feed")
                stframe = st.empty()  

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # YOLO prediction
                results = model.predict(frame, verbose=False) # Suppress verbose output
                annotated_frame = results[0].plot() # BGR numpy array

                out.write(annotated_frame)
                
                # FIX: Convert BGR to RGB for correct display in Streamlit
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                stframe.image(annotated_frame_rgb, channels="RGB", use_container_width=True)
                
                current_frame += 1
                progress_bar.progress(min(current_frame / frame_count, 1.0))

            cap.release()
            out.release()
            progress_bar.empty() # Remove progress bar after completion
            st.success("Video analysis complete!")

            # Display Annotated Video and Download Button
            st.markdown("#### Annotated Video Result")
            st.video(out_file)

            # Download button
            with open(out_file, "rb") as f:
                st.download_button(
                    label="Download Annotated Video",
                    data=f.read(),
                    file_name="annotated_video.mp4",
                    mime="video/mp4",
                    key="download_video"
                )
            
            # Clean up temporary files
            os.unlink(tfile_path)
            os.unlink(out_file)
