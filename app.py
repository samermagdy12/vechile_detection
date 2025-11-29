import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile
import os
import numpy as np

# --- Page Config ---
st.set_page_config(
    page_title="YOLOv11 Object Detection",
    layout="wide",
    initial_sidebar_state="auto",
)

# --- Full Dark Glass Theme + Slider Fix ---
st.markdown("""
<style>

    /* Global Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        color: #f1f5f9 !important;
    }

    /* Main Container */
    .main .block-container {
        padding: 2rem 2.5rem !important;
        background: rgba(255,255,255,0.02);
        border-radius: 18px;
        box-shadow: 0 0 25px rgba(0,0,0,0.25);
        backdrop-filter: blur(6px);
    }

    /* Header Blur */
    .stApp > header {
        background: rgba(15, 23, 42, 0.7) !important;
        backdrop-filter: blur(12px);
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }

    /* Title Style */
    h1 {
        color: #60a5fa !important;
        font-weight: 800 !important;
        text-shadow: 0 0 10px rgba(96,165,250,0.45);
        padding-bottom: 0.6rem;
        border-bottom: 2px solid #3b82f6;
    }

    section[data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.8) !important;
        backdrop-filter: blur(8px);
        border-right: 1px solid rgba(255,255,255,0.1);
        padding: 1.2rem;
    }

    /* File Uploader Style */
    .stFileUploader {
        background: rgba(255,255,255,0.05);
        border: 2px dashed #64748b;
        border-radius: 12px;
        padding: 1rem;
        transition: 0.3s ease;
    }
    .stFileUploader:hover {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 10px rgba(59,130,246,0.4);
    }

    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
        color: white !important;
        border-radius: 10px;
        padding: 0.7rem 1.4rem;
        font-size: 1.05rem;
        border: none;
        font-weight: 600;
        transition: 0.25s ease;
        box-shadow: 0 0 10px rgba(37,99,235,0.35);
    }
    .stButton button:hover {
        transform: scale(1.03);
        box-shadow: 0 0 15px rgba(37,99,235,0.45);
    }

    /* SLIDER FIX — Perfect Alignment */
    .stSlider {
        margin-top: 25px !important;
        margin-bottom: 35px !important;
    }

    .stSlider label {
        font-size: 1rem !important;
        margin-bottom: 10px !important;
        display: block !important;
    }

    /* Slider box */
    [data-baseweb="slider"] {
        padding: 14px !important;
        border-radius: 12px !important;
        background: rgba(255,255,255,0.06) !important;
    }

    /* Slider track */
    [data-baseweb="slider"] > div > div {
        height: 6px !important;
        border-radius: 8px !important;
        background: rgba(255,255,255,0.15) !important;
    }

    /* Active progress line */
    [data-baseweb="track"] {
        background: linear-gradient(90deg, #3b82f6, #2563eb) !important;
    }

    /* Slider thumb */
    [role="slider"] {
        width: 18px !important;
        height: 18px !important;
        background: #3b82f6 !important;
        border: 2px solid #93c5fd !important;
        box-shadow: 0 0 10px rgba(59,130,246,0.8);
    }

    /* Min/Max values clean alignment */
    .stSlider > div > div:nth-child(3) {
        display: flex !important;
        justify-content: space-between !important;
        padding-top: 6px !important;
        font-size: 0.9rem !important;
    }

    /* Images */
    .stImage > img {
        border-radius: 12px !important;
        box-shadow: 0 0 25px rgba(0,0,0,0.55);
        border: 2px solid rgba(255,255,255,0.08);
    }

</style>
""", unsafe_allow_html=True)

# --- Title ---
st.title("YOLOv11 Real-Time Object Detection")

# --- Load Model ---
@st.cache_resource
def load_model():
    try:
        return YOLO(r"D:\\deploy\\best (11).pt")
    except:
        st.error("Error loading YOLOv11 model.")
        return None

model = load_model()

# --- Sidebar Layout ---
with st.sidebar:
    st.header("Upload Media")

    uploaded_file = st.file_uploader(
        "Choose Image or Video",
        type=["jpg", "jpeg", "png", "mp4", "avi"]
    )

    conf_value = st.slider(
        "Confidence Threshold",
        0.1, 1.0, 0.25, step=0.01
    )

    run_prediction = st.button("Start Prediction")


# --- Main Logic ---
if uploaded_file and run_prediction and model:

    with tempfile.TemporaryDirectory() as temp_dir:

        tfile_path = os.path.join(temp_dir, uploaded_file.name)
        with open(tfile_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        ext = uploaded_file.name.split('.')[-1].lower()

        # ------------------ IMAGE ------------------
        if ext in ["jpg", "jpeg", "png"]:
            st.subheader("Processing Image...")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Original")
                img = Image.open(tfile_path)
                st.image(img, use_container_width=True)

            with col2:
                st.markdown("### Annotated")
                results = model.predict(tfile_path, conf=conf_value)
                ann = results[0].plot()

                rgb = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)
                st.image(rgb, use_container_width=True)

                outpath = os.path.join(temp_dir, "annotated.png")
                cv2.imwrite(outpath, ann)

                with open(outpath, "rb") as f:
                    st.download_button(
                        "Download Annotated Image",
                        data=f.read(),
                        file_name="annotated_image.png",
                        mime="image/png"
                    )

            st.success("Image processed successfully.")

        # ------------------ VIDEO ------------------
        else:
            st.subheader("Processing Video...")
            st.video(tfile_path)

            status = st.empty()
            status.info("Processing frames...")

            outpath = os.path.join(temp_dir, "annotated_video.mp4")

            cap = cv2.VideoCapture(tfile_path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            out = cv2.VideoWriter(outpath, fourcc, fps, (w, h))

            st_frame = st.empty()
            bar = st.progress(0)
            fcount = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                results = model.predict(frame, conf=conf_value, verbose=False)
                ann = results[0].plot()

                out.write(ann)

                rgb = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)
                st_frame.image(rgb, use_container_width=True)

                fcount += 1
                bar.progress((fcount % 100))

            cap.release()
            out.release()

            status.success("Video done!")
            st.video(outpath)

            with open(outpath, "rb") as f:
                st.download_button(
                    "Download Annotated Video",
                    data=f.read(),
                    file_name="annotated_video.mp4",
                    mime="video/mp4"
                )

else:
    st.info("Upload a file and press Start Prediction.")
