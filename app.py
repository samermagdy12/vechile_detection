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

# --- FULL UI / DARK GLASS THEME ---
st.markdown("""
<style>

    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        color: #f1f5f9 !important;
    }

    .main .block-container {
        padding: 2rem 2.5rem !important;
        background: rgba(255,255,255,0.02);
        border-radius: 18px;
        box-shadow: 0 0 25px rgba(0,0,0,0.25);
        backdrop-filter: blur(6px);
    }

    .stApp > header {
        background: rgba(15, 23, 42, 0.7) !important;
        backdrop-filter: blur(12px);
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }

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

</style>
""", unsafe_allow_html=True)

# --- Title ---
st.title("YOLOv11 Object Detection")

# =============== LOAD MODEL ==================
@st.cache_resource
def load_model():
    try:
        return YOLO("best_2.pt")
    except:
        st.error("Error loading YOLO model.")
        return None

model = load_model()

# =============== SIDEBAR ====================
with st.sidebar:
    st.header("Detection Settings")

    conf_value = st.slider(
        "Confidence Threshold", 0.1, 1.0, 0.25, step=0.01
    )

    uploaded_file = st.file_uploader(
        "Choose image or video:",
        type=["jpg", "jpeg", "png", "mp4", "avi"]
    )

    run_prediction = st.button("Start Prediction")

# =========================================================
# =============== IMAGE / VIDEO MODE ONLY =================
# =========================================================

if uploaded_file and run_prediction and model:

    with tempfile.TemporaryDirectory() as tmp:

        path = os.path.join(tmp, uploaded_file.name)

        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        ext = uploaded_file.name.split('.')[-1].lower()

        # -------- IMAGE --------
        if ext in ["jpg", "jpeg", "png"]:

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Original")
                img = Image.open(path)
                st.image(img, use_container_width=True)

            with col2:
                st.markdown("### Annotated")
                res = model(path, conf=conf_value)[0]
                ann = res.plot()
                ann_rgb = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)
                st.image(ann_rgb, use_container_width=True)

                # ---------- DOWNLOAD BUTTON FOR IMAGE ----------
                img_bytes = cv2.imencode(".png", ann)[1].tobytes()
                st.download_button(
                    label="⬇ Download Annotated Image",
                    data=img_bytes,
                    file_name="annotated_image.png",
                    mime="image/png"
                )

            st.success("Image processed successfully.")

        # -------- VIDEO --------
        else:
            st.subheader("Processing video...")

            st.video(path)
            status = st.info("Processing frames...")

            out_path = os.path.join(tmp, "annotated_video.mp4")

            cap = cv2.VideoCapture(path)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
            frame_placeholder = st.empty()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                res = model(frame, conf=conf_value, verbose=False)[0]
                ann = res.plot()
                out.write(ann)

                ann_rgb = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(ann_rgb, use_container_width=True)

            cap.release()
            out.release()

            status.success("Video Done!")
            st.video(out_path)

            # ---------- DOWNLOAD BUTTON FOR VIDEO ----------
            with open(out_path, "rb") as f:
                st.download_button(
                    label="⬇ Download Annotated Video",
                    data=f,
                    file_name="annotated_video.mp4",
                    mime="video/mp4"
                )

else:
    st.info("Upload a file and press Start Prediction.")
