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

    /* Header */
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

    /* File Upload */
    .stFileUploader {
        background: rgba(255,255,255,0.05);
        border: 2px dashed #64748b;
        border-radius: 12px;
        padding: 1rem;
    }
    .stFileUploader:hover {
        border-color: #3b82f6 !important;
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
    }
    .stButton button:hover {
        transform: scale(1.03);
    }

    /* Slider */
    .stSlider { margin-top: 25px !important; margin-bottom: 35px !important; }
    .stSlider label { font-size: 1rem !important; margin-bottom: 10px !important; }

    .stImage > img {
        border-radius: 12px !important;
        box-shadow: 0 0 25px rgba(0,0,0,0.55);
        border: 2px solid rgba(255,255,255,0.08);
    }

</style>
""", unsafe_allow_html=True)


# --- Title ---
st.title("YOLOv11 Real-Time Object Detection")


# =============== LOAD MODEL ==================
@st.cache_resource
def load_model():
    try:
        return YOLO("best_2.pt")   # ← عدل هنا
    except:
        st.error("Error loading YOLO model.")
        return None

model = load_model()


# =============== SIDEBAR ====================
with st.sidebar:
    st.header("Choose Mode")

    mode = st.radio(
        "Select Mode:",
        ["Image/Video", "Live Camera"],
        index=0
    )

    conf_value = st.slider(
        "Confidence Threshold", 0.1, 1.0, 0.25, step=0.01
    )

    if mode == "Image/Video":
        uploaded_file = st.file_uploader(
            "Choose image or video:",
            type=["jpg", "jpeg", "png", "mp4", "avi"]
        )
        run_prediction = st.button("Start Prediction")

    else:
        start_cam = st.button("Start Camera")
        stop_cam = st.button("Stop Camera")


# =========================================================
# =============== LIVE CAMERA MODE ========================
# =========================================================

if mode == "Live Camera" and model:

    st.subheader("Live Camera Detection")
    cam_placeholder = st.empty()

    # حالة تشغيل/إيقاف الكاميرا
    if "cam_running" not in st.session_state:
        st.session_state.cam_running = False

    # زر تشغيل
    if start_cam:
        st.session_state.cam_running = True

    # زر إيقاف
    if stop_cam:
        st.session_state.cam_running = False

    # تشغيل الكاميرا
    if st.session_state.cam_running:

        cap = cv2.VideoCapture(0)

        while True:

            # لو المستخدم ضغط Stop → اخرج فوراً
            if not st.session_state.cam_running:
                break

            ret, frame = cap.read()
            if not ret:
                st.error("Camera Not Found!")
                break

            results = model(frame, conf=conf_value)[0]
            annotated = results.plot()
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            cam_placeholder.image(annotated, use_container_width=True)

        # إغلاق الكاميرا نهائياً
        cap.release()
        cam_placeholder.empty()   # مسح آخر صورة
        st.success("Camera stopped successfully!")

    st.stop()


# =========================================================
# =============== IMAGE / VIDEO MODE ======================
# =========================================================

if mode == "Image/Video" and uploaded_file and run_prediction and model:

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
                ann = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)
                st.image(ann, use_container_width=True)

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

                ann = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(ann, use_container_width=True)

            cap.release()
            out.release()

            status.success("Video Done!")
            st.video(out_path)


else:
    if mode == "Image/Video":
        st.info("Upload a file and press Start Prediction.")
