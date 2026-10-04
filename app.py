import os
import re
import base64
import av
import textwrap
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer


# ============================================================
# PAGE CONFIGURATION
# ============================================================

favicon_path = "assets/logo.png"

if os.path.exists(favicon_path):
    page_icon = favicon_path
else:
    page_icon = None

st.set_page_config(
    page_title="Face Mask Detector",
    page_icon=page_icon,
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# HTML HELPER
# ------------------------------------------------------------
# Flattens indentation and newlines so Streamlit's Markdown
# parser never treats the HTML as a code block.
# ============================================================

def html(markup: str):
    flat = re.sub(r"\s+", " ", textwrap.dedent(markup)).strip()
    st.markdown(flat, unsafe_allow_html=True)


# ============================================================
# CUSTOM CSS
# ============================================================

html(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    html { scroll-behavior: smooth; }

    .stApp {
        background:
            radial-gradient(circle at 12% 8%, rgba(56, 189, 248, 0.22), transparent 32%),
            radial-gradient(circle at 88% 12%, rgba(52, 211, 153, 0.18), transparent 30%),
            radial-gradient(circle at 50% 100%, rgba(125, 211, 252, 0.20), transparent 40%),
            linear-gradient(180deg, #ffffff 0%, #f3fbff 55%, #eefcf6 100%);
        color: #0b2545;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 0.5rem;
        padding-bottom: 4rem;
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { background: transparent !important; }


    /* ---------- LOGO ---------- */

    .logo-wrap {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 0;
        margin-bottom: -20px;
        animation: fadeUp 0.7s ease-out;
    }

    .logo-wrap img {
        width: 340px;
        height: auto;
        display: block;
        filter: drop-shadow(0 14px 36px rgba(14, 165, 233, 0.30));
    }

    @media (max-width: 800px) {
        .logo-wrap img { width: 220px; }
    }


    /* ---------- HERO ---------- */

    .hero {
        text-align: center;
        padding: 4px 20px 20px 20px;
        animation: fadeUp 0.8s ease-out;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 9px;
        padding: 8px 18px;
        border: 1px solid rgba(14, 165, 233, 0.35);
        border-radius: 999px;
        background: linear-gradient(90deg, rgba(56, 189, 248, 0.14), rgba(52, 211, 153, 0.14));
        color: #0284c7;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 12px;
        backdrop-filter: blur(10px);
    }

    .hero-badge::before {
        content: '';
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #10b981;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.6);
        animation: pulseDot 1.6s ease-out infinite;
    }

    @keyframes pulseDot {
        0%   { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.55); }
        70%  { box-shadow: 0 0 0 12px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    .hero h1 {
        font-family: 'Space Grotesk', 'Inter', sans-serif;
        font-size: clamp(46px, 8vw, 96px);
        font-weight: 700;
        letter-spacing: -3px;
        margin: 12px 0 0 0;
        line-height: 1;
        display: inline-block;
        color: #0ea5e9;
        background: linear-gradient(120deg, #0369a1 0%, #0ea5e9 35%, #22d3ee 60%, #10b981 100%);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero h1 .letter {
        display: inline-block;
        transition: transform 0.35s cubic-bezier(.34, 1.56, .64, 1), filter 0.35s ease;
        will-change: transform;
    }

    .hero h1 .letter:hover {
        transform: translateY(-14px) scale(1.15) rotate(-4deg);
        filter: drop-shadow(0 8px 18px rgba(14, 165, 233, 0.55));
    }

    .hero h1 .space { display: inline-block; width: 0.35em; }

    .hero p {
        max-width: 720px;
        margin: 22px auto 0 auto;
        color: #3b5a7a;
        font-size: 18px;
        line-height: 1.7;
        font-weight: 400;
    }


    /* ---------- HERO CTA BUTTON ---------- */

    .hero-cta {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        margin-top: 30px;
        padding: 14px 34px;
        border-radius: 999px;
        background: linear-gradient(135deg, #0ea5e9, #10b981);
        color: #ffffff !important;
        font-family: 'Space Grotesk', 'Inter', sans-serif;
        font-size: 16px;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-decoration: none !important;
        box-shadow: 0 14px 34px rgba(14, 165, 233, 0.35);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .hero-cta:hover {
        transform: translateY(-3px);
        box-shadow: 0 20px 44px rgba(16, 185, 129, 0.45);
    }

    .hero-cta .arrow {
        display: inline-block;
        animation: bounceDown 1.6s ease-in-out infinite;
        font-size: 18px;
        line-height: 1;
    }

    @keyframes bounceDown {
        0%, 100% { transform: translateY(0); }
        50%      { transform: translateY(5px); }
    }


    /* ---------- FLOATING TEST BUTTON ---------- */

    .floating-cta {
        position: fixed;
        bottom: 28px;
        right: 28px;
        z-index: 9999;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 13px 24px;
        border-radius: 999px;
        background: linear-gradient(135deg, #0ea5e9, #10b981);
        color: #ffffff !important;
        font-family: 'Space Grotesk', 'Inter', sans-serif;
        font-size: 14px;
        font-weight: 700;
        letter-spacing: 0.4px;
        text-decoration: none !important;
        box-shadow: 0 14px 32px rgba(14, 165, 233, 0.42);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .floating-cta:hover {
        transform: translateY(-3px) scale(1.04);
        box-shadow: 0 20px 42px rgba(16, 185, 129, 0.52);
    }

    @media (max-width: 800px) {
        .floating-cta {
            bottom: 18px;
            right: 18px;
            padding: 11px 18px;
            font-size: 13px;
        }
    }


    /* ---------- SECTIONS ---------- */

    .section {
        margin-top: 70px;
        margin-bottom: 25px;
        animation: fadeUp 0.8s ease-out;
    }

    .section-label {
        color: #0ea5e9;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 2.2px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 36px;
        font-weight: 700;
        margin: 0;
        color: #0b2545;
        letter-spacing: -1px;
    }

    .section-description {
        color: #4a6582;
        font-size: 16px;
        line-height: 1.7;
        max-width: 780px;
        margin-top: 12px;
    }


    /* ---------- GLASS CARD ---------- */

    .glass-card {
        position: relative;
        background: linear-gradient(145deg, rgba(255, 255, 255, 0.85), rgba(240, 253, 255, 0.75));
        border: 1px solid rgba(14, 165, 233, 0.18);
        border-radius: 24px;
        padding: 30px;
        backdrop-filter: blur(16px);
        box-shadow: 0 12px 40px rgba(14, 165, 233, 0.10), 0 2px 8px rgba(0, 0, 0, 0.03);
        transition: transform 0.35s cubic-bezier(.34, 1.3, .64, 1), border-color 0.3s ease, box-shadow 0.35s ease;
        overflow: hidden;
        animation: fadeUp 0.8s ease-out;
    }

    .glass-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(115deg, transparent 40%, rgba(56, 189, 248, 0.14) 50%, transparent 60%);
        transform: translateX(-100%);
        transition: transform 0.9s ease;
        pointer-events: none;
    }

    .glass-card:hover::before { transform: translateX(100%); }

    .glass-card:hover {
        transform: translateY(-6px);
        border-color: rgba(14, 165, 233, 0.45);
        box-shadow: 0 24px 60px rgba(14, 165, 233, 0.20), 0 0 0 1px rgba(52, 211, 153, 0.15);
    }


    /* ---------- PIPELINE ---------- */

    .pipeline {
        display: grid;
        grid-template-columns: repeat(5, minmax(120px, 1fr));
        gap: 14px;
        margin-top: 28px;
    }

    .pipeline-item {
        text-align: center;
        padding: 22px 12px;
        border-radius: 18px;
        background: linear-gradient(160deg, rgba(255, 255, 255, 0.95), rgba(236, 254, 255, 0.85));
        border: 1px solid rgba(14, 165, 233, 0.16);
        transition: transform 0.3s cubic-bezier(.34, 1.4, .64, 1), border-color 0.3s ease, box-shadow 0.3s ease;
    }

    .pipeline-item:hover {
        transform: translateY(-8px);
        border-color: rgba(16, 185, 129, 0.45);
        box-shadow: 0 18px 36px rgba(14, 165, 233, 0.18);
    }

    .pipeline-number {
        width: 38px;
        height: 38px;
        margin: 0 auto 12px auto;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, #38bdf8, #10b981);
        color: #ffffff;
        font-weight: 800;
        font-size: 15px;
        box-shadow: 0 6px 16px rgba(14, 165, 233, 0.35);
        transition: transform 0.35s ease;
    }

    .pipeline-item:hover .pipeline-number { transform: rotate(360deg) scale(1.1); }

    .pipeline-item strong { display: block; color: #0b2545; font-size: 14px; font-weight: 700; }
    .pipeline-item span { display: block; color: #64748b; font-size: 12px; margin-top: 5px; }


    /* ---------- CLASS CARDS ---------- */

    .class-card {
        position: relative;
        min-height: 160px;
        padding: 26px;
        border-radius: 22px;
        background: linear-gradient(150deg, rgba(255, 255, 255, 0.95), rgba(240, 253, 255, 0.75));
        border: 1px solid rgba(14, 165, 233, 0.16);
        transition: transform 0.35s cubic-bezier(.34, 1.3, .64, 1), box-shadow 0.35s ease, border-color 0.3s ease;
        overflow: hidden;
    }

    .class-card::after {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 4px;
        background: linear-gradient(180deg, #38bdf8, #10b981);
        opacity: 0;
        transition: opacity 0.3s ease;
    }

    .class-card:hover {
        transform: translateY(-8px) scale(1.02);
        border-color: rgba(16, 185, 129, 0.45);
        box-shadow: 0 22px 50px rgba(14, 165, 233, 0.20);
    }

    .class-card:hover::after { opacity: 1; }

    .class-name {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 22px;
        font-weight: 700;
        color: #0b2545;
        margin-bottom: 10px;
    }

    .class-description { color: #4a6582; line-height: 1.6; font-size: 14px; }


    /* ---------- METRIC CARDS ---------- */

    .metric-card {
        position: relative;
        text-align: center;
        padding: 28px 15px;
        border-radius: 22px;
        background: linear-gradient(145deg, rgba(56, 189, 248, 0.14), rgba(16, 185, 129, 0.10));
        border: 1px solid rgba(14, 165, 233, 0.22);
        overflow: hidden;
        transition: transform 0.35s cubic-bezier(.34, 1.3, .64, 1), box-shadow 0.35s ease;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(115deg, transparent 40%, rgba(255, 255, 255, 0.65) 50%, transparent 60%);
        transform: translateX(-100%);
        transition: transform 0.9s ease;
    }

    .metric-card:hover::before { transform: translateX(100%); }

    .metric-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 20px 44px rgba(14, 165, 233, 0.25);
    }

    .metric-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 34px;
        font-weight: 700;
        background: linear-gradient(120deg, #0369a1, #0ea5e9, #10b981);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
    }

    .metric-label {
        color: #3b5a7a;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.4px;
        margin-top: 8px;
    }


    /* ---------- APPLICATION CARDS ---------- */

    .application-card {
        padding: 26px;
        min-height: 180px;
        border-radius: 22px;
        background: linear-gradient(160deg, rgba(255, 255, 255, 0.95), rgba(236, 254, 255, 0.85));
        border: 1px solid rgba(14, 165, 233, 0.16);
        transition: transform 0.35s cubic-bezier(.34, 1.3, .64, 1), border-color 0.3s ease, box-shadow 0.35s ease;
    }

    .application-card:hover {
        transform: translateY(-8px);
        border-color: rgba(16, 185, 129, 0.45);
        box-shadow: 0 22px 50px rgba(14, 165, 233, 0.18);
    }

    .application-card h4 {
        font-family: 'Space Grotesk', sans-serif;
        color: #0b2545;
        margin-bottom: 10px;
        font-size: 18px;
        font-weight: 700;
    }

    .application-card p { color: #4a6582; font-size: 14px; line-height: 1.65; }


    /* ---------- INFO BOX ---------- */

    .info-box {
        padding: 28px;
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.12), rgba(16, 185, 129, 0.10));
        border: 1px solid rgba(14, 165, 233, 0.22);
        color: #2c4a6b;
        line-height: 1.75;
        font-size: 15px;
    }

    .info-box strong { color: #0369a1; font-weight: 700; }


    /* ---------- FOOTER ---------- */

    .custom-footer {
        position: relative;
        margin-top: 90px;
        padding: 40px 10px 15px 10px;
        text-align: center;
        color: #64748b;
        font-size: 13px;
        letter-spacing: 0.3px;
    }

    .custom-footer::before {
        content: '';
        position: absolute;
        top: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 60%;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(14, 165, 233, 0.6), rgba(16, 185, 129, 0.6), transparent);
    }

    .custom-footer strong {
        background: linear-gradient(120deg, #0369a1, #0ea5e9, #10b981);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 16px;
        letter-spacing: 1.5px;
    }


    /* ---------- WIDGETS ---------- */

    div[role="radiogroup"] label {
        padding: 10px 18px !important;
        border-radius: 999px !important;
        border: 1px solid rgba(14, 165, 233, 0.22) !important;
        background: rgba(255, 255, 255, 0.75) !important;
        transition: all 0.25s ease !important;
        margin-right: 8px !important;
        font-weight: 600 !important;
        color: #0b2545 !important;
    }

    div[role="radiogroup"] label:hover {
        border-color: rgba(14, 165, 233, 0.55) !important;
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(14, 165, 233, 0.15);
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0ea5e9, #10b981) !important;
        border: none !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px !important;
        padding: 12px 26px !important;
        border-radius: 14px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 10px 24px rgba(14, 165, 233, 0.30) !important;
    }

    .stButton > button[kind="primary"]:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 16px 36px rgba(16, 185, 129, 0.40) !important;
    }

    section[data-testid="stFileUploaderDropzone"] {
        background: rgba(255, 255, 255, 0.8) !important;
        border: 2px dashed rgba(14, 165, 233, 0.4) !important;
        border-radius: 18px !important;
        transition: all 0.3s ease !important;
    }

    section[data-testid="stFileUploaderDropzone"]:hover {
        border-color: rgba(16, 185, 129, 0.6) !important;
        background: rgba(236, 254, 255, 0.9) !important;
    }

    .stSpinner > div { border-top-color: #0ea5e9 !important; }


    /* ---------- ANIMATIONS ---------- */

    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(18px); }
        to   { opacity: 1; transform: translateY(0); }
    }


    /* ---------- MOBILE ---------- */

    @media (max-width: 800px) {
        .pipeline { grid-template-columns: 1fr; }
        .hero { padding-top: 12px; }
        .hero h1 { letter-spacing: -2px; }
        .section-title { font-size: 28px; }
        .hero h1 .letter:hover { transform: translateY(-8px) scale(1.1); }
    }
    </style>
    """
)


# ============================================================
# LOGO
# ============================================================

logo_path = "assets/logo.png"

if os.path.exists(logo_path):
    logo_b64 = base64.b64encode(open(logo_path, "rb").read()).decode()
    html(
        f"""
        <div class="logo-wrap">
            <img src="data:image/png;base64,{logo_b64}" alt="Face Mask Detector logo">
        </div>
        """
    )
else:
    st.warning(f"Logo not found at: {logo_path}")


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return YOLO("best.pt")


model = load_model()


# ============================================================
# LIVE CAMERA STATE
# ============================================================

frame_count = 0
last_annotated_frame = None


# ============================================================
# LIVE CAMERA CALLBACK
# ============================================================

def video_frame_callback(frame):

    global frame_count
    global last_annotated_frame

    frame_count += 1

    img = frame.to_ndarray(format="bgr24")

    if frame_count % 2 != 0:
        if last_annotated_frame is not None:
            return av.VideoFrame.from_ndarray(last_annotated_frame, format="bgr24")
        return frame

    results = model.predict(source=img, imgsz=480, conf=0.40, verbose=False)
    annotated_frame = results[0].plot()
    last_annotated_frame = annotated_frame

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")


# ============================================================
# HERO
# ============================================================

def hover_title(text):
    letters = []
    for ch in text:
        if ch == " ":
            letters.append('<span class="space"></span>')
        else:
            letters.append(f'<span class="letter">{ch}</span>')
    return "".join(letters)


hero_letters = hover_title("FACE MASK AI")

html(
    f"""
    <div class="hero">
        <div class="hero-badge">FACE MASK DETECTOR</div>
        <h1>{hero_letters}</h1>
        <p>Real-time computer vision for detecting whether a face mask is worn correctly, incorrectly, or not at all.</p>
        <a href="#test-section" class="hero-cta">Test Now <span class="arrow">↓</span></a>
    </div>
    """
)


# ============================================================
# FLOATING CTA
# ============================================================

html(
    """
    <a href="#test-section" class="floating-cta">Test Now <span>↓</span></a>
    """
)


# ============================================================
# INTRODUCTION
# ============================================================

html(
    """
    <div class="section">
        <div class="section-label">THE PROJECT</div>
        <div class="section-title">Intelligent mask detection</div>
        <div class="section-description">
            Face Mask AI is a computer-vision prototype built around a YOLO object-detection model.
            It analyzes images and live camera frames and identifies three mask-wearing states.
        </div>
    </div>
    """
)


# ============================================================
# HOW IT WORKS
# ============================================================

html(
    """
    <div class="glass-card">
        <div class="section-label">HOW IT WORKS</div>
        <div class="section-title">From image to prediction</div>

        <div class="pipeline">
            <div class="pipeline-item">
                <div class="pipeline-number">1</div>
                <strong>Input</strong>
                <span>Image or camera</span>
            </div>
            <div class="pipeline-item">
                <div class="pipeline-number">2</div>
                <strong>Processing</strong>
                <span>Image preparation</span>
            </div>
            <div class="pipeline-item">
                <div class="pipeline-number">3</div>
                <strong>YOLO</strong>
                <span>Object detection</span>
            </div>
            <div class="pipeline-item">
                <div class="pipeline-number">4</div>
                <strong>Prediction</strong>
                <span>Class + confidence</span>
            </div>
            <div class="pipeline-item">
                <div class="pipeline-number">5</div>
                <strong>Result</strong>
                <span>Annotated output</span>
            </div>
        </div>
    </div>
    """
)


# ============================================================
# TEST SECTION (anchor target)
# ============================================================

html(
    """
    <div class="section" id="test-section">
        <div class="section-label">LIVE TESTING</div>
        <div class="section-title">Test the model</div>
        <div class="section-description">
            Upload an image or activate your camera to run the trained model in real time.
        </div>
    </div>
    """
)


# ============================================================
# DETECTION MODE
# ============================================================

mode = st.radio(
    "Detection mode",
    ["Image Upload", "Live Camera"],
    horizontal=True
)


# ============================================================
# IMAGE UPLOAD
# ============================================================

if mode == "Image Upload":

    html(
        """
        <div class="glass-card">
            <div class="section-label">IMAGE DETECTION</div>
            <div class="section-title">Upload an image</div>
        </div>
        """
    )

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        st.subheader("Original image")
        st.image(image, use_container_width=True)

        if st.button("Detect Mask", type="primary", use_container_width=True):

            with st.spinner("Analyzing image..."):
                results = model.predict(source=image, imgsz=640, conf=0.40, verbose=False)
                annotated_image = results[0].plot()

            st.subheader("Detection result")
            st.image(annotated_image, channels="BGR", use_container_width=True)


# ============================================================
# LIVE CAMERA
# ============================================================

else:

    html(
        """
        <div class="glass-card">
            <div class="section-label">REAL-TIME DETECTION</div>
            <div class="section-title">Live camera</div>
            <div class="section-description">
                Start your camera and position people inside the frame.
                The model will process the video stream in real time.
            </div>
        </div>
        """
    )

    webrtc_streamer(
        key="face-mask-live",
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False}
    )


# ============================================================
# DETECTION CLASSES
# ============================================================

html(
    """
    <div class="section">
        <div class="section-label">DETECTION CLASSES</div>
        <div class="section-title">What the model detects</div>
    </div>
    """
)

class_col1, class_col2, class_col3 = st.columns(3)

with class_col1:
    html(
        """
        <div class="class-card">
            <div class="class-name">Masque</div>
            <div class="class-description">
                A face mask is detected and appears to be worn correctly.
            </div>
        </div>
        """
    )

with class_col2:
    html(
        """
        <div class="class-card">
            <div class="class-name">Notcorrect</div>
            <div class="class-description">
                A face mask is detected but appears to be worn incorrectly.
            </div>
        </div>
        """
    )

with class_col3:
    html(
        """
        <div class="class-card">
            <div class="class-name">PasMasque</div>
            <div class="class-description">
                No face mask is detected on the person.
            </div>
        </div>
        """
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

html(
    """
    <div class="section">
        <div class="section-label">MODEL PERFORMANCE</div>
        <div class="section-title">Dataset A test results</div>
        <div class="section-description">
            These metrics were obtained from the independent test split of the original training dataset.
            They describe detection performance and should not be interpreted as simple accuracy.
        </div>
    </div>
    """
)

metric1, metric2, metric3, metric4 = st.columns(4)

with metric1:
    html('<div class="metric-card"><div class="metric-value">72.90%</div><div class="metric-label">Precision</div></div>')

with metric2:
    html('<div class="metric-card"><div class="metric-value">59.54%</div><div class="metric-label">Recall</div></div>')

with metric3:
    html('<div class="metric-card"><div class="metric-value">68.10%</div><div class="metric-label">mAP@50</div></div>')

with metric4:
    html('<div class="metric-card"><div class="metric-value">37.63%</div><div class="metric-label">mAP@50–95</div></div>')


# ============================================================
# INDEPENDENT DATASET TESTING
# ============================================================

html(
    """
    <div class="section">
        <div class="section-label">GENERALIZATION</div>
        <div class="section-title">Independent Dataset B testing</div>
    </div>
    """
)

html(
    """
    <div class="info-box">
        The trained model was tested on a separate Kaggle dataset containing <strong>2,079 images</strong>.
        <br><br>
        The dataset was used strictly for independent inference testing.
        It was <strong>not used for retraining or fine-tuning</strong>.
        <br><br>
        Because Dataset B does not contain bounding-box annotations,
        formal precision, recall, and mAP values were not calculated for this dataset.
    </div>
    """
)


# ============================================================
# POTENTIAL APPLICATIONS
# ============================================================

html(
    """
    <div class="section">
        <div class="section-label">POTENTIAL APPLICATIONS</div>
        <div class="section-title">Where computer vision can help</div>
        <div class="section-description">
            A system like this could serve as a starting point for visual compliance,
            monitoring and computer-vision research in several environments.
        </div>
    </div>
    """
)

app_col1, app_col2, app_col3 = st.columns(3)

with app_col1:
    html(
        """
        <div class="application-card">
            <h4>Healthcare</h4>
            <p>Potential support for monitoring mask-use requirements in controlled healthcare environments.</p>
        </div>
        """
    )

with app_col2:
    html(
        """
        <div class="application-card">
            <h4>Manufacturing</h4>
            <p>Potential PPE-compliance monitoring in industrial and controlled production environments.</p>
        </div>
        """
    )

with app_col3:
    html(
        """
        <div class="application-card">
            <h4>Food Processing</h4>
            <p>Potential support for face-covering requirements in controlled food-production environments.</p>
        </div>
        """
    )

app_col4, app_col5, app_col6 = st.columns(3)

with app_col4:
    html(
        """
        <div class="application-card">
            <h4>Education</h4>
            <p>A practical demonstration of object detection, computer vision and machine-learning deployment.</p>
        </div>
        """
    )

with app_col5:
    html(
        """
        <div class="application-card">
            <h4>Research</h4>
            <p>A foundation for experimentation with computer vision, model evaluation and generalization.</p>
        </div>
        """
    )

with app_col6:
    html(
        """
        <div class="application-card">
            <h4>Public Facilities</h4>
            <p>Potential monitoring support in environments where mask requirements are applicable.</p>
        </div>
        """
    )


# ============================================================
# LIMITATIONS
# ============================================================

html(
    """
    <div class="section">
        <div class="section-label">LIMITATIONS</div>
        <div class="section-title">What the prototype should be understood as</div>
    </div>
    """
)

html(
    """
    <div class="info-box">
        Face Mask AI is a <strong>computer-vision prototype</strong>, not a certified safety or compliance system.
        <br><br>
        Detection performance can be affected by lighting, camera quality, motion blur,
        occlusion, viewing angle and image composition.
        <br><br>
        The independent Dataset B does not contain bounding-box annotations, so qualitative
        inference testing was used rather than claiming unsupported mAP or accuracy values.
    </div>
    """
)


# ============================================================
# FOOTER
# ============================================================

html(
    """
    <div class="custom-footer">
        <strong>FACE MASK AI</strong><br><br>
        Computer Vision × Artificial Intelligence<br>
        YOLO-based face mask detection prototype
    </div>
    """
)
