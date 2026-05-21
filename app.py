import streamlit as st
from PIL import Image
from caption_model import generate_image_caption
from utils import translate_caption, text_to_speech
import time

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="VisionAI | Image Caption Generator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. THEME & CUSTOM CSS ---
st.markdown("""
    <style>
    .main-title { font-size: 2.8rem; font-weight: 800; color: #1E88E5; margin-bottom: 0.5rem; }
    .sub-title { font-size: 1.1rem; color: #666; margin-bottom: 2rem; }
    .caption-box { padding: 1.5rem; background-color: #f8f9fa; border-left: 5px solid #1E88E5; border-radius: 5px; margin: 1rem 0; }
    .dark .caption-box { background-color: #1e1e1e; border-left: 5px solid #1E88E5; }
    </style>
""", unsafe_allow_html=True)

# Initialize Session State Variables
if "history" not in st.session_state:
    st.session_state.history = []

# --- 3. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8649/8649607.png", width=100)
    st.title("VisionAI Engine")
    st.markdown("An advanced Deep Learning system translating pixel matrices into meaningful natural language descriptors.")
    st.markdown("---")
    
    app_mode = st.radio("Navigation", ["Dashboard", "Caption History", "Technical Documentation"])
    
    st.markdown("---")
    st.markdown("### Model Properties")
    st.info("🤖 **Model:** BLIP-Base\n\n🏋️ **Weights:** Pretrained (Salesforce)\n\n⚡ **Hardware:** Auto-detects CUDA/CPU")

# --- 4. MAIN APP LOGIC ---
if app_mode == "Dashboard":
    st.markdown("<h1 class='main-title'>AI Image Caption Generator</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Upload an image or use your webcam to let artificial intelligence describe the scene.</p>", unsafe_allow_html=True)
    
    # Input Selection
    input_source = st.selectbox("Choose Input Source", ["Local File Upload", "Live Webcam Capture"])
    uploaded_image = None
    
    if input_source == "Local File Upload":
        file = st.file_uploader("Drop your image here...", type=["jpg", "jpeg", "png"])
        if file is not None:
            try:
                uploaded_image = Image.open(file).convert("RGB")
            except Exception:
                st.error("Invalid image formatting. Please try a different standard file.")
    else:
        webcam_file = st.camera_input("Take a photo using your camera")
        if webcam_file is not None:
            uploaded_image = Image.open(webcam_file).convert("RGB")

    # Display Workspace Layout if Image is uploaded
    if uploaded_image is not None:
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.image(uploaded_image, caption="Uploaded Source Viewport", use_container_width=True)
            
        with col2:
            st.subheader("🤖 Visual Processing System")
            generation_type = st.radio("Generation Mode", ["Single Definitive", "Dual Modes (Creative & Deterministic)"])
            
            if st.button("✨ Generate Vision Caption", type="primary"):
                with st.spinner("Analyzing image features and running transformer inference..."):
                    start_time = time.time()
                    
                    strategy = "both" if generation_type == "Dual Modes (Creative & Deterministic)" else "conditional"
                    captions = generate_image_caption(uploaded_image, strategy=strategy)
                    
                    latency = time.time() - start_time
                    
                    st.success(f"Inference Completed in {latency:.2f} seconds!")
                    
                    # Store Results
                    for cap in captions:
                        st.session_state.history.append({"caption": cap, "time": time.strftime("%H:%M:%S")})
                        
                        # Interactive Output Cards
                        st.markdown(f"<div class='caption-box'>📢 <b>Generated Descriptor:</b><br>{cap}</div>", unsafe_allow_html=True)
                        
                        # Advanced Feature: Audio Narration
                        audio_file = text_to_speech(cap)
                        if audio_file:
                            st.audio(audio_file, format="audio/mp3")
                        
                        # Advanced Feature: Cross-lingual Translations
                        with st.expander("🌐 Translate to Local Languages (India)"):
                            hi_cap = translate_caption(cap, 'hi')
                            te_cap = translate_caption(cap, 'te')
                            st.markdown(f"**🇮🇳 Hindi Translation:** {hi_cap}")
                            st.markdown(f"**🇮🇳 Telugu Translation:** {te_cap}")
                            
                        # Advanced Feature: Native File Export Downloads
                        st.download_button(
                            label="📥 Download Caption (.txt)",
                            data=f"VisionAI System Caption:\n{cap}\nGenerated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                            file_name="visionai_caption.txt",
                            mime="text/plain"
                        )

# --- 5. LOG HISTORY COMPONENT ---
elif app_mode == "Caption History":
    st.title("⏳ Session Archives")
    if len(st.session_state.history) == 0:
        st.info("No logs present yet. Navigate to the Dashboard to initiate execution.")
    else:
        st.table(st.session_state.history)
        if st.button("Clear Logs"):
            st.session_state.history = []
            st.rerun()

# --- 6. ARCHITECTURE DOCUMENTATION ---
elif app_mode == "Technical Documentation":
    st.title("📘 Architecture Explanation")
    st.markdown("""
    ### How Salesforce BLIP Works
    **BLIP (Bootstrapping Language-Image Pre-training)** is a unified vision-language framework that uses an encoder-decoder setup:
    1. **Vision Transformer (ViT):** Cuts up the uploaded image into fixed-size patches, flattens them, and embeds them into dense feature vectors.
    2. **Multimodal Mixture of Encoder-Decoder (Med):** Mixes these vision patterns with language structures to create clean text descriptions.
    """)

# --- 7. FOOTER METRICS ---
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 0.8rem; color:#888;'>VisionAI 2026 Core | Developed under Professional Portfolio Architecture Specs</p>", unsafe_allow_html=True)