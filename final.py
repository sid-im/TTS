import streamlit as st
import tempfile
import os
import torch
import warnings
warnings.filterwarnings("ignore")

# PyTorch 2.6 compatibility fix
import torch.serialization
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig

# Safe globals for PyTorch 2.6
with torch.serialization.safe_globals([XttsConfig, XttsAudioConfig]):
    from TTS.api import TTS

# Page config
st.set_page_config(
    page_title="AI Voice Cloner",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for animations and style
st.markdown("""
    <style>
    .title {
        font-size: 36px;
        font-weight: bold;
        background: linear-gradient(90deg, #ff7e5f, #feb47b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: fadeIn 2s ease-in-out;
    }
    .subtitle {
        font-size: 18px;
        color: #555;
        margin-bottom: 20px;
        animation: slideUp 1.5s ease;
    }
    @keyframes fadeIn { from {opacity: 0;} to {opacity: 1;} }
    @keyframes slideUp { from {transform: translateY(20px); opacity: 0;} to {transform: translateY(0); opacity: 1;} }
    .stButton>button {
        border-radius: 12px;
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        background: linear-gradient(135deg, #2575fc 0%, #6a11cb 100%);
    }
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_xtts_model():
    st.info("🔄 Loading XTTS model...")
    try:
        with torch.serialization.safe_globals([XttsConfig, XttsAudioConfig]):
            tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        st.success("✅ Model loaded successfully!")
        return tts
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None

# Preprocess audio
def preprocess_audio(audio_path, output_path, max_duration=10):
    try:
        import librosa, soundfile as sf
        audio, sr = librosa.load(audio_path, sr=22050)
        max_samples = min(len(audio), int(max_duration * sr))
        sf.write(output_path, audio[:max_samples], sr)
        return True
    except:
        import shutil
        shutil.copy2(audio_path, output_path)
        return False

# Generate cloned voice
def generate_voice_clone(text, reference_audio_path, language="en"):
    try:
        tts = st.session_state.model
        if not tts:
            return None
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tts.tts_to_file(
                text=text,
                speaker_wav=reference_audio_path,
                language=language,
                file_path=tmp_file.name
            )
            return tmp_file.name
    except Exception as e:
        st.error(f"❌ Voice cloning failed: {str(e)}")
        return None

# Init model
if "model" not in st.session_state:
    st.session_state.model = load_xtts_model()

# ================= UI =================
st.markdown("<div class='title'>🎙️ AI Voice Cloner</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Clone any voice and make it speak any text in any language</div>", unsafe_allow_html=True)

if st.session_state.model is None:
    st.stop()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    language = st.selectbox(
        "Select language:",
        ["en", "hi", "kn", "ta", "te", "fr", "de", "es", "zh", "ar"],
        index=0
    )
    st.caption("Supports multilingual speech synthesis")

# Upload reference audio
st.subheader("1️⃣ Upload Reference Audio")
reference_audio = st.file_uploader("Upload voice sample (WAV/MP3)", type=["wav", "mp3"])

# Text input
st.subheader("2️⃣ Enter Text")
text = st.text_area("Enter text to synthesize:", placeholder="Type your sentence here...")

# Generate button
if st.button("✨ Generate Voice Clone", use_container_width=True):
    if not text.strip():
        st.warning("⚠️ Please enter some text.")
    elif not reference_audio:
        st.warning("⚠️ Please upload a reference audio file.")
    else:
        with st.spinner("🎨 Cloning voice... this may take a minute"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as ref_file:
                ref_file.write(reference_audio.getvalue())
                ref_path = ref_file.name
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as short_ref:
                preprocess_audio(ref_path, short_ref.name, max_duration=15)
                short_ref_path = short_ref.name

            output_file = generate_voice_clone(text, short_ref_path, language)
            if output_file:
                st.success("✅ Voice cloning complete!")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("🎧 **Reference Audio**")
                    st.audio(short_ref_path)
                with col2:
                    st.markdown("🔊 **Generated Clone**")
                    st.audio(output_file)

                with open(output_file, "rb") as f:
                    st.download_button("📥 Download", f, file_name="voice_clone.wav")

# Footer
st.markdown("---")
st.markdown("💡 *Tip: Use 5-10s clear reference audio for best results.*")
