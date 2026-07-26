# app_speaker_diarization.py
# --------------------------------
# Real-time meeting summarizer with speaker diarization & color-coded transcripts

import streamlit as st
import whisper
from transformers import BartTokenizer, BartForConditionalGeneration
import torch
import numpy as np
import sounddevice as sd
import queue
import threading
import re
from pyannote.audio import Pipeline
import random

# --------------------------
# 1. App Settings
# --------------------------
st.title("📝 Real-Time Meeting Summarizer with Speaker Diarization")
st.write("Live transcripts, summaries, action items with color-coded speakers.")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = "./bart_finetuned_medical"
CHUNK_DURATION = 10  # seconds
SAMPLE_RATE = 16000

# --------------------------
# 2. Load Models
# --------------------------
st.text("Loading models...")
whisper_model = whisper.load_model("base", device=DEVICE)
bart_tokenizer = BartTokenizer.from_pretrained(MODEL_DIR)
bart_model = BartForConditionalGeneration.from_pretrained(MODEL_DIR).to(DEVICE)

# Pretrained pyannote speaker diarization pipeline (requires Hugging Face token)
# Make sure you have set HF_HOME or environment variable with your token
st.text("Loading speaker diarization pipeline...")
diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=True)

# --------------------------
# 3. Helper Functions
# --------------------------
def generate_summary(text, max_input_length=512, max_target_length=50):
    inputs = bart_tokenizer(text, return_tensors="pt", truncation=True, max_length=max_input_length)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    summary_ids = bart_model.generate(
        inputs["input_ids"],
        max_length=max_target_length,
        num_beams=4,
        early_stopping=True
    )
    return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def extract_action_items(text):
    sentences = re.split(r'[.?!]\s+', text)
    task_keywords = ["assign", "complete", "submit", "review", "prepare", "schedule", "follow up", "update"]
    return [s for s in sentences if any(k in s.lower() for k in task_keywords)]

# --------------------------
# 4. Audio Queue
# --------------------------
audio_q = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        st.warning(str(status))
    audio_q.put(indata.copy())

# --------------------------
# 5. Processing Thread
# --------------------------
def process_audio():
    buffer = np.zeros((0,))
    while True:
        chunk = audio_q.get()
        buffer = np.concatenate((buffer, chunk.flatten()))
        if len(buffer) >= SAMPLE_RATE * CHUNK_DURATION:
            audio_chunk = buffer[:SAMPLE_RATE * CHUNK_DURATION]
            buffer = buffer[SAMPLE_RATE * CHUNK_DURATION:]
            
            # Save chunk temporarily
            tmp_wav = "temp_chunk.wav"
            from scipy.io.wavfile import write
            write(tmp_wav, SAMPLE_RATE, (audio_chunk * 32767).astype(np.int16))
            
            # 1️⃣ Speaker diarization
            diarization = diarization_pipeline(tmp_wav)
            
            # 2️⃣ Whisper transcription
            result = whisper_model.transcribe(tmp_wav, fp16=torch.cuda.is_available())
            transcript_text = result["text"].strip()
            
            if transcript_text:
                # Assign colors to speakers
                speaker_colors = {}
                for turn in diarization.itertracks(yield_label=True):
                    speaker_colors[turn[2]] = f"#{random.randint(0, 0xFFFFFF):06x}"
                
                # Build speaker-colored transcript
                speaker_texts = []
                for turn in diarization.itertracks(yield_label=True):
                    start, end, speaker = turn[0].start, turn[0].end, turn[2]
                    # For simplicity, assign full transcript to each speaker proportionally
                    speaker_texts.append((speaker, transcript_text))
                
                # Store in Streamlit session
                st.session_state["speaker_transcripts"].extend(speaker_texts)
                
                # 3️⃣ Generate summary + action items
                summary = generate_summary(transcript_text)
                st.session_state["summaries"].append(summary)
                st.session_state["action_items"].append(extract_action_items(summary))

# --------------------------
# 6. Streamlit UI
# --------------------------
if "speaker_transcripts" not in st.session_state:
    st.session_state["speaker_transcripts"] = []
if "summaries" not in st.session_state:
    st.session_state["summaries"] = []
if "action_items" not in st.session_state:
    st.session_state["action_items"] = []

if st.button("Start Real-Time Summarizer with Diarization"):
    st.info("Listening to microphone...")
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback)
    stream.start()
    threading.Thread(target=process_audio, daemon=True).start()

# --------------------------
# 7. Display Live Output
# --------------------------
st.subheader("🎤 Speaker-wise Transcripts")
for speaker, text in st.session_state["speaker_transcripts"]:
    color = f"#{hash(speaker) & 0xFFFFFF:06x}"  # consistent color per speaker
    st.markdown(f"<span style='color:{color}'><b>{speaker}:</b> {text}</span>", unsafe_allow_html=True)

st.subheader("📝 Summaries")
for s in st.session_state["summaries"]:
    st.write(s)

st.subheader("✅ Action Items")
for acts in st.session_state["action_items"]:
    for act in acts:
        st.write("-", act)
