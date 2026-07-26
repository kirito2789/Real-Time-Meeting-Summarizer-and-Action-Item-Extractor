# whisper_transcriber.py
# -----------------------
# Converts meeting audio into text transcript using OpenAI Whisper.

import whisper
import sys
import os

def transcribe_audio(file_path):
    model = whisper.load_model("small", download_root="~/.cache/whisper")  # tiny, base, small, medium, large
    print(f"Transcribing: {file_path}")

    result = model.transcribe(file_path, language="en")
    print("\n--- TRANSCRIPT ---\n")
    print(result["text"])

    # Save transcript
    transcript_path = os.path.splitext(file_path)[0] + "_transcript.txt"
    with open(transcript_path, "w") as f:
        f.write(result["text"])
    print(f"\nTranscript saved to {transcript_path}")
    return transcript_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python whisper_transcriber.py <audio_file.wav>")
        sys.exit(1)

    audio_file = sys.argv[1]
    transcribe_audio(audio_file)
