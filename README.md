# 🎙️ Real-Time Meeting Summarizer & Action-Item Extractor

An end-to-end AI-powered meeting intelligence system that converts spoken conversations into structured insights in real time. The project combines **Automatic Speech Recognition (ASR)**, **Speaker Diarization**, and a **fine-tuned BART Transformer** to generate concise meeting summaries and automatically extract actionable tasks.

---

## 📌 Project Overview

Meetings often contain valuable discussions, decisions, and action items that are difficult to capture manually.

This project automates the entire workflow by:

1. Converting speech into text using **OpenAI Whisper**
2. Identifying different speakers through **Speaker Diarization**
3. Cleaning and preprocessing transcripts
4. Generating concise meeting summaries using a fine-tuned **BART** model
5. Extracting important action items
6. Evaluating summary quality using multiple NLP metrics

The system is designed to reduce manual note-taking while improving productivity and documentation accuracy.

---

# ✨ Features

- 🎤 Real-time speech-to-text transcription
- 👥 Speaker diarization
- 📝 AI-generated meeting summaries
- ✅ Automatic action-item extraction
- 🚀 Fine-tuned BART summarization model
- 📊 Multi-metric evaluation framework
- ⚡ Mixed-precision training for faster learning
- 🔍 Beam Search decoding for improved text generation
- 📁 Modular pipeline for future expansion

---

# 🏗 System Architecture

```
                Audio Stream
                      │
                      ▼
          Whisper Automatic Speech Recognition
                      │
                      ▼
             Speaker Diarization
                      │
                      ▼
           Transcript Preprocessing
                      │
                      ▼
         Fine-Tuned BART Summarizer
             │                 │
             ▼                 ▼
      Meeting Summary     Action Items
             │
             ▼
     Evaluation & Metrics
```

---

# 🛠 Tech Stack

### Languages

- Python

### Deep Learning

- PyTorch

### NLP

- HuggingFace Transformers
- BART
- Tokenizers

### Speech Processing

- Whisper ASR
- Speaker Diarization

### Evaluation

- ROUGE-1
- ROUGE-2
- ROUGE-L
- Precision
- Recall
- F1 Score

---

# 📂 Project Structure

```
meeting-summarizer/
│
├── data/
│   ├── raw_audio/
│   ├── transcripts/
│   └── processed/
│
├── models/
│   ├── whisper/
│   └── bart/
│
├── training/
│   ├── train.py
│   ├── tokenizer.py
│   └── dataset.py
│
├── inference/
│   ├── summarize.py
│   ├── action_items.py
│   └── diarization.py
│
├── evaluation/
│   ├── rouge.py
│   ├── metrics.py
│   └── evaluate.py
│
├── utils/
│
├── requirements.txt
│
└── README.md
```

---

# ⚙️ Model Pipeline

## Step 1 — Audio Input

Meeting audio is collected in WAV or MP3 format.

↓

## Step 2 — Whisper ASR

Whisper converts speech into text.

↓

## Step 3 — Speaker Diarization

Different speakers are identified and labeled.

Example:

```
Speaker 1:
Let's deploy the model tomorrow.

Speaker 2:
I'll prepare the documentation.
```

↓

## Step 4 — Text Preprocessing

- Remove noise
- Normalize text
- Sentence segmentation
- Tokenization

↓

## Step 5 — Fine-Tuned BART

The cleaned transcript is passed into the fine-tuned BART model.

Output:

```
Summary:

The team discussed deployment planning.
Documentation will be completed before deployment.
```

↓

## Step 6 — Action Item Extraction

Example:

```
✔ Prepare documentation
✔ Deploy model tomorrow
✔ Schedule client demo
```

---

# 🧠 Training Pipeline

The model training process includes:

- Dataset preparation
- Transcript cleaning
- Tokenization
- Data batching
- Mixed Precision (FP16) Training
- Beam Search Decoding
- Checkpoint Saving
- Validation
- Metric Evaluation

---

# 📊 Evaluation Metrics

The summarization model is evaluated using:

| Metric | Purpose |
|----------|----------|
| ROUGE-1 | Unigram overlap |
| ROUGE-2 | Bigram overlap |
| ROUGE-L | Longest common subsequence |
| Precision | Correct generated information |
| Recall | Information coverage |
| F1 Score | Overall summarization quality |

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/kirito2789/meeting-summarizer.git

cd meeting-summarizer
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

### Train the Model

```bash
python training/train.py
```

### Generate Summary

```bash
python inference/summarize.py
```

### Extract Action Items

```bash
python inference/action_items.py
```

### Evaluate Performance

```bash
python evaluation/evaluate.py
```

---

# 📈 Example Output

## Input Transcript

```
Speaker 1:
We should deploy the API next Monday.

Speaker 2:
I'll complete testing before Friday.

Speaker 3:
Let's also update the project documentation.
```

---

## Generated Summary

```
The meeting focused on deployment planning.
Testing will be completed before Friday,
followed by deployment next Monday.
Project documentation will also be updated.
```

---

## Extracted Action Items

```
✔ Complete testing before Friday

✔ Deploy API next Monday

✔ Update project documentation
```

---

# 📚 Future Improvements

The project can be extended with several advanced capabilities:

### 🔹 Real-Time Live Meetings

Instead of processing recorded audio, integrate streaming audio for live meeting summarization.

---

### 🔹 LLM Integration

Replace or enhance BART with:

- Llama 3
- Mistral
- GPT-based models
- FLAN-T5

to generate more human-like summaries.

---

### 🔹 Meeting Analytics Dashboard

Build a web dashboard showing:

- Speaker statistics
- Speaking time
- Topic distribution
- Meeting sentiment
- Action item tracking

using Streamlit or React.

---

### 🔹 Sentiment Analysis

Analyze:

- Positive discussions
- Negative discussions
- Agreement/Disagreement
- Meeting mood

---

### 🔹 Topic Detection

Automatically detect discussion topics such as:

- Budget
- Hiring
- Marketing
- Product Development

---

### 🔹 Multi-language Support

Enable multilingual transcription and summarization using Whisper's multilingual capabilities.

---

### 🔹 Calendar Integration

Automatically send summaries and action items to:

- Google Calendar
- Outlook
- Slack
- Microsoft Teams

---

### 🔹 RAG-based Knowledge Retrieval

Store meeting transcripts in a vector database and allow users to ask questions such as:

> "What decision was made regarding deployment?"

using Retrieval-Augmented Generation (RAG).

---

### 🔹 Email Automation

Automatically email meeting summaries to participants after every meeting.

---

# 🎯 Applications

- Business Meetings
- Team Standups
- Online Classes
- Customer Support Calls
- Interviews
- Medical Consultations
- Legal Proceedings
- Project Management
- Remote Collaboration

---

# 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a new feature branch
3. Commit your changes
4. Push to your branch
5. Open a Pull Request

---

# 📜 License

This project is released under the MIT License.

---

# 👨‍💻 Author

**Your Name**

AI | Deep Learning | NLP | Speech Processing

Feel free to connect and contribute to improve this project.

---
