import os
import librosa
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
import numpy as np 

DATASET_DIR = "dataset"
OUTPUT_CSV = r"C:\transcripts\audio_sentiment_0shot.csv"

TARGET_SR = 16000
MAX_AUDIO_SECONDS = 30
CHUNK_SECONDS = 10
MAX_VIDEOS = None

MODEL_NAME = "superb/wav2vec2-base-superb-er"

EMOTION_TO_SENTIMENT = {
    "happy": "positive",
    "excited": "positive",
    "neutral": "neutral",
    "sad": "negative",
    "angry": "negative",
    "fearful": "negative",
    "disgust": "negative",
    "surprised": "neutral"
}

torch.set_num_threads(1
                      )
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()


def find_audio_file(folder_path):
    for fname in os.listdir(folder_path):
        if fname.lower().endswith((".mp3", ".m4a", ".wav")):
            return os.path.join(folder_path, fname)
    return None

def chunk_audio(audio, sr, chunk_seconds, max_chunks=3):
    total_samples = len(audio)
    chunk_size = int(sr*chunk_seconds)
    
    if total_samples < chunk_size:
        return []
    
    max_possible_chunks = total_samples // chunk_size

    
    num_chunks = min(max_possible_chunks, max_chunks)

    
    starts = np.linspace(0,
                         total_samples - chunk_size,
                         num_chunks,
                         dtype=int)

    return [audio[s:s + chunk_size] for s in starts]

results = []

video_ids = [
    d for d in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, d))
]

if MAX_VIDEOS is not None: 
    video_ids = video_ids[:MAX_VIDEOS]

for video_id in tqdm(video_ids, desc="Audio sentiment baseline"):
    video_path = os.path.join(DATASET_DIR, video_id)
    audio_path = find_audio_file(video_path)

    if audio_path is None:
        continue

    try:
        audio, sr = librosa.load(audio_path, sr=TARGET_SR)
        chunks = chunk_audio(audio, sr, CHUNK_SECONDS, MAX_AUDIO_SECONDS)

        if len(chunks) == 0:
            continue

        all_probs = []

        for chunk in chunks:
            inputs = feature_extractor(
                chunk,
                sampling_rate=TARGET_SR,
                return_tensors="pt",
                padding=True
            )

            with torch.no_grad():
                logits = model(**inputs).logits

            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            all_probs.append(probs)

        mean_probs = np.mean(all_probs, axis=0)
        pred_id = int(np.argmax(mean_probs))

        emotion = model.config.id2label[pred_id]
        sentiment = EMOTION_TO_SENTIMENT.get(emotion, "neutral")
        score = float(mean_probs[pred_id])

        results.append({
            "video_id": video_id,
            "label": sentiment,
            "score": score
        })

    except Exception as e:
        print(f"Skipping {video_id}: {e}")

df = pd.DataFrame(results, columns=["video_id", "label", "score"])
df.to_csv(OUTPUT_CSV, index=False)

print(f"\nSaved {len(df)} rows to {OUTPUT_CSV}")