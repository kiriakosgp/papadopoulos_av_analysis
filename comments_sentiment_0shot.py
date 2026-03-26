from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import pandas as pd
import os 
import torch
import json

DATASET_DIR = "dataset"
INFO_NAME = "info.json"
MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()


labels = ["negative", "neutral", "positive"]

results = []

for video_id in os.listdir(DATASET_DIR):
    video_dir = os.path.join(DATASET_DIR, video_id)

    if not os.path.isdir(video_dir):
        continue

    info_path = os.path.join(video_dir, INFO_NAME)

    if not os.path.exists(info_path):
        continue

    try:
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)

        comments = info.get("comments", [])

        if not comments:
            continue

        # Combine all comments into one text
        text = " ".join(comments)

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)

        score, idx = probs[0].max(dim=0)

        results.append({
            "video_id": video_id,
            "label": labels[idx.item()],
            "score": score.item()
        })

    except Exception as e:
        print(f"Failed processing {video_id}: {e}")

output_csv = r"C:\transcripts\comments_sentiment_test.csv"
pd.DataFrame(results).to_csv(output_csv, index=False)

print(f"Done — processed {len(results)} transcripts")