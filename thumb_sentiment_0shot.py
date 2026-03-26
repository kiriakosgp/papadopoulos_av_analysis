from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os
import pandas as pd

DATASET_DIR = "dataset"
THUMBNAIL_NAME = "thumbnail.jpg"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

candidate_labels = [
    "positive, happy, YouTube thumbnail",
    "neutral YouTube thumbnail, no particular sentiment",
    "negative, angry or sad YouTube thumbnail"
]

results = []

for video_id in os.listdir(DATASET_DIR):
    video_dir = os.path.join(DATASET_DIR, video_id)

    if not os.path.isdir(video_dir):
        continue

    thumbnail_path = os.path.join(video_dir, THUMBNAIL_NAME)

    # Skip if thumbnail does not exist
    if not os.path.exists(thumbnail_path):
        continue

    try:
        image = Image.open(thumbnail_path).convert("RGB")

        inputs = processor(
            text=candidate_labels,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(device)

        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)

        score, idx = probs[0].max(0)

        results.append({
            "video_id": video_id,
            "thumbnail": THUMBNAIL_NAME,
            "label": candidate_labels[idx],
            "score": score.item()
        })

    except Exception as e:
        print(f"Failed processing {video_id}: {e}")

# Save results
output_csv = r"C:\thumbnail_folder\thumb_sentiment.csv"
pd.DataFrame(results).to_csv(output_csv, index=False)

print(f"Done — processed {len(results)} thumbnails")
