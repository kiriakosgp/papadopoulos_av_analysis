import os
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

DATASET_DIR = "dataset"
INFO_NAME = "info.json"

SAVE_DIR = "comments_agd_embeddings"
os.makedirs(SAVE_DIR, exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")


def process_comment_embeddings(dataset_dir):
    video_ids = [
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ]

    for video_id in tqdm(video_ids, desc="Processing aggregated comment embeddings"):
        info_path = os.path.join(dataset_dir, video_id, INFO_NAME)

        if not os.path.exists(info_path):
            continue

        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)

        comments = info.get("comments", [])
        comments = [c.strip() for c in comments if isinstance(c, str) and c.strip()]

        if len(comments) == 0:
            continue

        # Encode all comments
        embeddings = model.encode(
            comments,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        # Aggregate by mean
        agg_embedding = embeddings.mean(axis=0)

        # Optional: normalize the aggregated embedding
        agg_embedding /= np.linalg.norm(agg_embedding)

        save_path = os.path.join(SAVE_DIR, f"{video_id}.npy")
        np.save(save_path, agg_embedding)

    print(f"Finished! Aggregated comment embeddings saved to: {SAVE_DIR}")


process_comment_embeddings(DATASET_DIR)
