import os
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import numpy as np

DATASET_DIR = "dataset"
SAVE_DIR = "image_embeddings"
IMAGE_NAME = "thumbnail.jpg"

os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()


def process_image_embeddings(dataset_dir):
    video_ids = [
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ]

    for video_id in tqdm(video_ids, desc="Processing image embeddings"):
        img_path = os.path.join(dataset_dir, video_id, IMAGE_NAME)

        if not os.path.exists(img_path):
            continue

        image = Image.open(img_path).convert("RGB")

        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.get_image_features(**inputs)

        embedding = outputs.squeeze(0).cpu().numpy()

        # Normalize to unit length
        embedding /= np.linalg.norm(embedding)

        np.save(os.path.join(SAVE_DIR, f"{video_id}.npy"), embedding)

    print(f"Finished! Image embeddings saved to: {SAVE_DIR}")


process_image_embeddings(DATASET_DIR)
