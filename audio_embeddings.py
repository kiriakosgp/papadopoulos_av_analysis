import os
import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model

DATASET_DIR = "dataset"
SAVE_DIR = "audio_embeddings"

AUDIO_EXTS = [".wav", ".mp3", ".m4a"]
TARGET_SR = 16000
CHUNK_SECONDS = 15
CHUNK_SAMPLES = TARGET_SR * CHUNK_SECONDS

os.makedirs(SAVE_DIR, exist_ok=True)

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
model.eval()


def find_audio_file(video_dir):
    for ext in AUDIO_EXTS:
        path = os.path.join(video_dir, f"audio{ext}")
        if os.path.exists(path):
            return path
    return None


def chunk_waveform(waveform, chunk_size):
    return [
        waveform[i : i + chunk_size]
        for i in range(0, waveform.shape[0], chunk_size)
        if waveform[i : i + chunk_size].shape[0] > chunk_size // 2
    ]


def process_audio_embeddings(dataset_dir):
    video_ids = [
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ]

    for video_id in tqdm(video_ids, desc="Processing audio embeddings"):
        video_dir = os.path.join(dataset_dir, video_id)
        audio_path = find_audio_file(video_dir)

        if audio_path is None:
            continue

        waveform, sample_rate = torchaudio.load(audio_path)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.squeeze(0)

        # Resample
        if sample_rate != TARGET_SR:
            waveform = torchaudio.transforms.Resample(sample_rate, TARGET_SR)(waveform)

        # Chunk audio
        chunks = chunk_waveform(waveform, CHUNK_SAMPLES)

        if len(chunks) == 0:
            continue

        chunk_embeddings = []

        for chunk in chunks:
            inputs = processor(
                chunk.numpy(),
                sampling_rate=TARGET_SR,
                return_tensors="pt"
            )

            with torch.no_grad():
                outputs = model(**inputs)

            hidden = outputs.last_hidden_state
            mask = inputs.attention_mask.unsqueeze(-1)

            emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
            chunk_embeddings.append(emb.squeeze(0).numpy())

        # Aggregate chunks
        audio_embedding = np.mean(chunk_embeddings, axis=0)

        # Normalize
        audio_embedding /= np.linalg.norm(audio_embedding)

        np.save(
            os.path.join(SAVE_DIR, f"{video_id}.npy"),
            audio_embedding
        )

    print(f"Finished! Audio embeddings saved to: {SAVE_DIR}")


process_audio_embeddings(DATASET_DIR)


