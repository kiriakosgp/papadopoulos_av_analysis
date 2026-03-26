import os
import pandas as pd
import whisper

AUDIO_FOLDER = r"C:\audio\audio_december_3"  
OUTPUT_CSV = r"C:\transcripts\transcripts_december_3.csv"
WHISPER_MODEL = "base"

audio_extensions = (".mp3", ".wav", ".m4a", ".aac", ".ogg")

audio_files = [
    os.path.join(AUDIO_FOLDER, f)
    for f in os.listdir(AUDIO_FOLDER)
    if f.lower().endswith(audio_extensions)
]

if not audio_files:
    print(f"No audio files found in {AUDIO_FOLDER}. Exiting.")
    exit()

print(f"Loading Whisper model '{WHISPER_MODEL}'...")
model = whisper.load_model(WHISPER_MODEL)

transcripts = []

for audio_path in audio_files:
    filename = os.path.basename(audio_path)
    video_id = os.path.splitext(filename)[0]   
    print(f"\nTranscribing ({video_id}): {filename}")
    try:
        result = model.transcribe(audio_path, fp16=False)
        transcripts.append({
            "video_id": video_id,
            "file": filename,
            "text": result["text"].strip()
        })
    except Exception as e:
        print(f"Failed to transcribe {audio_path}: {e}")

transcripts_df = pd.DataFrame(transcripts)
transcripts_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
print(f"Transcripts saved to: {OUTPUT_CSV}")
