import os 
import json
import shutil
from pathlib import Path
import csv

 
original_json = r"C:\Users\admin\OneDrive\Υπολογιστής\youtube_data\data\dataset_december.json"
audio_dir = r"C:\audio"
transcripts_json = r"C:\transcripts\dataset_december_fixed.json"
thumbnails_dir = r"C:\thumbnail_folder\thumbnails_december"

output = "dataset"
os.makedirs(output, exist_ok=True)

def load_transcripts(json_path):
    
    transcripts = {}
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for item in data:
            vid = item.get("video_id")
            text = item.get("text", "")
            if vid:
                transcripts[vid] = text
    return transcripts

TRANSCRIPTS = load_transcripts(transcripts_json)


#find audio file 
def get_audio_path(vid):
    for ext in ["mp3", "m4a"]:
        path = Path(audio_dir) / f"{vid}.{ext}"
        if path.exists():
            return path, ext
    return None, None


def build():
    with open(original_json, "r", encoding="utf-8") as f:
        videos = json.load(f)
    
    manifest =[]

    for item in videos:
        vid = item["video_id"]
        folder = Path(output) / vid
        folder.mkdir(parents=True, exist_ok=True)

        #audio
        audio_path, ext = get_audio_path(vid)
        if audio_path:
            shutil.copy(audio_path, folder / f"audio.{ext}")
            item["audio_path"] = f"audio.{ext}"
        else:
            item["audio_path"] = None

        #thumbnail
        thumb = Path(thumbnails_dir) / f"{vid}.jpg"
        if thumb.exists():
            shutil.copy(thumb, folder / "thumbnail.jpg")
            item["thumbnail_path"] = "thumbnail.jpg"
        else:
            item["thumbnail_path"] = None

        #transcript
        transcript_text = TRANSCRIPTS.get(vid)
        if transcript_text:
            transcript_file = folder / "transcript.txt"
            with open(transcript_file, "w", encoding="utf-8") as f:
                f.write(transcript_text)
            item["transcript_path"] = "transcript.txt"
        else:
            item["transcript_path"] = None


        # write per-video info.json
        with open(folder / "info.json", "w", encoding="utf-8") as f:
            json.dump(item, f, indent=4, ensure_ascii=False)

        manifest.append({
            "video_id": vid,
            "path": str(folder)
        })

    

if __name__ == "__main__":
    build()