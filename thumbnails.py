import os
import pandas as pd
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

IDS_CSV = r"C:\ids\downloaded_december_3_id.csv"
SAVE_DIR = r"C:\thumbnail_folder\thumbnails_december_3"
os.makedirs(SAVE_DIR, exist_ok=True)

# Read video IDs
df_ids = pd.read_csv(IDS_CSV, encoding='latin-1')
video_ids = df_ids['video_ids'].dropna().astype(str).str.strip().tolist()

headers = {
    "User-Agent": "Mozilla/5.0"
}

def download_thumbnail(vid):
    save_path = os.path.join(SAVE_DIR, f"{vid}.jpg")

    for res in ["maxresdefault.jpg", "hqdefault.jpg"]:
        url = f"https://img.youtube.com/vi/{vid}/{res}"
        try:
            resp = requests.get(url, headers=headers, timeout=5)
            if resp.status_code == 200 and resp.content[:10]:
                with open(save_path, "wb") as f:
                    f.write(resp.content)
                return f"Downloaded {vid} ({res})"
        except:
            pass
        
        time.sleep(0.1)  

    return f"Failed {vid}"

with ThreadPoolExecutor(max_workers=25) as executor:
    futures = [executor.submit(download_thumbnail, vid) for vid in video_ids]
    for f in as_completed(futures):
        print(f.result())
