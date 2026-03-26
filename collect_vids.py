from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
import json
from tqdm import tqdm
import time



API_KEY = 'AIzaSyCZsq3Ih_KRcMe2MSGuamJfz6YIdT5JYrs'
youtube = build("youtube", "v3", developerKey=API_KEY)

# Parameters
SEARCH_QUERIES = ["#newsupdate", "#latestnews", "#dailynews", "#localnews"]
MAX_RESULTS_PER_QUERY = 200
MAX_COMMENTS = 200  

def search_shorts(query, max_videos=200):
    results = []
    next_page_token = None

    while len(results) < max_videos:
        req = youtube.search().list(
            q=query,
            part="snippet",
            maxResults=min(50, max_videos - len(results)),  
            type="video",
            videoDuration="short",
            pageToken=next_page_token
        )
        response = req.execute()
        results.extend(response.get("items", []))
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return results

def get_comments(video_id, max_comments=200):
    comments = []
    next_page_token = None

    while len(comments) < max_comments:
        try:
            req = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(100, max_comments - len(comments)),
                textFormat="plainText",
                pageToken=next_page_token
            )
            response = req.execute()
            comments.extend([item["snippet"]["topLevelComment"]["snippet"]["textDisplay"] 
                             for item in response.get("items", [])])
            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break
        except:
            break

    return comments


def get_captions(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([x["text"] for x in transcript])
    except (TranscriptsDisabled, Exception):
        return ""

def collect_data():
    dataset = {}
    
    for query in SEARCH_QUERIES:
        print(f"🔍 Collecting videos for query: {query}")
        response = search_shorts(query, max_videos=MAX_RESULTS_PER_QUERY)

        for item in tqdm(response):
            video_id = item["id"]["videoId"]
            if video_id in dataset:  # Deduplicate
                continue

            snippet = item["snippet"]
            title = snippet.get("title", "")
            description = snippet.get("description", "")
            channel = snippet.get("channelTitle", "")
            published = snippet.get("publishedAt", "")

            comments = get_comments(video_id, MAX_COMMENTS)
            captions = get_captions(video_id)

            dataset[video_id] = {
                "video_id": video_id,
                "title": title,
                "description": description,
                "channel": channel,
                "published": published,
                "captions": captions,
                "comments": comments,
                "query": query
            }

            
            time.sleep(1)  # avoid hitting quota too fast

    # Save results
    save_path = r"C:\Users\admin\OneDrive\Υπολογιστής\youtube_data\data\dataset_december_3.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(list(dataset.values()), f, ensure_ascii=False, indent=2)

    print(f"Data collection complete. Saved {len(dataset)} videos.")


if __name__ == "__main__":
    collect_data()



