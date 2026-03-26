import re
import nltk
from nltk.corpus import stopwords
import json
import pandas as pd
from langdetect import detect, LangDetectException 
from deep_translator import GoogleTranslator


with open(r"C:\Users\admin\OneDrive\Υπολογιστής\youtube_data/data/shorts_dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

# (title + description + captions + comments)
texts = []
for item in dataset:
    combined_text = " ".join([
        item["title"], 
        item["description"], 
        item["captions"], 
        " ".join(item["comments"])
    ])
    texts.append(combined_text)

nltk.download("punkt")
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# preprocessing
def preprocess_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    try:
        lang = detect(text)
    except LangDetectException:
        lang = "unknown"
    allowed_languages = {"es", "fr", "de", "pt"}
    if lang != "en" and lang in allowed_languages:
        try:
            text = GoogleTranslator(source="auto", target="en").translate(text)
        except:
            pass  
    text = text.lower()
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text


texts_clean = [preprocess_text(t) for t in texts]

# Save to CSV
df = pd.DataFrame({"cleaned_text": texts_clean})
df.to_csv(r"C:\Users\admin\OneDrive\Υπολογιστής\youtube_data/data/preprocessed_shorts.csv", index=False, encoding="utf-8")
print("Preprocessed data saved to CSV!")
