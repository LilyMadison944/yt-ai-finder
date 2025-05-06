from fastapi import FastAPI, Request
from pydantic import BaseModel
import googleapiclient.discovery
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import nltk
from nltk.tokenize import sent_tokenize
import uuid
import re
import numpy as np
import logging
import os

# Setup
nltk.download('punkt', quiet=True)
app = FastAPI()

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# YouTube API
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY", "")
youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

# NLP models
relevance_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
relevance_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=-1)

# Request model
class PromptInput(BaseModel):
    prompt: str

def generate_search_queries(prompt):
    prompt = prompt.lower().strip()
    queries = [prompt]
    if "how to" in prompt:
        queries.append(prompt + " tutorial")
    else:
        queries.append(prompt + " explained")
    topic = re.sub(r"how to|what is", "", prompt).strip()
    queries.append(topic + " guide")
    return queries

def search_youtube(query, max_results=3):
    try:
        request = youtube.search().list(
            part="snippet",
            q=query,
            type="video",
            maxResults=max_results
        )
        response = request.execute()
        return [
            {
                "video_id": item["id"]["videoId"],
                "title": item["snippet"]["title"],
                "channel": item["snippet"]["channelTitle"]
            }
            for item in response.get("items", [])
        ]
    except Exception as e:
        logger.error(f"YouTube API error: {e}")
        return []

def fetch_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-US', 'en-GB'])
        return " ".join([entry["text"] for entry in transcript])
    except Exception as e:
        logger.warning(f"Transcript error: {e}")
        return None

def analyze_relevance(transcript, prompt):
    if not transcript:
        return 0.0
    inputs = relevance_tokenizer(prompt, transcript, return_tensors="pt", truncation=True, max_length=256)
    outputs = relevance_model(**inputs)
    score = outputs.logits.softmax(dim=1)[0][1].item()
    return score

def summarize_transcript(transcript):
    try:
        sentences = sent_tokenize(transcript)
        text = " ".join(sentences[:5])
        summary = summarizer(text, do_sample=False)[0]["summary_text"]
        return summary
    except Exception as e:
        logger.warning(f"Summarization error: {e}")
        return "Summary unavailable."

def fetch_comments_and_likes(video_id):
    try:
        video_request = youtube.videos().list(part="statistics", id=video_id)
        video_response = video_request.execute()
        likes = int(video_response["items"][0]["statistics"].get("likeCount", 0))

        comment_request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=3,
            textFormat="plainText"
        )
        comment_response = comment_request.execute()

        comments = [item["snippet"]["topLevelComment"]["snippet"]["textDisplay"] for item in comment_response.get("items", [])]
        sentiments = sentiment_analyzer(comments) if comments else []
        sentiment_scores = [1 if s["label"] == "POSITIVE" else -1 for s in sentiments]
        avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0

        return {
            "total_likes": likes,
            "top_comments": comments,
            "sentiment_score": avg_sentiment
        }
    except Exception as e:
        logger.error(f"Error fetching comments/likes: {e}")
        return {"total_likes": 0, "top_comments": [], "sentiment_score": 0.0}

@app.get("/")
def root():
    return {"message": "Use POST /find-videos with {'prompt': 'your question'}"}

@app.post("/find-videos")
def find_videos(data: PromptInput):
    prompt = data.prompt
    queries = generate_search_queries(prompt)
    all_videos = []

    for query in queries:
        videos = search_youtube(query)
        for video in videos:
            video["search_query"] = query
            all_videos.append(video)

    results = []
    for video in all_videos:
        video_id = video["video_id"]
        transcript = fetch_transcript(video_id)
        relevance_score = analyze_relevance(transcript, prompt)
        summary = summarize_transcript(transcript)
        comments_likes = fetch_comments_and_likes(video_id)

        results.append({
            "video_id": video_id,
            "title": video["title"],
            "channel": video["channel"],
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "relevance_score": relevance_score,
            "summary": summary,
            "total_likes": comments_likes["total_likes"],
            "top_comments": comments_likes["top_comments"],
            "comment_sentiment": comments_likes["sentiment_score"],
            "search_query": video["search_query"]
        })

    results = sorted(results, key=lambda x: x["relevance_score"], reverse=True)
    return {"results": results}
