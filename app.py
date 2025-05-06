import os
import flask
from flask import Flask, request, jsonify
import googleapiclient.discovery
import googleapiclient.errors
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import nltk
from nltk.tokenize import sent_tokenize
import re
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Initialize Flask app
app = Flask(__name__)

# YouTube API setup
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY")
if not YOUTUBE_API_KEY:
    logger.error("YOUTUBE_API_KEY environment variable is not set.")
    raise ValueError("YOUTUBE_API_KEY is required.")
youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

# Initialize NLP models (force CPU)
device = -1  # -1 for CPU
relevance_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
relevance_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)
summarizer = pipeline(
    "summarization", model="facebook/bart-large-cnn", device=device, max_length=100, min_length=30
)
sentiment_analyzer = pipeline(
    "sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=device
)

def generate_search_queries(prompt):
    """Generate three YouTube search queries from the user prompt."""
    prompt = prompt.lower().strip()
    queries = []
    queries.append(prompt)
    if "how to" in prompt:
        queries.append(prompt + " tutorial")
    else:
        queries.append(prompt + " explained")
    topic = re.sub(r"how to|what is", "", prompt).strip()
    queries.append(topic + " guide")
    return queries

def search_youtube(query, max_results=3):
    """Search YouTube for videos matching the query."""
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
    except googleapiclient.errors.HttpError as e:
        logger.error(f"YouTube API error: {e}")
        return []

def fetch_transcript(video_id):
    """Fetch transcript for a YouTube video."""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-US', 'en-GB'])
        return " ".join([entry["text"] for entry in transcript])
    except Exception as e:
        logger.warning(f"Transcript error for {video_id}: {e}")
        return None

def analyze_relevance(transcript, prompt):
    """Analyze transcript relevance to the prompt using DistilBERT."""
    if not transcript:
        return 0.0
    inputs = relevance_tokenizer(
        prompt, transcript, return_tensors="pt", truncation=True, max_length=256
    )
    outputs = relevance_model(**inputs)
    score = outputs.logits.softmax(dim=1)[0][1].item()
    return score

def summarize_transcript(transcript):
    """Generate a summary of the transcript."""
    if not transcript:
        return "No transcript available."
    try:
        sentences = sent_tokenize(transcript)
        text = " ".join(sentences[:5])
        summary = summarizer(text, do_sample=False)[0]["summary_text"]
        return summary
    except Exception as e:
        logger.warning(f"Summarization error: {e}")
        return "Summary unavailable."

def fetch_comments_and_likes(video_id):
    """Fetch total likes, top 3 comments, and analyze comment sentiment."""
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
        
        comments = []
        comment_texts = []
        for item in comment_response.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)
            comment_texts.append(comment)
        
        sentiments = sentiment_analyzer(comment_texts) if comment_texts else []
        sentiment_scores = [1 if s["label"] == "POSITIVE" else -1 for s in sentiments]
        avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0
        
        return {
            "total_likes": likes,
            "top_comments": comments,
            "sentiment_score": avg_sentiment
        }
    except googleapiclient.errors.HttpError as e:
        logger.error(f"Comments/Likes error for {video_id}: {e}")
        return {
            "total_likes": 0,
            "top_comments": [],
            "sentiment_score": 0.0
        }

@app.route("/", methods=["GET", "POST"])
def root():
    """Handle requests to the root endpoint."""
    logger.warning(f"Received request to root endpoint: {request.method} {request.url}")
    return jsonify({
        "error": "Not Found",
        "message": "Use POST /find-videos with a JSON payload containing a 'prompt' field."
    }), 404

@app.route("/find-videos", methods=["POST"])
def find_videos():
    """API endpoint to process user prompt and return relevant videos."""
    try:
        logger.info(f"Received request to /find-videos with payload: {request.get_json()}")
        data = request.get_json()
        prompt = data.get("prompt", "")
        if not prompt:
            logger.warning("Prompt missing in request")
            return jsonify({"error": "Prompt is required"}), 400
        
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
        logger.info(f"Returning {len(results[:5])} videos for prompt: {prompt}")
        return jsonify({
            "prompt": prompt,
            "videos": results[:5]
        })
    except Exception as e:
        logger.error(f"Error in find-videos endpoint: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
