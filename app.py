import os
import flask
from flask import Flask, request, jsonify
import googleapiclient.discovery
import googleapiclient.errors
from youtube_transcript_api import YouTubeTranscriptApi
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import re
import numpy as np
import logging
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize Flask app
app = Flask(__name__)

# YouTube API setup
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY")
if not YOUTUBE_API_KEY:
    logger.error("YOUTUBE_API_KEY environment variable is not set.")
    raise ValueError("YOUTUBE_API_KEY is required.")
youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

# Simple sentiment analysis word lists
POSITIVE_WORDS = {'good', 'great', 'awesome', 'excellent', 'helpful', 'clear', 'amazing', 'love', 'fantastic'}
NEGATIVE_WORDS = {'bad', 'poor', 'terrible', 'awful', 'confusing', 'useless', 'horrible', 'hate', 'worst'}

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
    """Analyze transcript relevance using keyword overlap."""
    if not transcript:
        return 0.0
    prompt_words = set(word_tokenize(prompt.lower()))
    transcript_words = set(word_tokenize(transcript.lower()))
    common_words = prompt_words.intersection(transcript_words)
    return len(common_words) / max(len(prompt_words), 1)  # Normalize by prompt length

def summarize_transcript(transcript):
    """Generate a summary of at least 500 characters using frequency-based extractive summarization."""
    if not transcript:
        return "No transcript available."
    
    try:
        # Tokenize sentences
        sentences = sent_tokenize(transcript)
        if not sentences:
            return transcript[:500] + ("..." if len(transcript) > 500 else "") or "No transcript available."
        
        # Tokenize words and remove stopwords
        stop_words = set(stopwords.words('english'))
        word_freq = defaultdict(int)
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            for word in words:
                if word.isalnum() and word not in stop_words:
                    word_freq[word] += 1
        
        # Score sentences based on word frequency
        sentence_scores = []
        for sentence in sentences:
            score = sum(word_freq[word.lower()] for word in word_tokenize(sentence) if word.isalnum() and word.lower() not in stop_words)
            sentence_scores.append((sentence, score / max(len(word_tokenize(sentence)), 1)))
        
        # Sort sentences by score
        sorted_sentences = [sentence for sentence, _ in sorted(sentence_scores, key=lambda x: x[1], reverse=True)]
        
        # Select sentences until summary is at least 500 characters
        summary = []
        char_count = 0
        for sentence in sorted_sentences:
            summary.append(sentence)
            char_count += len(sentence)
            if char_count >= 500:
                break
        
        # If summary is too short, add more sentences
        if char_count < 500 and len(summary) < len(sentences):
            remaining_sentences = [s for s in sentences if s not in summary]
            for sentence in remaining_sentences:
                summary.append(sentence)
                char_count += len(sentence)
                if char_count >= 500:
                    break
        
        # Join sentences and ensure at least 500 characters (truncate or pad if needed)
        summary_text = " ".join(summary)
        if len(summary_text) < 500 and len(transcript) >= 500:
            summary_text = transcript[:500] + "..."
        elif len(summary_text) < 500 and len(transcript) < 500:
            summary_text = transcript  # Use full transcript if too short
        
        return summary_text
    except Exception as e:
        logger.warning(f"Summarization error: {e}")
        return transcript[:500] + ("..." if len(transcript) > 500 else "") or "No transcript available."

def analyze_sentiment(comment):
    """Simple rule-based sentiment analysis."""
    words = set(word_tokenize(comment.lower()))
    positive_count = len(words.intersection(POSITIVE_WORDS))
    negative_count = len(words.intersection(NEGATIVE_WORDS))
    return 1 if positive_count > negative_count else -1 if negative_count > positive_count else 0

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
        
        sentiment_scores = [analyze_sentiment(comment) for comment in comment_texts]
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
        
        results = sorted(results, key=lambda x: x[1], reverse=True)
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
