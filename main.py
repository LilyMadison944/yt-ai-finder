import streamlit as st
import re
import os
import time
import requests
import string
import yt_dlp
from pathlib import Path

# --- NLTK and SentenceTransformer Imports ---
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util

# --- ⚙️ App Configuration & Setup ---
st.set_page_config(page_title="AI YouTube Content Finder", layout="wide")

# Get API key from Streamlit secrets or local .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass # Fails silently if dotenv is not installed

API_KEY = os.getenv("RAPIDAPI_KEY")

# 10 is the ID for the "Music" category on YouTube.
DISALLOWED_CATEGORIES = ['10']

# --- Enhanced Logging for Debug Mode ---
def log_message(message, data=None):
    if st.session_state.get("debug_mode", False):
        log_entry = f"[{time.strftime('%H:%M:%S')}] {message}"
        if data:
            import json
            log_entry += f"\n└── DATA: {json.dumps(data, indent=2)}\n"
        st.session_state.logs.append(log_entry)

# Initialize Streamlit Session State
if "logs" not in st.session_state:
    st.session_state.logs = []
if "videos" not in st.session_state:
    st.session_state.videos = None
if "selected_video_for_shorts" not in st.session_state:
    st.session_state.selected_video_for_shorts = None

# --- NLTK Download Function (Crucial for Streamlit Cloud) ---
@st.cache_resource
def download_nltk_data():
    """Downloads necessary NLTK models directly."""
    try:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        return True
    except Exception as e:
        st.error(f"Error downloading NLTK data: {e}")
        return False

# Ensure NLTK data is downloaded before the app runs
nltk_data_available = download_nltk_data()

# --- Load SentenceTransformer Model ---
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# --- Backend Core Functions ---
def clean_text(text):
    if not text: return ""
    text = re.sub(r'\[.*?\]', '', text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [WordNetLemmatizer().lemmatize(w) for w in tokens if w.isalpha() and w not in stop_words]
    return " ".join(tokens)

def search_videos_api(query, max_results=15):
    url = "https://youtube-v311.p.rapidapi.com/search/"
    headers = {"x-rapidapi-key": API_KEY, "x-rapidapi-host": "youtube-v311.p.rapidapi.com"}
    params = {"q": query, "part": "snippet", "maxResults": max_results, "type": "video"}
    log_message("➡️ Sending Search Request to API...", data=params)

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        result = response.json()
        log_message(f"✅ API Search Success. Found {len(result.get('items', []))} items.")
        return result.get('items', [])
    except requests.exceptions.RequestException as e:
        st.error(f"API Search Error: {e}")
        log_message(f"🔴 API Search Error", data=str(e))
        return []

def get_video_info(video_id):
    url = "https://youtube-v311.p.rapidapi.com/videos/"
    headers = {"x-rapidapi-key": API_KEY, "x-rapidapi-host": "youtube-v311.p.rapidapi.com"}
    params = {"part": "snippet,statistics", "id": video_id}
    log_message(f"➡️ Fetching Video Info for ID: {video_id}", data=params)
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        info = response.json().get('items', [{}])[0]
        cat_id = info.get('snippet', {}).get('categoryId', 'N/A')
        log_message(f"✅ Fetched Info for '{info.get('snippet', {}).get('title', 'N/A')}' (Category ID: {cat_id})")
        return info
    except requests.exceptions.RequestException as e:
        log_message(f"🔴 Failed to fetch info for {video_id}", data=str(e))
        return None

def get_timed_transcript(video_id):
    log_message(f"⏳ Fetching captions for video ID: {video_id} using yt-dlp...")
    ydl_opts = {'writesubtitles': True, 'writeautomaticsub': True, 'subtitleslangs': ['en'], 'skip_download': True, 'outtmpl': f'subtitles/%(id)s'}
    Path("subtitles").mkdir(exist_ok=True)
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f'https://www.youtube.com/watch?v={video_id}'])

        vtt_file_path = next(Path("subtitles").glob(f"{video_id}.*.vtt"), None)
        if not vtt_file_path:
            log_message("🔴 VTT caption file not found.")
            return None

        with open(vtt_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        transcript_data = []
        for i, line in enumerate(lines):
            if "-->" in line:
                try:
                    start, end = line.strip().split(" --> ")
                    text = lines[i+1].strip()
                    start_s = sum(x * float(t) for x, t in zip([3600, 60, 1], start.split('.')[0].split(':'))) + float("0." + start.split('.')[-1])
                    transcript_data.append({"start": start_s, "text": text})
                except (ValueError, IndexError):
                    continue

        log_message(f"✅ Captions parsed. Segments: {len(transcript_data)}")
        os.remove(vtt_file_path)
        return transcript_data
    except Exception as e:
        log_message(f"🔴 Error fetching transcript with yt-dlp", data=str(e))
        return None

def find_best_clips(transcript, query, clip_duration=58, num_clips=3):
    log_message("🧠 Analyzing transcript to find best clips...")
    if not transcript or len(transcript) < 2: return []

    clean_query = clean_text(query)
    query_embedding = model.encode(clean_query, convert_to_tensor=True)

    potential_clips = []
    for i in range(len(transcript)):
        start_time = transcript[i]['start']
        end_time = start_time + clip_duration
        current_chunk_texts = [t['text'] for t in transcript if start_time <= t['start'] < end_time and not t['text'].startswith('[')]

        if current_chunk_texts:
            potential_clips.append({"start": start_time, "end": end_time, "transcript": " ".join(current_chunk_texts)})

    if not potential_clips: return []

    clip_texts = [clean_text(c['transcript']) for c in potential_clips]
    clip_embeddings = model.encode(clip_texts, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, clip_embeddings)

    for i, clip in enumerate(potential_clips):
        clip['score'] = cosine_scores[0][i].item()

    potential_clips.sort(key=lambda x: x['score'], reverse=True)

    final_clips = []
    for clip in potential_clips:
        if not any(max(clip['start'], fc['start']) < min(clip['end'], fc['end']) for fc in final_clips):
            final_clips.append(clip)
        if len(final_clips) == num_clips:
            break

    log_message(f"✅ Found {len(final_clips)} potential clips.")
    return final_clips

# --- 🖼️ Streamlit UI ---
st.title("AI YouTube Content Finder")
st.markdown("Your intelligent guide to clickbait-free content and instant video highlights.")

with st.sidebar:
    st.header("⚙️ Settings")
    st.session_state.debug_mode = st.checkbox("Enable Debug Mode")
    if st.session_state.debug_mode:
        st.subheader("📝 Debug Logs")
        if st.button("Clear Logs"):
            st.session_state.logs = []
        log_container = st.container(height=400)
        log_container.text("\n".join(st.session_state.logs[::-1]))

search_query = st.text_input("Enter your search query:", placeholder="e.g., how to make money using rapidapi")

if st.button("Analyze Content", type="primary"):
    if not nltk_data_available:
        st.error("NLTK data could not be downloaded. The app cannot proceed.")
    elif not API_KEY:
        st.error("RAPIDAPI_KEY not found. Please add it to your Streamlit secrets.")
    elif not search_query:
        st.error("Please enter a search query.")
    else:
        st.session_state.logs = []
        log_message("--- 🚀 Starting New Analysis ---")

        with st.spinner("Finding and analyzing relevant videos..."):
            search_results = search_videos_api(search_query)
            if not search_results:
                st.warning("No videos found for this query.")
                st.stop()

            all_video_details = []
            for item in search_results:
                video_id = item['id']['videoId']
                info = get_video_info(video_id)
                if info:
                    category_id = info.get('snippet', {}).get('categoryId')
                    if category_id in DISALLOWED_CATEGORIES:
                        log_message(f"🚫 Skipping video '{info['snippet']['title']}' due to disallowed category: {category_id}")
                        continue

                    transcript = get_timed_transcript(video_id)
                    details = {
                        "video_id": video_id,
                        "title": info['snippet']['title'],
                        "description": info['snippet']['description'],
                        "thumbnail": info['snippet']['thumbnails']['high']['url'],
                        "views": int(info['statistics'].get('viewCount', 0)),
                        "likes": int(info['statistics'].get('likeCount', 0)),
                        "transcript": transcript
                    }
                    all_video_details.append(details)

            if not all_video_details:
                st.error("Could not fetch details for any relevant videos.")
                st.stop()

            log_message("⚖️ Calculating relevance scores...")
            query_embedding = model.encode(clean_text(search_query))
            for video in all_video_details:
                if video['transcript']:
                    combined_text = video['title'] + " " + video['description'] + " " + " ".join([t['text'] for t in video['transcript']])
                    content_embedding = model.encode(clean_text(combined_text))
                    sim_score = util.cos_sim(query_embedding, content_embedding)[0].item()
                    like_ratio = (video.get("likes", 0) / (video.get("views", 1) + 1)) * 1000
                    video["composite_score"] = (0.8 * sim_score * 100) + (0.2 * like_ratio)
                else:
                    video["composite_score"] = 0

            all_video_details.sort(key=lambda x: x['composite_score'], reverse=True)
            st.session_state.videos = all_video_details[:5]
            log_message("✅ Analysis complete. Top 5 videos identified.")

if st.session_state.videos:
    st.header("🏆 Top 5 Relevant Videos")
    for i, video in enumerate(st.session_state.videos):
        if video['composite_score'] == 0: continue
        with st.container(border=True):
            col1, col2 = st.columns([1, 3])
            col1.image(video['thumbnail'])
            col2.subheader(f"{i+1}. {video['title']}")
            col2.caption(f"🔗 https://www.youtube.com/watch?v={video['video_id']} |  score: {video['composite_score']:.2f}")
            col2.markdown(f"👀 **Views:** {video['views']:,} | 👍 **Likes:** {video['likes']:,}")

            if col2.button("✨ Generate Short Ideas", key=f"shorts_{video['video_id']}"):
                st.session_state.selected_video_for_shorts = video
                st.rerun()

if st.session_state.selected_video_for_shorts:
    video = st.session_state.selected_video_for_shorts
    st.header(f"💡 Generated Ideas for: {video['title']}")

    transcript_data = video['transcript']

    with st.spinner("Finding the best clips..."):
        if transcript_data:
            clips = find_best_clips(transcript_data, search_query)
            if clips:
                cols = st.columns(len(clips))
                for i, col in enumerate(cols):
                    with col:
                        clip_data = clips[i]
                        start_time = clip_data['start']
                        end_time = clip_data['end']

                        st.subheader(f"Idea #{i+1}")
                        st.image(video['thumbnail'])
                        st.markdown(f"**Timestamp:** `{time.strftime('%M:%S', time.gmtime(start_time))}` to `{time.strftime('%M:%S', time.gmtime(end_time))}`")
                        with st.expander("Show Transcript"):
                            st.text(clip_data['transcript'])
            else:
                st.error("Could not find any relevant clips in this video's transcript.")
        else:
            st.error("No captions were found for this video. Unable to generate ideas.")
