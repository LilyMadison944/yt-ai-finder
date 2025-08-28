import streamlit as st
import re
import os
import time
import random
import nltk
import cv2 # OpenCV for video processing
import numpy as np
import string
from sentence_transformers import SentenceTransformer, util
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import yt_dlp
from dotenv import load_dotenv
from io import BytesIO # <--- FIX: Added for in-memory file handling

# --- Page Configuration ---
st.set_page_config(
    page_title="YouTube Content AI",
    page_icon="🎬",
    layout="wide"
)

# --- Load Environment Variables & NLTK Data ---
load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
COOKIES_FILE = "cookies.txt"

@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
download_nltk_data()

# --- Caching for Performance ---
@st.cache_resource
def load_sentence_model():
    """Loads the SentenceTransformer model once and caches it."""
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def get_authenticated_service():
    """Authenticates with the YouTube API and caches the service object."""
    if not YOUTUBE_API_KEY:
        st.error("YouTube API key not found. Please add it to your .env file.")
        st.stop()
    try:
        return build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    except Exception as e:
        st.error(f"Failed to initialize YouTube API: {e}")
        st.stop()

# --- Core Logic Functions (Adapted for Streamlit) ---

def clean_text(text):
    if not text: return ""
    text = re.sub(r'\b(uh|um|like|you know)\b', '', text, flags=re.IGNORECASE)
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    tokens = nltk.word_tokenize(text)
    stop_words = set(nltk.corpus.stopwords.words('english')) - {'how', 'make', 'money', 'online', 'podcast'}
    lemmatizer = nltk.stem.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

@st.cache_data
def search_and_analyze_videos(_youtube_service, query):
    """Searches and analyzes videos, caching the results for a given query."""
    status_area = st.empty()
    
    try:
        # 1. Search for videos
        status_area.info("🔎 Searching for top 10 relevant videos...")
        search_response = _youtube_service.search().list(q=query, part="id,snippet", maxResults=10, type="video").execute()
        video_items = search_response.get("items", [])
        if not video_items:
            st.warning("No videos found for this query.")
            return []
        
        video_ids = [item["id"]["videoId"] for item in video_items]
        video_response = _youtube_service.videos().list(part="statistics", id=",".join(video_ids)).execute()
        stats_dict = {item['id']: item['statistics'] for item in video_response['items']}

        videos = []
        for item in video_items:
            video_id = item["id"]["videoId"]
            stats = stats_dict.get(video_id, {})
            videos.append({
                "video_id": video_id, "title": item["snippet"]["title"],
                "likes": int(stats.get("likeCount", 0)), "views": int(stats.get("viewCount", 0))
            })

        # 2. Analyze videos
        video_data = []
        progress_bar = st.progress(0, text="Analyzing videos...")
        for i, video in enumerate(videos):
            status_area.info(f"🧐 Analyzing video {i+1}/{len(videos)}: {video['title'][:40]}...")
            time.sleep(random.uniform(0.5, 1.5)) # Rate limiting
            
            ydl_opts = {'skip_download': True, 'writeautomaticsub': True, 'subtitleslangs': ['en'], 'subtitlesformat': 'vtt', 'outtmpl': f'/tmp/{video["video_id"]}.%(ext)s', 'quiet': True, 'no_warnings': True}
            if os.path.exists(COOKIES_FILE): ydl_opts['cookiefile'] = COOKIES_FILE
            
            vtt_file = f'/tmp/{video["video_id"]}.en.vtt'
            captions = ""
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl: ydl.download([f"https://www.youtube.com/watch?v={video['video_id']}"])
                if os.path.exists(vtt_file):
                    with open(vtt_file, "r", encoding="utf-8") as f: captions = f.read()
                    captions = re.sub(r'WEBVTT\n.*?\n\n', '', captions)
                    captions = re.sub(r'\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}.*?\n', '', captions)
                    captions = re.sub(r'<[^>]+>', '', captions).strip()
                    captions = re.sub(r'\n+', ' ', captions)
            finally:
                if os.path.exists(vtt_file): os.remove(vtt_file)

            combined_text = clean_text(video['title'] + " " + captions)
            if combined_text:
                video_data.append({"video_id": video["video_id"], "title": video["title"], "captions": combined_text, "likes": video["likes"], "views": video["views"]})
            progress_bar.progress((i + 1) / len(videos), text=f"Analyzing video {i+1}/{len(videos)}")
        
        status_area.info("🤖 Calculating relevance scores...")
        model = load_sentence_model()
        query_embedding = model.encode(clean_text(query))
        caption_embeddings = model.encode([v["captions"] for v in video_data])
        sim_scores = util.cos_sim(query_embedding, caption_embeddings)[0]

        for i, video in enumerate(video_data):
            rel_score = sim_scores[i].item() * 100
            like_ratio = (video["likes"] / (video["views"] + 1)) * 100
            video["relevance_score"] = rel_score
            video["composite_score"] = (0.7 * rel_score) + (0.3 * like_ratio)
        
        video_data.sort(key=lambda x: x['composite_score'], reverse=True)
        status_area.empty()
        return video_data[:5]
        
    except HttpError as e:
        if "quota" in str(e).lower():
            st.error("YouTube API Quota Exceeded. Please check your API key or try again tomorrow.")
        else:
            st.error(f"An API error occurred: {e}")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return []

# --- ✨ Thumbnail & Download Features ✨ ---

def get_frame_at_timestamp(video_url, timestamp_seconds):
    """Captures a single frame from a video URL at a specific timestamp."""
    try:
        ydl_opts = {'format': 'best[ext=mp4][height<=480]/best[height<=480]', 'quiet': True}
        if os.path.exists(COOKIES_FILE): ydl_opts['cookiefile'] = COOKIES_FILE
            
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            stream_url = info['url']

        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened(): return None
        
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_seconds * 1000)
        success, frame = cap.read()
        cap.release()
        
        if success:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None
    except Exception as e:
        print(f"Frame capture error: {e}")
        return None

def format_time(seconds):
    return f"{int(seconds // 60):02}:{int(seconds % 60):02}"

# --- ✨ NEW: Function to download video segment ✨ ---
def download_short_segment(video_id, start_time, end_time, title):
    """Downloads a specific segment of a YouTube video and returns it as bytes."""
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    temp_filename = f"temp_{video_id}_{int(start_time)}.mp4"
    
    ydl_opts = {
        'format': 'best[ext=mp4][height<=720]/best[ext=mp4]',
        'outtmpl': temp_filename,
        'quiet': True,
        'no_warnings': True,
        # Using ffmpeg to cut the video segment
        'postprocessor_args': [
            '-ss', str(start_time),
            '-to', str(end_time),
            '-c:v', 'libx264', '-c:a', 'aac'
        ],
    }
    if os.path.exists(COOKIES_FILE): ydl_opts['cookiefile'] = COOKIES_FILE

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        if os.path.exists(temp_filename):
            with open(temp_filename, 'rb') as f:
                video_bytes = f.read()
            os.remove(temp_filename)
            return video_bytes
        else:
            st.error("Download failed. Could not create video clip.")
            return None
    except Exception as e:
        st.error(f"An error occurred during download: {e}")
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        return None

@st.cache_data
def generate_short_ideas(_video_id, query):
    """Generates short ideas and captures a thumbnail for each."""
    video_url = f"https://www.youtube.com/watch?v={_video_id}"
    ydl_opts = {'skip_download': True, 'writesubtitles': True, 'writeautomaticsub': True, 'subtitleslangs': ['en'], 'subtitlesformat': 'vtt', 'outtmpl': f'/tmp/{_video_id}.%(ext)s', 'quiet': True, 'no_warnings': True}
    if os.path.exists(COOKIES_FILE): ydl_opts['cookiefile'] = COOKIES_FILE
    vtt_file = f'/tmp/{_video_id}.en.vtt'
    if os.path.exists(vtt_file): os.remove(vtt_file)

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl: ydl.download([video_url])
        if not os.path.exists(vtt_file): return []
        
        with open(vtt_file, "r", encoding="utf-8") as f: lines = f.read().splitlines()
        
        timed_lines = []
        for i, line in enumerate(lines):
            if '-->' in line:
                try:
                    start_str, _ = line.split(' --> ')
                    h, m, s_ms = start_str.split(':')
                    s, ms = s_ms.split('.')
                    start_time = int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0
                    caption_text = re.sub(r'<[^>]+>', '', lines[i + 1]).strip()
                    if caption_text: timed_lines.append({"text": caption_text, "start": start_time})
                except (ValueError, IndexError): continue

        chunks = []
        for i in range(len(timed_lines)):
            start_line = timed_lines[i]
            current_text = ""
            for j in range(i, len(timed_lines)):
                line = timed_lines[j]
                end_time = line['start']
                duration = end_time - start_line['start']
                if 20 <= duration <= 60:
                    current_text += " " + line['text']
                    if len(clean_text(current_text)) > 15:
                        chunks.append({"text": current_text.strip(), "start": start_line['start'], "end": end_time})
                    break
                current_text += " " + line['text']

        if not chunks: return []
        
        model = load_sentence_model()
        query_embedding = model.encode(clean_text(query))
        chunk_embeddings = model.encode([clean_text(c['text']) for c in chunks])
        similarities = util.cos_sim(query_embedding, chunk_embeddings)[0]
        for i, chunk in enumerate(chunks): chunk['score'] = similarities[i].item()

        sorted_chunks = sorted(chunks, key=lambda x: x['score'], reverse=True)
        final_shorts = []
        for chunk in sorted_chunks:
            if len(final_shorts) >= 3: break
            if not any(abs(chunk['start'] - chosen['start']) < 60 for chosen in final_shorts):
                final_shorts.append(chunk)

        # Generate thumbnails for the final selection
        for short in final_shorts:
            thumbnail_timestamp = short['start'] + (short['end'] - short['start']) / 2
            short['thumbnail'] = get_frame_at_timestamp(video_url, thumbnail_timestamp)
        
        return final_shorts

    finally:
        if os.path.exists(vtt_file): os.remove(vtt_file)

# --- Streamlit UI ---

st.title("🎬 YouTube Content AI")
st.markdown("Discover the most relevant videos for your topic and generate viral Short ideas in one click.")

if not os.path.exists(COOKIES_FILE):
    st.warning(f"For best results, place your `cookies.txt` file in the same directory as this app.")

# Initialize session state
if 'relevant_videos' not in st.session_state:
    st.session_state.relevant_videos = None
if 'query' not in st.session_state:
    st.session_state.query = ""
if 'shorts_to_generate' not in st.session_state:
    st.session_state.shorts_to_generate = None

# --- Search Bar ---
query = st.text_input("Enter a topic or keyword", placeholder="e.g., how to start a successful podcast")
if st.button("Search Videos", type="primary") and query:
    st.session_state.query = query
    # Clear previous shorts when a new search is made
    st.session_state.shorts_to_generate = None 
    st.session_state.relevant_videos = search_and_analyze_videos(get_authenticated_service(), query)

# --- Display Results ---
if st.session_state.relevant_videos:
    st.header("🏆 Top 5 Recommended Videos")
    for video in st.session_state.relevant_videos:
        video_url = f"https://www.youtube.com/watch?v={video['video_id']}"
        with st.container(border=True):
            col1, col2 = st.columns([1, 4])
            with col1:
                st.image(f"https://i.ytimg.com/vi/{video['video_id']}/hqdefault.jpg")
            with col2:
                st.subheader(f"[{video['title']}]({video_url})")
                st.caption(f"**Views:** {video['views']:,} | **Likes:** {video['likes']:,} | **Relevance:** {video['relevance_score']:.0f}/100")
                if st.button("💡 Generate Short Ideas", key=video['video_id']):
                    st.session_state.shorts_to_generate = video
                    # Use st.rerun() for a smoother state update
                    st.rerun()
                        
# --- Display Generated Shorts ---
if 'shorts_to_generate' in st.session_state and st.session_state.shorts_to_generate:
    video = st.session_state.shorts_to_generate
    st.header(f"💡 Short Ideas for: *{video['title']}*")
    
    with st.spinner("Analyzing video transcript and generating thumbnails... This can take a minute."):
        short_ideas = generate_short_ideas(video['video_id'], st.session_state.query)

    if not short_ideas:
        st.error("Could not generate any short ideas. The video may not have captions or enough relevant content.")
    else:
        # Check if number of ideas is less than 3 to avoid layout errors
        num_shorts = len(short_ideas)
        cols = st.columns(num_shorts if num_shorts > 0 else 1)

        for i, short in enumerate(short_ideas):
            with cols[i]:
                with st.container(border=True):
                    if short['thumbnail'] is not None:
                        st.image(short['thumbnail'], caption=f"Frame from {format_time(short['start'])}")
                    else:
                        st.image(f"https://i.ytimg.com/vi/{video['video_id']}/hqdefault.jpg", caption="Could not generate thumbnail")
                    
                    st.markdown(f"**🕒 Timestamp:** `{format_time(short['start'])} - {format_time(short['end'])}`")
                    st.markdown(f"**💬 Transcript:** *\"{short['text']}\"*")

                    # --- ✨ NEW: "Open in new tab" link ✨ ---
                    timestamp_url = f"https://www.youtube.com/watch?v={video['video_id']}&t={int(short['start'])}s"
                    st.markdown(f'<a href="{timestamp_url}" target="_blank">🔗 Open in new tab</a>', unsafe_allow_html=True)
                    
                    # --- ✨ NEW: Download Button ✨ ---
                    # The download logic is handled by a button click which will re-run the script.
                    # We prepare the data for the download button.
                    placeholder = st.empty()
                    with placeholder.container():
                        if st.button(f"📥 Download Clip {i+1}", key=f"download_{video['video_id']}_{i}"):
                            with st.spinner("Preparing your download..."):
                                video_bytes = download_short_segment(video['video_id'], short['start'], short['end'], video['title'])
                                if video_bytes:
                                    # When the download is ready, we replace the button with a download_button
                                    # We need to use session state to manage this transition
                                    st.session_state[f'video_bytes_{i}'] = video_bytes
                    
                    # If the bytes are in the session state, show the actual download button
                    if f'video_bytes_{i}' in st.session_state:
                        video_bytes_data = st.session_state.pop(f'video_bytes_{i}') # Use pop to clear state after use
                        
                        # Sanitize title for filename
                        safe_title = re.sub(r'[^\w\s-]', '', video['title']).strip()
                        safe_title = re.sub(r'[-\s]+', '-', safe_title)
                        
                        placeholder.download_button(
                            label=f"✅ Click to Save Clip {i+1}",
                            data=video_bytes_data,
                            file_name=f"{safe_title}_short_{i+1}.mp4",
                            mime="video/mp4",
                            key=f"final_download_{video['video_id']}_{i}"
                        )
