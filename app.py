import streamlit as st
import requests
import re
import json
from openai import OpenAI
import os
import pandas as pd
import altair as alt
from dotenv import load_dotenv

# -----------------------------
# CONFIG
# -----------------------------

def load_secrets(file_path="secrets.txt"):
    secrets = {}
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, value = line.split("=", 1)
            secrets[key.strip()] = value.strip()
    return secrets

# Load secrets
#secrets = load_secrets()

YOUTUBE_TRANSCRIPT_API_KEY = st.secrets["YOUTUBE_TRANSCRIPT_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

YOUTUBE_TRANSCRIPT_API_URL = "https://www.youtube-transcript.io/api/transcripts"

# Load environment variables from the .env file
#load_dotenv(".env")

# Ensure the OpenAI API key is set
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY environment variable is not set. Please set it to proceed.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# FUNCTIONS
# -----------------------------

def extract_video_id(youtube_url):
    print("Getting video ID from URL:", youtube_url)
    """
    Extracts YouTube video ID from different URL formats
    """
    patterns = [
        r"v=([^&]+)",
        r"youtu\.be/([^?&]+)",
        r"youtube\.com/embed/([^?&]+)"
    ]
    for pattern in patterns:
        print(pattern)
        match = re.search(pattern, youtube_url)
        if match:
            return match.group(1)
    return None


def fetch_transcript(video_id):
    """
    Fetch transcript from youtube-transcript.io
    """
    response = requests.post(
        YOUTUBE_TRANSCRIPT_API_URL,
        headers={
            "Authorization": f"Basic {YOUTUBE_TRANSCRIPT_API_KEY}",
            "Content-Type": "application/json"
        },
        json={"ids": [video_id]}
    )

    response.raise_for_status()
    data = response.json()
    try:
        # Case 1: If top-level combined text exists
        transcript_text = data[0]["text"]

    except (KeyError, IndexError, TypeError):
        # Case 2: Fallback – build from transcript chunks
        transcript_chunks = data[0]["tracks"][0]["transcript"]
        transcript_text = " ".join(
            chunk["text"] for chunk in transcript_chunks if chunk.get("text")
        )

    return transcript_text




def analyze_misinformation(transcript):

    prompt = f"""
        You are a misinformation evaluation model.

        Your task is to analyze the provided transcript and assign a misinformation score
        for EACH of the following categories.

        CATEGORIES (ALL REQUIRED):
        1. Factual Core
        2. Distorted Legal or Investigative Claims
        3. Exaggerated or Unsupported Statistics
        4. Ethnic or Religious Framing Bias
        5. Anecdotal or Possibly Fictional Cases
        6. Real Events or Scandals Referenced Accurately

        SCORING RULES:
        - Each value MUST be an INTEGER between 0 and 100.
        - 0 = fully accurate / no misinformation
        - 100 = completely false / extreme misinformation
        - Use intermediate values where appropriate.

        OUTPUT RULES (STRICT):
        - Output MUST be VALID JSON
        - Output MUST contain ALL six keys exactly as written
        - Output MUST contain ONLY numbers as values
        - Do NOT include explanations
        - Do NOT include markdown
        - Do NOT include extra text
        - Do NOT change key names
        - Do NOT add or remove keys

        EXACT OUTPUT FORMAT:
        {{
        "Factual Core": <integer>,
        "Distorted Legal or Investigative Claims": <integer>,
        "Exaggerated or Unsupported Statistics": <integer>,
        "Ethnic or Religious Framing Bias": <integer>,
        "Anecdotal or Possibly Fictional Cases": <integer>,
        "Real Events or Scandals Referenced Accurately": <integer>
        }}

        --------------------
        ONE-SHOT EXAMPLE
        --------------------

        Transcript:
        "The government passed a new law last week. A viral post claims the law reduces everyone's income by 50% and targets only single mothers. No official document supports this claim."

        Correct Output:
        {{
        "Factual Core": 30,
        "Distorted Legal or Investigative Claims": 80,
        "Exaggerated or Unsupported Statistics": 90,
        "Ethnic or Religious Framing Bias": 0,
        "Anecdotal or Possibly Fictional Cases": 10,
        "Real Events or Scandals Referenced Accurately": 20
        }}

        --------------------
        NOW ANALYZE THIS TRANSCRIPT
        --------------------

        Transcript:
        {transcript}
        """

    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
    )
    

    return response.choices[0].message.content


# -----------------------------
# STREAMLIT UI
# -----------------------------

st.set_page_config(page_title="PRAMAAN", layout="centered")


col_left, col_center, col_right = st.columns([1, 1, 1])

with col_left:
    st.image(
        "Images/Cyberpeace logo.png",
        width=140
    )

with col_center:
    st.image(
        "Images/Google support logo.png",
        width=140
    )

with col_right:
    st.image(
        "Images/ISB IDS Logo Main.png",
        width=140
    )

#st.title("YouTube Video Misinformation Analyzer")

st.markdown("""
## **PRAMAAN**
*Platform for Reliability Assessment & Misinformation Analysis Network*

A research-driven tool for evaluating misinformation severity across multiple dimensions.
""")


youtube_url = st.text_input(
    "Paste YouTube Video URL",
    placeholder="https://www.youtube.com/watch?v=XXXXXXXX"
)

if st.button("Analyze Video"):
    if not youtube_url:
        st.error("Please enter a YouTube URL.")
        st.stop()

    video_id = extract_video_id(youtube_url)

    if not video_id:
        st.error("Invalid YouTube URL.")
        st.stop()

    with st.spinner("Fetching transcript..."):
        try:
            transcript = fetch_transcript(video_id)
        except Exception as e:
            st.error(f"Transcript fetch failed: {e}")
            st.stop()

    st.success("Transcript fetched successfully!")

    with st.spinner("Analyzing misinformation..."):
        try:
            result = analyze_misinformation(transcript)
            parsed_result = json.loads(result)

            # Convert JSON result to DataFrame
            df_scores = pd.DataFrame(
                parsed_result.items(),
                columns=["Category", "Misinformation Score"]
            )

            # Ensure score is integer
            df_scores["Misinformation Score"] = df_scores["Misinformation Score"].astype(int)

        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.text(result)
            st.stop()

    st.subheader("Misinformation Scores")
    st.write("Scores range from 0 (accurate) to 100 (misinformation).")
    st.dataframe(
        df_scores,
        use_container_width=True,
        hide_index=True
    )

    chart = (
        alt.Chart(df_scores)
        .mark_bar()
        .encode(
            x=alt.X("Category:N", sort="-y", title="Category"),
            y=alt.Y(
                "Misinformation Score:Q",
                scale=alt.Scale(domain=[0, 100]),
                title="Misinformation Score (0–100)"
            ),
            tooltip=["Category", "Misinformation Score"]
        )
        .properties(height=400)
    )

    st.altair_chart(chart, use_container_width=True)

    with st.expander("📄 View Transcript"):
        st.write(transcript)

