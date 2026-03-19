from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "muse_v3.csv"
MODEL_PATH = BASE_DIR / "model.h5"

EMOTION_LABELS = {
    0: "Angry",
    1: "Disgusted",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprised",
}

SONG_BUCKETS = {
    "Neutral": (54000, 72000),
    "Angry": (36000, 54000),
    "Fearful": (18000, 36000),
    "Happy": (72000, None),
    "Sad": (0, 18000),
    "Disgusted": (0, 18000),
    "Surprised": (72000, None),
}


st.set_page_config(
    page_title="Emotion-Based Song Recommendation",
    page_icon="🎵",
    layout="wide",
)


@st.cache_data
def load_song_data() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)
    df["link"] = df["spotify_id"].apply(
        lambda track_id: f"https://open.spotify.com/track/{track_id}" if pd.notna(track_id) else None
    )
    df["name"] = df["track"]
    df["emotional"] = df["number_of_emotion_tags"]
    df["pleasant"] = df["valence_tags"]
    df = df[["name", "emotional", "pleasant", "link", "artist"]]
    return df.sort_values(by=["emotional", "pleasant"]).reset_index(drop=True)


@st.cache_resource
def load_model() -> Sequential:
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(48, 48, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation="softmax"))
    model.load_weights(MODEL_PATH)
    return model


def ranked_unique_emotions(emotions: list[str]) -> list[str]:
    counts = Counter(emotions)
    return [emotion for emotion, _ in counts.most_common()]


def get_bucket(df: pd.DataFrame, emotion: str) -> pd.DataFrame:
    start, end = SONG_BUCKETS.get(emotion, SONG_BUCKETS["Sad"])
    return df.iloc[start:end] if end is not None else df.iloc[start:]


def recommend_songs(df: pd.DataFrame, emotions: list[str]) -> pd.DataFrame:
    if not emotions:
        return pd.DataFrame(columns=df.columns)

    sample_sizes = {
        1: [30],
        2: [30, 20],
        3: [55, 20, 15],
        4: [30, 29, 18, 9],
    }.get(len(emotions), [10, 7, 6, 5, 2])

    picks = []
    for emotion, sample_size in zip(emotions[: len(sample_sizes)], sample_sizes):
        bucket = get_bucket(df, emotion)
        picks.append(bucket.sample(n=min(sample_size, len(bucket)), random_state=42))

    return pd.concat(picks, ignore_index=True).drop_duplicates(subset=["name", "artist"])


def detect_emotion(image_bytes: bytes, model: Sequential) -> tuple[str | None, np.ndarray | None]:
    image = Image.open(image_bytes if hasattr(image_bytes, "read") else image_bytes)
    rgb_image = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return None, rgb_image

    x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
    roi_gray = gray[y : y + h, x : x + w]
    cropped = cv2.resize(roi_gray, (48, 48))
    cropped = np.expand_dims(np.expand_dims(cropped, axis=-1), axis=0)

    prediction = model.predict(cropped, verbose=0)
    emotion = EMOTION_LABELS[int(np.argmax(prediction))]

    annotated = rgb_image.copy()
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (58, 134, 255), 3)
    cv2.putText(
        annotated,
        emotion,
        (x, max(30, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return emotion, annotated


st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(255, 166, 77, 0.18), transparent 32%),
            radial-gradient(circle at top right, rgba(69, 179, 157, 0.18), transparent 24%),
            linear-gradient(135deg, #0d1b2a 0%, #1b263b 45%, #415a77 100%);
    }
    .hero {
        padding: 1.5rem 1.8rem;
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 22px;
        background: rgba(13, 27, 42, 0.62);
        backdrop-filter: blur(10px);
        color: white;
        margin-bottom: 1rem;
    }
    .hero h1 {
        margin-bottom: 0.25rem;
    }
    .song-card {
        padding: 0.9rem 1rem;
        border-radius: 16px;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.12);
        margin-bottom: 0.8rem;
    }
    .song-card a {
        color: #ffd166;
        text-decoration: none;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>Emotion-Based Song Recommendation</h1>
        <p>Take a photo or upload one, detect your facial emotion, and get music picks that match the mood.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

song_df = load_song_data()
model = load_model()

left, right = st.columns([1.05, 1.2])

with left:
    st.subheader("Capture or Upload")
    photo = st.camera_input("Use your camera")
    uploaded = st.file_uploader("Or upload an image", type=["jpg", "jpeg", "png"])
    image_source = photo or uploaded

    if image_source is None:
        st.info("Add a photo to detect emotion and unlock recommendations.")
    else:
        emotion, annotated_image = detect_emotion(image_source, model)
        if annotated_image is not None:
            st.image(annotated_image, caption="Detected face", use_container_width=True)

        if emotion is None:
            st.warning("No face was detected. Try a clearer, front-facing image with better lighting.")
        else:
            st.success(f"Detected emotion: {emotion}")
            ranked_emotions = ranked_unique_emotions([emotion])
            recommendations = recommend_songs(song_df, ranked_emotions)
            st.session_state["recommendations"] = recommendations
            st.session_state["emotion"] = emotion

with right:
    st.subheader("Recommended Songs")
    current_emotion = st.session_state.get("emotion")
    recommendations = st.session_state.get("recommendations")

    if current_emotion and recommendations is not None and not recommendations.empty:
        st.caption(f"Showing songs for {current_emotion}")
        for index, row in recommendations.head(30).iterrows():
            song_link = row["link"] if pd.notna(row["link"]) else "#"
            st.markdown(
                f"""
                <div class="song-card">
                    <a href="{song_link}" target="_blank">{index + 1}. {row['name']}</a>
                    <div>{row['artist']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.write("Your recommendations will appear here after emotion detection.")
