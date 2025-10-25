# app.py
import streamlit as st
from transformers import pipeline
from fer.fer import FER
import cv2
import pandas as pd
from datetime import datetime
import time
import os

# -------------------------
# Config / Setup
# -------------------------
st.set_page_config(page_title="Amdox AI Task Optimizer", layout="wide")
st.title("🤖 Amdox AI-Powered Task Optimizer")
st.write("Text + Webcam emotion detection · simple · real-time prototype")

DATA_PATH = "data"
CSV_PATH = os.path.join(DATA_PATH, "mood_history.csv")
os.makedirs(DATA_PATH, exist_ok=True)

# ensure CSV exists
if not os.path.exists(CSV_PATH):
    pd.DataFrame(columns=["timestamp", "source", "emotion", "score"]).to_csv(CSV_PATH, index=False)

# -------------------------
# Load models (simple)
# -------------------------
@st.cache_resource(show_spinner=False)
def load_text_model():
    # lightweight emotion model
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

@st.cache_resource(show_spinner=False)
def load_fer_detector():
    return FER(mtcnn=True)

text_model = load_text_model()
fer_detector = load_fer_detector()

# -------------------------
# Task map (simple rules)
# -------------------------
task_map = {
    "joy": ["Creative/brainstorming tasks", "Pair programming / collaborative work"],
    "happy": ["Creative/brainstorming tasks", "Knowledge sharing"],
    "sadness": ["Light documentation", "Short wellness break", "One-on-one check-in"],
    "sad": ["Light documentation", "Wellness break"],
    "anger": ["Independent focused tasks", "Short break", "Calming exercise"],
    "fear": ["Supportive tasks / guidance", "Low-pressure work"],
    "surprise": ["Exploration / discovery tasks"],
    "disgust": ["Code review / independent tasks"],
    "neutral": ["Routine tasks", "Reporting / admin"]
}

# normalize keys used by different models
def normalize_emotion(e):
    e = e.lower()
    # transform similar labels if needed
    mapping = {
        "joy": "joy", "happy": "joy",
        "sadness": "sad", "sad": "sad",
        "anger": "angry", "angry": "angry",
        "fear": "fear", "surprise": "surprise",
        "disgust": "disgust", "neutral": "neutral"
    }
    for k in mapping:
        if k in e:
            return mapping[k]
    return e

# -------------------------
# Utilities: log & read
# -------------------------
def log_emotion(source, emotion, score):
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
        "emotion": emotion,
        "score": round(float(score), 4)
    }
    df = pd.read_csv(CSV_PATH)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(CSV_PATH, index=False)

@st.cache_data(ttl=5)
def read_history():
    return pd.read_csv(CSV_PATH)

# -------------------------
# Left column: controls
# -------------------------
left, right = st.columns([1, 2])

with left:
    st.header("Input Modes")
    # TEXT input
    st.subheader("1) Text input")
    text = st.text_input("Type how you feel (e.g., I'm tired but okay):")
    if st.button("Detect from text"):
        if text.strip() == "":
            st.warning("Please type something first.")
        else:
            try:
                out = text_model(text)[0]  # {'label': 'joy', 'score': 0.xxx}
                label = normalize_emotion(out["label"])
                score = out.get("score", 0.0)
                st.success(f"Detected emotion: **{label}** ({score:.2f})")
                # Recommend tasks
                recs = task_map.get(label, ["Keep going — no special recommendation"])
                st.write("**Recommended tasks:**")
                for r in recs:
                    st.write("- " + r)
                # Log
                log_emotion("text", label, score)
            except Exception as e:
                st.error("Text model error: " + str(e))

    st.markdown("---")
    # WEBCAM control
    st.subheader("2) Webcam (real-time)")
    webcam_on = st.checkbox("Turn On Webcam", value=False)
    st.write("Tip: allow camera access when browser asks.")

    st.markdown("---")
    st.subheader("Quick Controls")
    if st.button("Clear history CSV"):
        pd.DataFrame(columns=["timestamp", "source", "emotion", "score"]).to_csv(CSV_PATH, index=False)
        st.experimental_rerun()

# -------------------------
# Right column: display / webcam / analytics
# -------------------------
with right:
    st.subheader("Live / Analytics")

    # Show recent history and counts
    history = read_history()
    st.markdown("**Recent moods**")
    if history.empty:
        st.write("No detections yet.")
    else:
        st.dataframe(history.tail(10).reset_index(drop=True), use_container_width=True)

    st.markdown("**Emotion counts**")
    if not history.empty:
        counts = history["emotion"].value_counts()
        st.bar_chart(counts)

    # Stress alert: examine last 3 entries (any source)
    if not history.empty and len(history) >= 3:
        last3 = history["emotion"].tail(3).tolist()
        if all(e in ["sad", "angry"] or "sad" in e or "angry" in e for e in last3):
            st.error("⚠️ Alert: last 3 detections show repeated negative emotions (sad/angry). Consider notifying HR/support.")

    # Webcam live feed and detection
    if webcam_on:
        frame_window = st.empty()
        status_text = st.empty()
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Webcam not accessible. Check your camera and browser permissions.")
            else:
                status_text.info("Webcam running — press the checkbox again to stop.")
                # run a short loop (Streamlit dislikes infinite loops; we'll loop until checkbox flips)
                while webcam_on:
                    ret, frame = cap.read()
                    if not ret:
                        status_text.warning("No frame from webcam.")
                        break

                    # mirror and resize for speed
                    frame = cv2.flip(frame, 1)
                    small = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)

                    # detect faces/emotions
                    try:
                        results = fer_detector.detect_emotions(small)
                    except Exception as e:
                        results = []
                    label_text = "No face"
                    score_val = 0.0

                    if results:
                        top = results[0]["emotions"]
                        # choose highest score emotion
                        top_emotion, top_score = max(top.items(), key=lambda x: x[1])
                        normalized = normalize_emotion(top_emotion)
                        label_text = f"{normalized} ({top_score:.2f})"
                        score_val = top_score

                        # log to CSV
                        log_emotion("webcam", normalized, top_score)

                        # show recommendations (small)
                        recs = task_map.get(normalized, ["Keep going!"])
                        status_text.success(f"Detected: {normalized} — Tasks: {', '.join(recs[:2])}")

                    # overlay text
                    cv2.putText(small, label_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2, cv2.LINE_AA)

                    # convert BGR->RGB
                    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                    frame_window.image(rgb, channels="RGB")

                    # short sleep to make it real-time but not too fast
                    time.sleep(0.25)

                    # update webcam_on from session (stop when user unchecks)
                    webcam_on = st.checkbox("Turn On Webcam", value=True)
                cap.release()
                cv2.destroyAllWindows()
        except Exception as e:
            st.error("Webcam failed: " + str(e))
    else:
        st.info("Webcam is off. Use the checkbox to enable it.")

# -------------------------
# Footer: quick notes
# -------------------------
st.markdown("---")
st.caption("Data stored: data/mood_history.csv (timestamp, source, emotion, score). No personal identifiers are saved.")

emotions = ["Angry", "Sad", "Neutral", "Happy", "Surprised", "Fear", "Disgust", "Excited", "Calm"]
selected_emotion = st.selectbox("How are you feeling today?", emotions)

st.write(f"Detected: {selected_emotion}")
# Save to CSV / database
