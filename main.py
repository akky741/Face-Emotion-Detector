# pip install streamlit opencv-python fer matplotlib pillow

import streamlit as st
import cv2
from fer import FER
import numpy as np

st.set_page_config(page_title="Emotion Recognition", layout="centered")
st.title("🎭 Real-Time Emotion Recognition")

# Initialize session state
if "run" not in st.session_state:
    st.session_state.run = False

# Sidebar controls
st.sidebar.header("Controls")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 0.0, 1.0, 0.7, 0.05
)

start = st.sidebar.button("▶ Start Webcam")
stop = st.sidebar.button("⏹ Stop Webcam")

# Emotion detector
detector = FER(mtcnn=True)
EMOTIONS = ["happy", "sad", "neutral", "angry"]

# Start webcam
if start:
    st.session_state.run = True

# Stop webcam
if stop:
    st.session_state.run = False
    st.warning("Webcam stopped")
    st.stop()

# Webcam frame placeholder
frame_placeholder = st.empty()
chart_placeholder = st.empty()

if st.session_state.run:
    cap = cv2.VideoCapture(0)

    while st.session_state.run:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.detect_emotions(rgb_frame)

        emotion_counts = {e: 0 for e in EMOTIONS}

        for result in results:
            (x, y, w, h) = result["box"]
            emotions = result["emotions"]

            filtered = {k: v for k, v in emotions.items() if k in EMOTIONS}
            if filtered:
                dominant = max(filtered, key=filtered.get)
                confidence = filtered[dominant]

                if confidence >= confidence_threshold:
                    emotion_counts[dominant] += 1
                    cv2.rectangle(rgb_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(
                        rgb_frame,
                        f"{dominant} ({confidence:.2f})",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (255, 0, 0),
                        2
                    )

        frame_placeholder.image(rgb_frame, channels="RGB")
        chart_placeholder.bar_chart(emotion_counts)

    cap.release()
    cv2.destroyAllWindows()
