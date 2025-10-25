
# 🤖 Amdox AI-Powered Task Optimizer

**Real-time text + webcam emotion detection with task recommendations**  
A simple prototype designed to monitor employee moods and suggest tasks that align with their emotional state, fostering productivity and well-being.

---

## **Features**

1. **Real-Time Emotion Detection**
   - Detect emotions from **text input** or **webcam video**.
   - Supports moods like: Angry, Sad, Neutral, Happy, Surprised, Fear, Disgust, Excited, Calm.

2. **Manual Emotion Selection**
   - Users can **manually select how they feel** from a dropdown menu.
   - Provides an alternative or supplement to automated detection.

3. **Task Recommendation**
   - Suggests tasks aligned with the detected or selected emotion.
   - Example: `Detected: Sad → Tasks: Light documentation, Wellness break`.

4. **Historical Mood Tracking**
   - Stores mood data in `data/mood_history.csv`.
   - Tracks trends over time to analyze employee well-being.

5. **Team Mood Analytics**
   - Aggregates mood data to monitor **overall team morale and productivity trends**.

6. **Stress Management Alerts**
   - Notifies HR or managers if prolonged negative emotions or stress are detected.

7. **Data Privacy**
   - All data is **anonymized**.
   - Only timestamps, source, emotion, and score are stored. No personal identifiers.

---

## **Input Modes**

1. **Text Input**
   - Type how you feel: `"I'm tired but okay"`.
   
2. **Webcam (Real-Time)**
   - Detect facial expressions via live video.
   
3. **Manual Selection**
   - Select your mood from a predefined list.

---

## **Quick Start**

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/amdox-task-optimizer.git
cd amdox-task-optimizer
````

2. **Create virtual environment & install dependencies**

```bash
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

3. **Run the Streamlit app**

```bash
streamlit run app.py
```

4. **Allow camera access** if using webcam input.

---

## **Data Storage**

* Mood data is stored in:
  `data/mood_history.csv` with columns: `timestamp, source, emotion, score`.
* Source indicates input type: `text`, `webcam`, or `manual`.

---

## **Future Improvements**

* Add **speech-based emotion detection**.
* Implement **team-wide dashboards with visual insights**.
* Integrate **personalized task recommendations** based on historical mood patterns.
* Introduce **hybrid mode** to compare automated detection vs. manual selection for model tuning.

---

## **License**

This project is released under the MIT License.

---

## **Acknowledgements**

* [Hugging Face Transformers](https://huggingface.co/transformers/) for emotion classification models.
* [FER](https://github.com/justinshenk/fer) library for facial expression recognition.
* [Streamlit](https://streamlit.io/) for creating the interactive dashboard.

```

