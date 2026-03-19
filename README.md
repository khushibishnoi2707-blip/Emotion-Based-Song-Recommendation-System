# Emotion-Based Song Recommendation System

This is a Streamlit app that detects the emotion in a face photo and recommends songs from the bundled dataset based on the detected mood.

## Features

- Detects facial emotion from a camera capture or uploaded image
- Uses a trained TensorFlow model stored in `model.h5`
- Recommends songs with artist names and clickable links
- Ready for local run and Streamlit Community Cloud deployment

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the app:

```bash
streamlit run app.py
```

## Deploy on Render

1. Open `https://dashboard.render.com/`
2. Sign in with GitHub
3. Click **New +**
4. Choose **Blueprint**
5. Select repository: `khushibishnoi2707-blip/Emotion-Based-Song-Recommendation-System`
6. Render will detect `render.yaml`
7. Click **Apply**

After deployment, Render will generate a live link for your app.

Note:

- The free Render plan may spin down after inactivity.
- For an always-on app, switch the service plan in Render to a paid instance.

## Project Files

- `app.py` - Streamlit application
- `model.h5` - trained emotion detection model weights
- `muse_v3.csv` - song dataset
- `requirements.txt` - Python dependencies
- `render.yaml` - Render deployment configuration

## Notes

- The cloud version uses browser camera capture or image upload instead of direct desktop webcam access.
- If no face is detected, try a brighter and clearer front-facing photo.
