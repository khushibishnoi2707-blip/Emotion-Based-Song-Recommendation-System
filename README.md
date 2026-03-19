# Emotion-Based Song Recommendation System

This project is a Streamlit web app that detects emotion from a face image and recommends songs based on the detected mood.

## Features

- Detects facial emotion from a browser camera capture or uploaded image
- Uses a TensorFlow emotion-classification model from `model.h5`
- Recommends songs from the bundled dataset
- Opens recommended tracks as public Spotify links
- Ready to deploy on Render

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
5. Select the repository `khushibishnoi2707-blip/Emotion-Based-Song-Recommendation-System`
6. Render will detect `render.yaml`
7. Click **Apply**

After deployment, Render will generate a public live link.

## Important Note About Render

- The free Render plan may spin down after inactivity.
- If you want the app to stay always on, upgrade the service to a paid plan.

## Project Files

- `app.py` - main Streamlit application
- `model.h5` - trained model weights for emotion detection
- `muse_v3.csv` - dataset used for song recommendations
- `requirements.txt` - Python dependencies
- `.python-version` - pins a TensorFlow-compatible Python version for Render
- `render.yaml` - Render deployment configuration

## Notes

- The hosted version uses browser camera input or image upload instead of direct desktop webcam access.
- If no face is detected, try a brighter and clearer front-facing photo.
- Song links are generated from Spotify track IDs, so they open public Spotify track pages instead of private API endpoints.
