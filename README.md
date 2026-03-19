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

## Deploy on Streamlit Community Cloud

1. Push this project to a public GitHub repository.
2. Go to `https://share.streamlit.io/`
3. Click **New app**
4. Select your repository: `khushibishnoi2707-blip/Emotion-Based-Song-Recommendation-System`
5. Set the main file path to `app.py`
6. Click **Deploy**

After deployment, Streamlit will generate a live link for your app.

## Project Files

- `app.py` - Streamlit application
- `model.h5` - trained emotion detection model weights
- `muse_v3.csv` - song dataset
- `requirements.txt` - Python dependencies

## Notes

- The cloud version uses browser camera capture or image upload instead of direct desktop webcam access.
- If no face is detected, try a brighter and clearer front-facing photo.
