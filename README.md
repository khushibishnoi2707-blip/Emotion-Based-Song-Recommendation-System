# Emotion-Based Song Recommendation System

An end-to-end music recommendation web application that detects facial emotion from an image and suggests songs that match the detected mood. The project combines computer vision, deep learning, and a lightweight Streamlit interface to deliver a simple interactive user experience.

## Live Demo

Application URL: `https://emotion-based-song-recommendation-4tvs.onrender.com`

## Overview

This project analyzes a face image captured through the browser camera or uploaded by the user, predicts the dominant facial emotion using a trained TensorFlow model, and recommends songs from a curated dataset. Each recommendation opens as a public Spotify track link for quick listening.

The application is designed to be easy to run locally and easy to deploy on Render.

## Key Features

- Emotion detection from a live browser camera capture
- Emotion detection from uploaded image files
- TensorFlow-based facial emotion classification
- Song recommendation based on the detected mood
- Public Spotify links for recommended tracks
- Streamlit-based interface with a clean two-column layout
- Render-ready deployment configuration included in the repository

## How It Works

1. The user captures a photo using the browser camera or uploads an image.
2. The app converts the image to grayscale and detects the face using OpenCV Haar cascades.
3. The detected face is resized to `48 x 48` and passed to the trained CNN model.
4. The model predicts one of the supported emotion classes.
5. The detected emotion is mapped to a section of the music dataset.
6. The app samples songs from the matching bucket and displays them in the interface.
7. Clicking a recommendation opens the corresponding public Spotify track page.

## Supported Emotions

The model is configured to classify the following emotions:

- Angry
- Disgusted
- Fearful
- Happy
- Neutral
- Sad
- Surprised

## Tech Stack

- Python
- Streamlit
- TensorFlow / Keras
- OpenCV
- NumPy
- Pandas
- Pillow
- Render

## Project Structure

```text
Emotion-Based-Song-Recommendation-System/
|-- app.py
|-- model.h5
|-- muse_v3.csv
|-- requirements.txt
|-- .python-version
|-- render.yaml
`-- README.md
```

## File Description

- `app.py`
  Main Streamlit application. Handles image input, face detection, emotion prediction, song recommendation, and UI rendering.

- `model.h5`
  Pretrained model weights used for facial emotion prediction.

- `muse_v3.csv`
  Song dataset used to generate recommendations. It includes metadata such as track name, artist name, emotion-related scores, and Spotify track IDs.

- `requirements.txt`
  Python dependencies required to run the application.

- `.python-version`
  Pins the Python runtime to a TensorFlow-compatible version for Render deployment.

- `render.yaml`
  Render blueprint configuration for one-click deployment.

## Installation

### Prerequisites

- Python `3.12`
- `pip`

### Clone the Repository

```bash
git clone https://github.com/khushibishnoi2707-blip/Emotion-Based-Song-Recommendation-System.git
cd Emotion-Based-Song-Recommendation-System
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Locally

Start the Streamlit development server:

```bash
streamlit run app.py
```

After the server starts, open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Deployment on Render

This repository already includes the required Render deployment configuration.

### Steps

1. Push the repository to GitHub.
2. Open `https://dashboard.render.com/`
3. Sign in with your GitHub account.
4. Click `New +`
5. Choose `Blueprint`
6. Select the repository `khushibishnoi2707-blip/Emotion-Based-Song-Recommendation-System`
7. Render will detect the `render.yaml` file automatically.
8. Click `Apply` to create and deploy the web service.

### Render Configuration

- Build command:

```bash
pip install -r requirements.txt
```

- Start command:

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

## Important Deployment Notes

- Render free web services may spin down after inactivity.
- The application uses `.python-version` to ensure a TensorFlow-compatible Python version during deployment.
- The hosted app uses browser-based camera input instead of direct desktop webcam access.

## Recommendation Logic

The app sorts the music dataset using emotional and pleasantness-related columns and then splits the dataset into fixed emotion buckets. Based on the detected emotion, songs are sampled from the corresponding bucket and displayed as recommendations.

This is a lightweight rule-based recommendation layer on top of the emotion classifier rather than a collaborative filtering system.

## Known Limitations

- Recommendation quality depends on the dataset structure and the current bucket-based sampling logic.
- If no face is detected in the image, the app cannot generate emotion-based recommendations.
- The current version predicts a single dominant emotion from one image.
- Render free plan deployments may experience cold starts.

## Future Improvements

- Improve recommendation logic using similarity scores or hybrid recommendation methods
- Add support for multiple face detection and multi-user scenes
- Show confidence scores for the predicted emotion
- Add genre filtering and language filtering
- Save recommendation history for users
- Add better error handling for missing or invalid Spotify IDs

## Troubleshooting

### No face detected

Try using:

- a brighter image
- a front-facing image
- an image with a clearly visible face

### Song link opens an API error

This issue has been fixed in the current version by generating public Spotify links from `spotify_id` values instead of using protected API endpoints.

### Render build fails

Check the Render logs for:

- Python version mismatch
- TensorFlow installation errors
- dependency build failures

The included `.python-version` file is important for successful Render deployment.

## Repository

GitHub repository: `https://github.com/khushibishnoi2707-blip/Emotion-Based-Song-Recommendation-System`

## Author

Khushi Bishnoi

## License

This project is intended for educational and portfolio use. Add a formal license file if you want to define reuse permissions explicitly.
