# Multimodal Music Classification

This repository contains the implementation of a **Multimodal Music Classification** system, designed to preserve and promote the rich musical heritage of Nepal. The project utilizes **audio and video data** with deep learning techniques to classify indigenous Nepali music genres accurately.

## Features
- **Multimodal Approach**: Combines audio and video data for enhanced classification accuracy.
- **Audio Analysis**: Utilizes **VGGNet** for extracting audio features from Mel spectrograms.
- **Video Analysis**: Employs **Res3D** for video-based feature extraction and classification.
- **Ensemble Method**: Combines predictions from audio and video models to improve performance.
- **Extensive Dataset**: Classifies 35 indigenous Nepali music genres.

## Motivation
This project aims to:
1. Preserve Nepali musical heritage by automating genre classification.
2. Promote indigenous music globally through technological innovation.
3. Explore the potential of multimodal deep learning for cultural preservation.

## Methodology
1. **Data Collection**: 
   - **Audio**: Gathered from platforms like YouTube and Spotify and converted into Mel spectrograms.
   - **Video**: Extracted features from collected video data using Res3D.
2. **Model Design**:
   - **VGGNet**: Used for audio feature extraction and classification.
   - **Res3D**: Used for video-based feature extraction and classification.
3. **Ensemble Method**:
   - Combined predictions from the audio (VGGNet) and video (Res3D) models to achieve final classification.
4. **Evaluation**:
   - Metrics: Accuracy and F1-score.

## Installation
1. Clone the repository:
   ```bash
  https://github.com/Melina-Singh/Multimodel_Music_Classification.git

2. Install dependencies:
    ```bash
    pip install -r requirements.txt


## Usage
3. Audio Classification: Run the main.py file for audio processing and classification:
    ```bash
    python audio_main.py

4. Video Classification: Run the main.py file for video processing and classification:
    ```bash
    python video_main.py


Let me know if you want to include additional technical details, results, or visualizations!



