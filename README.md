# Deepfake Detection using Deep Learning (ResNext and LSTM)

## 1. Introduction
This project aims at detection of video deepfakes using deep learning techniques like ResNext and LSTM. Deepfake detection is achieved by using transfer learning where the pretrained ResNext CNN is used to obtain a feature vector, further the LSTM layer is trained using the features.

## 2. Directory Structure
For ease of understanding the project is structured in below format
```
Deepfake_detection_using_deep_learning
    |
    |--- Django Application
    |--- Model Creation
    |--- Documentation
    |--- Desktop
```
1. Django Application 
   - This directory consists of the Django made application of our work. Where a user can upload the video and submit it to the model for prediction. The trained model performs the prediction and the result is displayed on the screen.
2. Model Creation
   - This directory consists of the step by step process of creating and training a deepfake detection model using our approach.
3. Documentation
   - This directory consists of all the documentation done during the project.
4. Desktop
   - This directory contains a standalone desktop application for deepfake detection.
   
## 3. System Architecture
The system uses a combination of:
- **ResNext50** - A pretrained CNN for feature extraction from video frames
- **LSTM** - For temporal analysis of sequential frame features
- **Face Detection** - To crop and focus on facial regions

## 4. Our Results

| Model Name | No of videos | No of Frames | Accuracy |
|------------|--------------|--------------|----------|
|model_84_acc_10_frames_final_data.pt |6000 |10 |84.21461|
|model_87_acc_20_frames_final_data.pt | 6000 |20 |87.79160|
|model_89_acc_40_frames_final_data.pt | 6000| 40 |89.34681|
|model_90_acc_60_frames_final_data.pt | 6000| 60 |90.59097 |
|model_91_acc_80_frames_final_data.pt | 6000 | 80 | 91.49818 |
|model_93_acc_100_frames_final_data.pt| 6000 | 100 | 93.58794|

## 5. Getting Started

### Prerequisites
- Python >= 3.6
- CUDA compatible GPU (recommended)
- Required Python packages (see requirements.txt in Django Application folder)
- Trained Models:
  - You can download [trained models](https://drive.google.com/drive/folders/1UX8jXUXyEjhLLZ38tcgOwGsZ6XFSLDJ-?usp=sharing) and run the predict file for prediction.

### Running the Web Application
1. Open a terminal in the project root.
2. Move into the Django web app folder:
   ```bash
   cd "Django Application"
   ```
3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Make sure the trained model files are available in the `models/` folder:
   - `model_90_acc_20_frames_FF_data.pt` for videos
   - `model_90_acc_60_frames_final_data.pt` for images
5. Optional: add Gemini settings in a `.env` file:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ENABLE_GEMINI_REVIEW=true
   ```
6. Start the Django web server:
   ```bash
   python manage.py runserver
   ```

