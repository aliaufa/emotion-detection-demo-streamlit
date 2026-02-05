# Emotion Detection with YOLO

A real-time emotion detection system built with YOLOv8 and Streamlit. This project detects human emotions from images, videos, and live webcam feeds.

## Features

- Real-time emotion detection from webcam
- Image-based emotion detection
- Video emotion detection with tracking
- Interactive Streamlit web interface
- Comprehensive model evaluation metrics
- Support for multiple emotion classes

## Project Structure

```
emotion_detection/
├── app_demo/                          # Streamlit application
│   ├── streamlit_app.py              # Main application entry point
│   ├── landing.py                    # Landing page
│   ├── camera_infer.py               # Webcam inference
│   ├── image_infer.py                # Image inference
│   ├── video_infer.py                # Video inference
│   ├── requirements.txt              # Python dependencies
│   ├── model/                        # Trained models (download separately)
│   ├── temp_images/                  # Temporary image storage
│   └── temp_videos/                  # Temporary video storage
├── exploratory_data_analysis.ipynb   # EDA notebook
├── test_yolo_training.ipynb          # Model training notebook
├── model_evaluation.ipynb            # Model evaluation notebook
├── inference.ipynb                   # Inference examples notebook
├── .env                              # Environment variables (create from template)
├── .gitignore                        # Git ignore rules
└── README.md                         # Project documentation
```

**Note:** The following directories are excluded from version control:
- `emotion-8/` - Dataset files (download using Roboflow API)
- `model_eval_files/` - Evaluation results (generated during training)
- `runs/` and `runs_2/` - Training run outputs (generated during training)
- `inference_results/` - Inference outputs (generated during inference)

## Installation

### Prerequisites

- Python 3.8 or higher
- Webcam (for live detection)
- GPU (optional, for faster inference)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd emotion_detection
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies for the Streamlit app:
```bash
cd app_demo
pip install -r requirements.txt
```

Or install all dependencies manually:
```bash
pip install ultralytics streamlit opencv-python pillow numpy pandas matplotlib roboflow python-dotenv
```

4. Set up environment variables:
   - Create a `.env` file in the project root
   - Add your Roboflow API key: `ROBOFLOW_API_KEY=your_api_key_here`

5. Download the dataset (optional, for training):
   - Run the second cell in `test_yolo_training.ipynb` to download the dataset via Roboflow API
   - Or manually place your dataset in the `emotion-8/` directory

## Usage

### Running the Streamlit Application

Navigate to the app directory and run:

```bash
cd app_demo
streamlit run streamlit_app.py
```

**Note:** Ensure you have the trained model file (`best.pt`) in `app_demo/model/` directory before running the application.

The application provides three inference modes:

1. **Webcam Inference**: Real-time emotion detection from your camera
   - Single frame capture
   - Live continuous feed

2. **Image Inference**: Upload and analyze images
   - Supports JPG, PNG, JPEG formats
   - Displays detection results with confidence scores

3. **Video Inference**: Process video files
   - Upload MP4, AVI, MOV files
   - Frame-by-frame emotion tracking
   - Downloadable annotated results

### Training the Model

Open and run the training notebook:

```bash
jupyter notebook test_yolo_training.ipynb
```

The notebook includes:
- Dataset preparation
- Model configuration
- Training with validation
- Hyperparameter tuning

### Model Evaluation

Evaluate model performance using:

```bash
jupyter notebook model_evaluation.ipynb
```

Evaluation includes:
- Per-class metrics (precision, recall, F1-score)
- Confusion matrix
- Detailed predictions analysis
- Annotated test set visualizations

## Dataset

The project uses the emotion-8 dataset from Roboflow. To obtain the dataset:

1. Set up your `.env` file with your Roboflow API key
2. Run the dataset download cell in `test_yolo_training.ipynb`
3. The dataset will be downloaded to `emotion-8/` directory with:
   - Training set: `emotion-8/train/`
   - Validation set: `emotion-8/valid/`
   - Test set: `emotion-8/test/`
   - Configuration: `emotion-8/data.yaml`

**Note:** The dataset is not included in version control due to size.

## Model

- **Architecture**: YOLOv8
- **Task**: Object Detection (Emotion Detection)
- **Input**: RGB images
- **Output**: Bounding boxes with emotion class labels and confidence scores
- **Model Location**: `app_demo/model/best.pt`

## Performance

Model evaluation metrics are generated after running `model_evaluation.ipynb`:
- Per-class metrics (precision, recall, F1-score)
- Detailed predictions with confidence scores
- Annotated test images

**Note:** Evaluation files are generated locally and not included in version control.

## Notebooks

- **exploratory_data_analysis.ipynb**: Dataset exploration and visualization
- **test_yolo_training.ipynb**: Model training and validation
- **model_evaluation.ipynb**: Comprehensive model evaluation
- **inference.ipynb**: Inference examples and demonstrations

## Technical Details

### Dependencies

- ultralytics: YOLOv8 implementation
- streamlit: Web application framework
- opencv-python: Image and video processing
- pillow: Image manipulation
- numpy: Numerical operations
- pandas: Data analysis
- matplotlib: Visualization
- roboflow: Dataset management
- python-dotenv: Environment variable management

### Inference Configuration

- Default confidence threshold: 0.25
- Input resolution: Automatically adjusted
- Webcam resolution: 640x480 (default)

## Troubleshooting

### Webcam Issues

If webcam doesn't work:
1. Check camera permissions in system settings
2. Ensure no other application is using the camera
3. Try restarting the Streamlit application

### Model Loading Issues

If the model fails to load:
1. Verify `app_demo/model/best.pt` exists
2. Ensure the file is not corrupted
3. Check ultralytics installation

### Performance Issues

For better performance:
1. Use a GPU if available
2. Reduce input resolution
3. Adjust confidence threshold
4. Close unnecessary applications

## Acknowledgments

- YOLOv8 by Ultralytics
- Streamlit for the web framework
- Dataset contributors

## Contact

For questions or issues, please open an issue in the repository.
