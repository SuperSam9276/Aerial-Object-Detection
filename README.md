# 🛸 Aerial Object Detection — Bird vs Drone

Deep learning pipeline for classifying and detecting birds and drones in aerial imagery.

## Project Structure

```
aerial_detection/
├── 01_data_exploration.ipynb    # Phase 1 & 2: EDA + Augmentation preview
├── 02_model_training.ipynb      # Phase 3 & 4: Custom CNN + Transfer Learning
├── 03_evaluation.ipynb          # Phase 5: Confusion matrix, ROC, comparison
├── 04_yolov8.ipynb              # Phase 6: YOLOv8 object detection (optional)
├── app.py                       # Phase 7: Streamlit web app
├── requirements.txt
├── classification_dataset/      # ← YOUR DATASET HERE
│   ├── TRAIN/bird/  TRAIN/drone/
│   ├── VALID/bird/  VALID/drone/
│   └── TEST/bird/   TEST/drone/
└── object_detection_Dataset/    # ← YOUR YOLO DATASET HERE
    ├── train/images/ train/labels/
    ├── valid/images/ valid/labels/
    └── test/images/  test/labels/
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place your datasets
- Put `classification_dataset/` folder in project root
- Put `object_detection_Dataset/` folder in project root

### 3. Run notebooks in order
```
01_data_exploration.ipynb   → Explore data, see augmentations
02_model_training.ipynb     → Train Custom CNN + EfficientNetB0
03_evaluation.ipynb         → Evaluate, compare, save best_model.h5
04_yolov8.ipynb             → Train YOLOv8, save yolo_best.pt
```

### 4. Launch Streamlit app
```bash
streamlit run app.py
```

## Models

| Model               | Type              | Input   | Output          |
|---------------------|-------------------|---------|-----------------|
| Custom CNN          | 4-block CNN       | 224×224 | Bird/Drone prob |
| EfficientNetB0 (TL) | Transfer learning | 224×224 | Bird/Drone prob |
| YOLOv8n             | Object detection  | 640×640 | Bounding boxes  |

## Dataset

- **Classification**: 2662 train / 442 val / 215 test (Bird + Drone, .jpg)
- **Detection**: 3319 images with YOLO-format `.txt` labels

## Tech Stack

- TensorFlow 2.x / Keras
- Ultralytics YOLOv8
- Streamlit
- scikit-learn, matplotlib, seaborn
