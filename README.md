# Navinex – AI-Powered Smart Assistive Navigation System

**An intelligent computer vision-based navigation assistant designed to help visually impaired individuals understand and navigate their surroundings.**

## About the Project

Navinex is an AI-powered assistive navigation system that uses computer vision, depth estimation, semantic segmentation, and voice-based interaction to interpret the surrounding environment.

The system processes visual information to identify objects, estimate their relative depth, understand walkable areas, and provide navigation-related guidance through voice feedback.

## Key Features

- **Real-Time Object Detection:** Identifies surrounding objects using YOLOv8.
- **Depth Estimation:** Uses MiDaS to estimate the relative depth of objects and obstacles.
- **Custom Object Detection:** Supports custom-trained YOLO model weights.
- **Semantic Segmentation:** Uses SegFormer to identify walkable areas.
- **Object Tracking:** Tracks detected objects across video frames.
- **Voice Assistance:** Provides audio-based navigation guidance.
- **Speech Recognition:** Integrates Vosk for voice input.
- **Navigation Guidance:** Combines visual information to support obstacle awareness and navigation.

## Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Object Detection | YOLOv8 |
| Depth Estimation | MiDaS |
| Semantic Segmentation | SegFormer |
| Computer Vision | OpenCV |
| Object Tracking | Deep SORT |
| Speech Recognition | Vosk |
| Machine Learning | PyTorch |
| Development Environment | VS Code |

## Project Structure

```text
Navinex/
│
├── app.py
├── main.py
├── config.py
├── requirements.txt
├── .gitignore
│
├── models/
│   ├── best_v2.pt
│   ├── yolov8n.pt
│   ├── dpt_swin2_tiny_256.pt
│   ├── segformer_best/
│   └── vosk-model-small-en-us-0.15/
│
├── src/
│   ├── depth/
│   ├── detection/
│   ├── feature_logger.py
│   ├── guidance.py
│   ├── navigation.py
│   ├── segmentation.py
│   ├── utils.py
│   ├── visualizer.py
│   └── voice.py
│
└── test_navigation_smoothing.py
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/sree-lekshmips/NaviNex.git
cd NaviNex
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download model files

The repository uses Git LFS for `.pt` model weights.

Install Git LFS and pull the model files:

```bash
git lfs install
git lfs pull
```

Ensure all required model files are available in the `models/` directory.

## Running the Project

The system uses a webcam for real-time visual input.

See the application entry point and configuration for the appropriate execution command.

## Project Objective

The primary objective of Navinex is to explore how artificial intelligence and computer vision can contribute to assistive technology by improving environmental awareness and supporting independent navigation for visually impaired individuals.

## Future Enhancements

- Improve real-time detection and tracking performance.
- Enhance obstacle distance estimation.
- Improve navigation guidance accuracy.
- Expand voice interaction capabilities.
- Optimize the system for portable deployment.

## Developer

**Sreelekshmi P S**

Integrated MSc Computer Science – Data Science

Nirmala College, Muvattupuzha, Kerala

---

*Navinex – Making surroundings more understandable through AI.*
