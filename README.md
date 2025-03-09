# Vehicle Tracking and Speed Analysis System

This project implements a real-time vehicle tracking system using YOLOv8 object detection and custom coordinate mapping. The system can detect vehicles, track their movements, calculate their real-world speeds, and export tracking data for further analysis.

## Features

- Real-time vehicle detection and tracking using YOLOv8
- Camera lens distortion correction
- Real-world coordinate mapping
- Real-time speed calculation (km/h)
- Data export in CSV format
- Speed visualization for each tracked vehicle
- Post-processing speed analysis tools

## Project Structure

```
vehicle-tracking-project/
├── src/
│   ├── main.py                   # Main tracking script
│   ├── preprocess.py             # Frame preprocessing utilities
│   ├── speed_utils.py            # Speed calculation utilities
│   ├── visualization_utils.py     # Visualization functions
│   ├── coordinate_transformer.py  # Coordinate mapping utilities
│   ├── config.py                 # Configuration parameters
│   └── data_export.py            # Data export utilities
├── data/
│   └── gopro_calibration.npz     # Camera calibration data
├── models/
│   └── best.pt                   # YOLOv8 trained model
├── requirements.txt
└── README.md
```

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- GoPro camera calibration data
- YOLOv8 trained model

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/vehicle-tracking-project.git
cd vehicle-tracking-project
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Configuration

1. Place your trained YOLOv8 model (`best.pt`) in the `models/` directory
2. Place your camera calibration data (`gopro_calibration.npz`) in the `data/` directory
3. Update `config.py` with your specific settings:
   - Video input path
   - Frame sizes
   - Output paths
   - Calibration file path

## Usage
### 1. Calibrate the camera with GoPro_fisheye_calibration.py to get .npz file
### 2. Set up coordinate mapping

Run the coordinate mapping script to establish real-world coordinates:
```bash
python src/coordinate_mapping.py
```
Follow the on-screen instructions to select points and enter their real-world coordinates.

### 3. Run the main tracking script

Execute the main tracking script:
```bash
python src/main.py
```
This will:
- Process the input video
- Track vehicles
- Calculate speeds
- Export tracking data to tracking_data.csv
- Display real-time visualization

### 4. Analyze the data
Run car_tracking.py, it will use tracking_data.csv as input.
It will ask you to write a number of a vehicle of interest. The numbers are visible during the run of main.py.
When it asks for the number of frames, press enter to export all available frames.
It will export file named "car_###_transformed.csv" where ### is car number.

### 5. Estimate the car size
Run calculation_model_2points.py to estimate the size of the car. Replace the name of the .csv file in the script.

