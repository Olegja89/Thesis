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

### 1. Set up coordinate mapping

Run the coordinate mapping script to establish real-world coordinates:
```bash
python src/coordinate_mapping.py
```
Follow the on-screen instructions to select points and enter their real-world coordinates.

### 2. Run the main tracking script

Execute the main tracking script:
```bash
python src/main.py
```
This will:
- Process the input video
- Track vehicles
- Calculate speeds
- Export tracking data to CSV files
- Display real-time visualization

### 3. Analyze speed data

Run the speed analysis script to visualize speed patterns:
```bash
python src/speed_analysis.py
```
Follow the prompts to:
- Plot speed graphs for specific vehicles
- View vehicle trajectories
- Analyze speed patterns

## Output Files

- `tracking_data.csv`: Contains frame-by-frame tracking data including positions and keypoints
- `world_coordinates.csv`: Contains real-world coordinates and speed data
- `coordinate_mapping.json`: Stores the coordinate transformation data


## Contact

Your Name - your.email@example.com
Project Link: [https://github.com/your-username/vehicle-tracking-project](https://github.com/your-username/vehicle-tracking-project)
