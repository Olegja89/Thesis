import cv2
import numpy as np
from ultralytics import YOLO
from preprocess import preprocess_frame, load_calibration_data, rescale_coordinates
from config import VIDEO_PATH, RECOGNITION_SIZE, DISPLAY_SIZE, MAPPING_FILE
from data_export import CSVExporter
from coordinate_transformer import (
    CoordinateTransformer,
    calculate_real_world_coordinates,
    calculate_real_box_width  # <-- NEW IMPORT
)
from speed_utils import SpeedTracker
from visualization_utils import draw_annotations

def main():
    # Load the YOLOv8 model
    model = YOLO("best.pt")
    model.to("cuda")
    print(f"Using device: {model.device}")
    
    # Load calibration data
    K, D, DIM = load_calibration_data()
    if K is None or D is None or DIM is None:
        print("Failed to load calibration data. Exiting.")
        return

    # Initialize coordinate transformer and speed tracker
    transformer = CoordinateTransformer(MAPPING_FILE)
    speed_tracker = SpeedTracker()

    # Open the video file
    cap = cv2.VideoCapture(VIDEO_PATH)

    # Get video FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print(f"Warning: Invalid FPS ({fps}), defaulting to 30")
        fps = 30.0

    # Modify the CSV headers: drop 'height' and add 'real_width'
    tracking_header = ['frame', 'id', 'x', 'y', 'width', 'real_width']
    for i in range(10):  # 10 keypoints, if needed
        tracking_header.extend([f'kp{i}_x', f'kp{i}_y', f'kp{i}_conf'])
    tracking_exporter = CSVExporter('tracking_data.csv', tracking_header)

    world_coord_header = ['frame', 'id', 'world_x', 'world_y', 'speed_kmh']
    world_coord_exporter = CSVExporter('world_coordinates.csv', world_coord_header)

    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        
        # Preprocess the frame
        recognition_frame, display_frame = preprocess_frame(
            frame, K, D, DIM, RECOGNITION_SIZE, DISPLAY_SIZE
        )

        # Run YOLOv8 tracking
        results = model.track(recognition_frame, persist=True)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            keypoints = results[0].keypoints.data.cpu().numpy()

            # Rescale boxes and keypoints to display size
            scaled_boxes = [
                rescale_coordinates(box.tolist(), RECOGNITION_SIZE, DISPLAY_SIZE) 
                for box in boxes
            ]
            scaled_keypoints = [
                [
                    rescale_coordinates(kp[:2], RECOGNITION_SIZE, DISPLAY_SIZE) + [kp[2]]
                    if len(kp) == 3 and kp[2] > 0 else [0, 0, 0]
                    for kp in obj_kps
                ]
                for obj_kps in keypoints
            ]

            # Calculate real-world coordinates (the "middle-bottom" point)
            real_world_coords = calculate_real_world_coordinates(scaled_boxes, transformer)

            # Calculate speeds using frame count and fps
            speeds = speed_tracker.get_speeds(track_ids, real_world_coords, frame_count, fps)

            # For each object, compute real_width and export data
            for (box, track_id, kps, world_coord, speed) in zip(
                scaled_boxes, track_ids, scaled_keypoints, real_world_coords, speeds
            ):
                x, y, w, h = box

                # Calculate the real-world width between bottom-left and bottom-right corners
                real_width = calculate_real_box_width(box, transformer)

                # Write tracking data: [frame, id, x, y, width, real_width, <keypoints>...]
                row = [frame_count, track_id, x, y, w, real_width]
                for kp in kps:
                    row.extend(kp)
                tracking_exporter.write_row(row)

                # Write real-world coords and speeds (already what you had)
                world_coord_exporter.write_row([
                    frame_count,
                    track_id,
                    world_coord[0],  # "middle-bottom" real x
                    world_coord[1],  # "middle-bottom" real y
                    speed
                ])

            # Draw annotations with speeds
            annotated_frame = draw_annotations(display_frame.copy(), scaled_boxes, scaled_keypoints, track_ids, speeds)
        else:
            annotated_frame = display_frame

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    tracking_exporter.close()
    world_coord_exporter.close()

if __name__ == "__main__":
    main()