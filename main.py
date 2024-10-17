import cv2
import numpy as np
from ultralytics import YOLO
from preprocess import preprocess_frame, load_calibration_data, rescale_coordinates
from utils import draw_annotations
from config import VIDEO_PATH, RECOGNITION_SIZE, DISPLAY_SIZE
from data_export import CSVExporter

def main():
    # Load the YOLOv8 model
    model = YOLO("best.pt")

    # Load calibration data
    K, D, DIM = load_calibration_data()
    if K is None or D is None or DIM is None:
        print("Failed to load calibration data. Exiting.")
        return

    # Open the video file
    cap = cv2.VideoCapture(VIDEO_PATH)

    # Prepare CSV file for tracking data
    csv_exporter = CSVExporter('tracking_data.csv')

    frame_count = 0

    # Main loop
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        
        # Preprocess the frame
        recognition_frame, display_frame = preprocess_frame(frame, K, D, DIM, RECOGNITION_SIZE, DISPLAY_SIZE)

        # Run YOLOv8 tracking
        results = model.track(recognition_frame, persist=True)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            keypoints = results[0].keypoints.data.cpu().numpy()

            # Rescale boxes and keypoints to display size
            scaled_boxes = []
            scaled_keypoints = []

            for box, kps in zip(boxes, keypoints):
                scaled_box = rescale_coordinates(box.tolist(), RECOGNITION_SIZE, DISPLAY_SIZE)
                scaled_boxes.append(scaled_box)

                scaled_kps = []
                for kp in kps:
                    if len(kp) == 3:
                        kp_x, kp_y, kp_conf = kp
                        if kp_conf > 0:
                            kp_x, kp_y = rescale_coordinates([kp_x, kp_y], RECOGNITION_SIZE, DISPLAY_SIZE)
                        else:
                            kp_x, kp_y = 0, 0
                    else:
                        kp_x, kp_y, kp_conf = 0, 0, 0
                    scaled_kps.append([kp_x, kp_y, kp_conf])
                scaled_keypoints.append(scaled_kps)

            # Write tracking data to CSV
            for box, track_id, kps in zip(scaled_boxes, track_ids, scaled_keypoints):
                x, y, w, h = box
                row = [frame_count, track_id, x, y, w, h]
                for kp in kps:
                    row.extend(kp)
                csv_exporter.write_row(row)

            # Draw annotations
            annotated_frame = draw_annotations(display_frame.copy(), scaled_boxes, scaled_keypoints, track_ids)
        else:
            annotated_frame = display_frame

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    csv_exporter.close()

if __name__ == "__main__":
    main()