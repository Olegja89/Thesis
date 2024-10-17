import cv2
import csv
import numpy as np
from ultralytics import YOLO
from preprocess import preprocess_frame, load_calibration_data, rescale_coordinates

def draw_annotations(image, boxes, keypoints, track_ids):
    for box, kps, track_id in zip(boxes, keypoints, track_ids):
        x, y, w, h = box
        # Draw bounding box
        cv2.rectangle(image, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), (0, 255, 0), 2)
        
        # Draw track ID
        cv2.putText(image, f"ID: {track_id}", (int(x - w/2), int(y - h/2 - 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw keypoints
        for kp in kps:
            kp_x, kp_y, kp_conf = kp
            if kp_conf > 0:
                cv2.circle(image, (int(kp_x), int(kp_y)), 5, (255, 0, 0), -1)
    
    return image

# Load the YOLOv8 model
model = YOLO("best.pt")

# Load calibration data
K, D, DIM = load_calibration_data()
if K is None or D is None or DIM is None:
    print("Failed to load calibration data. Exiting.")
    exit(1)

# Open the video file
video_path = "video_2.mp4"
cap = cv2.VideoCapture(video_path)

# Define the frame sizes
recognition_size = (640, 640)
display_size = (1280, 720)  # Adjust this to your preferred display size

# Prepare CSV file for tracking data
csv_file = open('tracking_data.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)

# Prepare the header row for CSV
header = ['frame', 'id', 'x', 'y', 'width', 'height']
for i in range(10):  # 10 keypoints
    header.extend([f'kp{i}_x', f'kp{i}_y', f'kp{i}_conf'])
csv_writer.writerow(header)

frame_count = 0

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        frame_count += 1
        
        # Preprocess the frame (undistort and resize)
        recognition_frame, display_frame = preprocess_frame(frame, K, D, DIM, recognition_size, display_size)

        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(recognition_frame, persist=True)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            keypoints = results[0].keypoints.data.cpu().numpy()

            # Rescale boxes and keypoints to display size
            scaled_boxes = []
            scaled_keypoints = []

            for box, kps in zip(boxes, keypoints):
                scaled_box = rescale_coordinates(box.tolist(), recognition_size, display_size)
                scaled_boxes.append(scaled_box)

                scaled_kps = []
                for kp in kps:
                    if len(kp) == 3:
                        kp_x, kp_y, kp_conf = kp
                        if kp_conf > 0:
                            kp_x, kp_y = rescale_coordinates([kp_x, kp_y], recognition_size, display_size)
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
                csv_writer.writerow(row)

            # Draw annotations on the display frame
            annotated_frame = draw_annotations(display_frame.copy(), scaled_boxes, scaled_keypoints, track_ids)
        else:
            annotated_frame = display_frame

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
csv_file.close()