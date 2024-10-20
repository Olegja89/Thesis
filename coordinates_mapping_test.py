import cv2
import numpy as np
import json
from preprocess import preprocess_frame, load_calibration_data

# Global variables
image = None
transformation_matrix = None
display_size = (1920, 1080)  # You can adjust this

def click_event(event, x, y, flags, param):
    global image, transformation_matrix
    if event == cv2.EVENT_LBUTTONDOWN:
        # Convert clicked point to real-world coordinates
        point = np.array([x, y, 1]).reshape(3, 1)
        real_world_point = np.dot(transformation_matrix, point)
        real_world_point = real_world_point / real_world_point[2]
        
        # Draw circle at clicked point
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        
        # Display real-world coordinates
        print(f"Image coordinates: ({x}, {y})")
        print(f"Real-world coordinates: ({real_world_point[0][0]:.2f}, {real_world_point[1][0]:.2f})")
        
        # Update image display
        cv2.imshow("Image", image)

def load_mapping_data(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return np.array(data['transformation_matrix'])

def main():
    global image, transformation_matrix

    # Load the first frame from the video
    video = cv2.VideoCapture("video_2.mp4")
    ret, frame = video.read()
    video.release()

    if not ret:
        print("Failed to read the video file.")
        return

    # Load calibration data
    K, D, DIM = load_calibration_data()
    if K is None or D is None or DIM is None:
        print("Failed to load calibration data. Exiting.")
        return

    # Preprocess the frame
    recognition_frame, _ = preprocess_frame(frame, K, D, DIM)
    
    # Resize the frame to display size
    image = cv2.resize(recognition_frame, display_size, interpolation=cv2.INTER_AREA)

    # Load the transformation matrix
    transformation_matrix = load_mapping_data("coordinate_mapping.json")

    cv2.imshow("Image", image)
    cv2.setMouseCallback("Image", click_event)

    print("Click on points in the image to see their real-world coordinates.")
    print("Press 'q' to quit.")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()