import cv2
import numpy as np
from preprocess import preprocess_frame, load_calibration_data
import json

# Global variables
image = None
points = []
real_world_coords = []
display_size = (1280, 720)  # You can adjust this

def click_event(event, x, y, flags, param):
    global image, points
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        points.append([x, y])
        cv2.imshow("Image", image)
        
        # Prompt for real-world coordinates
        print(f"Clicked point at ({x}, {y})")
        real_x = float(input("Enter real-world X coordinate: "))
        real_y = float(input("Enter real-world Y coordinate: "))
        real_world_coords.append([real_x, real_y])
        
        print(f"Point {len(points)} added. Real-world coordinates: ({real_x}, {real_y})")
        
        if len(points) >= 4:
            print("You have selected 4 or more points. Press 'q' to finish and compute the transformation.")

def compute_transformation(src_points, dst_points):
    src_points = np.array(src_points, dtype=np.float32)
    dst_points = np.array(dst_points, dtype=np.float32)
    return cv2.findHomography(src_points, dst_points)

def main():
    global image

    # Load the first frame from the video
    video = cv2.VideoCapture("video.mp4")
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

    cv2.imshow("Image", image)
    cv2.setMouseCallback("Image", click_event)

    print("Click on at least 4 points in the image and enter their real-world coordinates.")
    print("Press 'q' to finish selecting points and compute the transformation.")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

    if len(points) < 4:
        print("Not enough points selected. At least 4 points are required.")
        return

    # Compute the transformation matrix
    matrix, _ = compute_transformation(points, real_world_coords)

    # Save the transformation matrix and points
    data = {
        "transformation_matrix": matrix.tolist(),
        "image_points": points,
        "real_world_points": real_world_coords
    }

    with open("coordinate_mapping.json", "w") as f:
        json.dump(data, f, indent=4)

    print("Coordinate mapping data saved to 'coordinate_mapping.json'")

if __name__ == "__main__":
    main()