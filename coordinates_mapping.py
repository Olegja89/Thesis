import cv2
import numpy as np
import json

# Optional preprocessing dependencies
USE_PREPROCESSING = True  # Set to False to disable preprocessing
if USE_PREPROCESSING:
    from preprocess import preprocess_frame, load_calibration_data

# Configuration
IMAGE_PATH = "mapping.png"  # Update with your actual image path
DISPLAY_SIZE = (1920, 1080)  # Adjust as desired

# Global variables
image = None
points = []
real_world_coords = []

def click_event(event, x, y, flags, param):
    """
    Mouse callback to record clicked points and ask for their real-world coordinates.
    """
    global image, points, real_world_coords
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Draw a small circle at the clicked location
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
            print("You have selected 4 or more points. Press 'q' to finish whenever you're ready.")

def compute_transformation(src_points, dst_points):
    """
    Compute the homography using RANSAC for better robustness.
    """
    src_points = np.array(src_points, dtype=np.float32)
    dst_points = np.array(dst_points, dtype=np.float32)
    
    # Use RANSAC with a certain reprojection threshold (e.g., 3.0 pixels).
    H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 3.0)
    return H, mask

def main():
    global image, points, real_world_coords

    # Load the image
    frame = cv2.imread(IMAGE_PATH)
    if frame is None:
        print(f"Failed to load the image from {IMAGE_PATH}. Please check the path.")
        return

    if USE_PREPROCESSING:
        # Load calibration data (depends on your own implementation of 'load_calibration_data')
        K, D, DIM = load_calibration_data()
        if K is None or D is None or DIM is None:
            print("Failed to load calibration data. Exiting.")
            return

        # Preprocess the frame (undistort, etc.) - depends on your 'preprocess_frame' function
        frame, _ = preprocess_frame(frame, K, D, DIM)

    # Resize the frame for display
    image = cv2.resize(frame, DISPLAY_SIZE, interpolation=cv2.INTER_AREA)
    cv2.imshow("Image", image)
    
    # Set the mouse callback
    cv2.setMouseCallback("Image", click_event)

    print("Click on as many points (>=4) as you wish in the image and enter their real-world coordinates.")
    print("Press 'q' when you're done selecting points to compute the transformation.")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

    # Make sure we have at least 4 points
    if len(points) < 4:
        print("Not enough points selected. At least 4 points are required for a homography.")
        return

    # Compute the transformation matrix using RANSAC
    H, mask = compute_transformation(points, real_world_coords)

    if H is None:
        print("Homography could not be computed. Check your points or try again.")
        return

    print("Homography matrix:\n", H)
    
    # Optionally show how many inliers / outliers if you'd like
    # mask is an array of 0/1 indicating which points are inliers
    if mask is not None:
        inliers = np.sum(mask)
        total = len(mask)
        print(f"RANSAC inliers: {inliers}/{total}")

    # Save the transformation matrix and points
    data = {
        "transformation_matrix": H.tolist(),
        "image_points": points,
        "real_world_points": real_world_coords
    }

    with open("coordinate_mapping.json", "w") as f:
        json.dump(data, f, indent=4)

    print("Coordinate mapping data saved to 'coordinate_mapping.json'")

if __name__ == "__main__":
    main()