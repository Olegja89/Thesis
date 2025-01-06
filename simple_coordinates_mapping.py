import cv2
import numpy as np
from preprocess import preprocess_frame, load_calibration_data
import json
from config import VIDEO_PATH, DISPLAY_SIZE  # RECOGNITION_SIZE, MAPPING_FILE, etc. if needed

# Global variables
image = None
points = []
real_world_coords = []
display_size = (1920, 1080)  # Adjust if you need a different display size

def click_event(event, x, y, flags, param):
    global image, points, real_world_coords
    if event == cv2.EVENT_LBUTTONDOWN:
        # Draw a small circle where the user clicked
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        points.append([x, y])
        cv2.imshow("Image", image)
        
        # Prompt for real-world coordinates
        print(f"Clicked point at ({x}, {y})")
        real_x = float(input("Enter real-world X coordinate: "))
        real_y = float(input("Enter real-world Y coordinate: "))
        real_world_coords.append([real_x, real_y])
        
        print(f"Point {len(points)} added. Real-world coordinates: ({real_x}, {real_y})")
        
        if len(points) >= 3:
            print("You have selected 3 (or more) points. Press 'q' to finish and compute the transformation.")

def compute_transformation(src_points, dst_points):
    """
    Returns a 3x3 matrix (for consistency) and a dummy mask, 
    mimicking the behavior of cv2.findHomography’s return signature.
    """
    # Convert input to float32 arrays
    src_points = np.array(src_points[:3], dtype=np.float32)  # Just take first 3 if user gave more
    dst_points = np.array(dst_points[:3], dtype=np.float32)
    
    # getAffineTransform() returns a 2x3 matrix
    affine_2x3 = cv2.getAffineTransform(src_points, dst_points)
    
    # Convert 2x3 into a 3x3 by appending [0,0,1]
    affine_3x3 = np.vstack([affine_2x3, [0, 0, 1]])
    
    # Return (matrix, mask) to match findHomography’s style
    return affine_3x3, None

def main():
    global image

    # 1. Load the first frame from the video
    video = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = video.read()
    video.release()

    if not ret:
        print("Failed to read the video file.")
        return

    # 2. Load calibration data (if you have it set up; otherwise, omit)
    K, D, DIM = load_calibration_data()
    if K is None or D is None or DIM is None:
        print("Failed to load calibration data. Exiting.")
        return
    
    # 3. Preprocess the frame (e.g., undistort if needed)
    recognition_frame, _ = preprocess_frame(frame, K, D, DIM)
    
    # 4. Resize the frame to display size
    image = cv2.resize(recognition_frame, display_size, interpolation=cv2.INTER_AREA)

    # 5. Set up the mouse callback
    cv2.imshow("Image", image)
    cv2.setMouseCallback("Image", click_event)

    print("Click on exactly 3 points in the image and enter their real-world coordinates.")
    print("Press 'q' to finish selecting points and compute the transformation.")
    
    # 6. Wait for user to press 'q'
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

    # 7. Validate number of points
    if len(points) < 3:
        print("Not enough points selected. Exactly 3 points are required.")
        return
    
    # (Optional) If user selected more than 3 by mistake, we just take the first 3
    if len(points) > 3:
        print("More than 3 points selected. Will use the first 3 points only.")
    
    # 8. Compute the transformation matrix (affine -> 3x3)
    matrix, _ = compute_transformation(points, real_world_coords)

    # 9. Save the transformation matrix and points
    #    This matches your original JSON structure exactly.
    data = {
        "transformation_matrix": matrix.tolist(),
        "image_points": points,              # you may store all points
        "real_world_points": real_world_coords
    }

    # 10. Export the same filename "coordinate_mapping.json"
    with open("coordinate_mapping.json", "w") as f:
        json.dump(data, f, indent=4)

    print("Coordinate mapping data saved to 'coordinate_mapping.json'")

if __name__ == "__main__":
    main()