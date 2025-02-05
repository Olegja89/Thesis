import cv2
import numpy as np
import json

# Optional preprocessing dependencies
USE_PREPROCESSING = True  # Set to False to disable preprocessing
if USE_PREPROCESSING:
    from preprocess import preprocess_frame, load_calibration_data

# Configuration
IMAGE_PATH = "30kmph_mapping.png"
DISPLAY_SIZE = (1920, 1080)

def load_homography(json_path="coordinate_mapping.json"):
    """
    Loads the homography matrix from a JSON file, 
    assumed to be image->world if you used:
    findHomography(image_points, real_world_points)
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    matrix_list = data["transformation_matrix"]
    return np.array(matrix_list, dtype=np.float64)

def transform_image_to_world(px, py, H_img_to_world):
    """
    Transforms a pixel (px, py) from image coordinates
    to real-world coordinates using a homography that
    maps image->world.
    """
    pt = np.array([px, py, 1.0], dtype=np.float64)
    proj = H_img_to_world @ pt
    if abs(proj[2]) < 1e-12:
        return None  # Avoid division by zero
    wx = proj[0] / proj[2]
    wy = proj[1] / proj[2]
    return (wx, wy)

def mouse_callback(event, x, y, flags, param):
    """
    Mouse callback function: 
    - param is a dictionary holding {"image": image, "H": homography}
    - On left click, transform pixel->world and draw text on the image
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        display_img = param["image"]     # The image we are showing
        H_img_to_world = param["H"]      # image->world homography
        
        # 1) Transform from clicked pixel to real-world
        world_pt = transform_image_to_world(x, y, H_img_to_world)
        if world_pt is None:
            return
        wx, wy = world_pt
        
        # 2) Draw a small circle at the clicked point
        cv2.circle(display_img, (x, y), 5, (0, 255, 0), -1)
        
        # 3) Label the point with the real-world coordinates
        text = f"({wx:.2f}, {wy:.2f})"  # Format as you like
        cv2.putText(display_img, text, (x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2, cv2.LINE_AA)
        
        # Update the window
        cv2.imshow("Validation", display_img)

def main():
    # 1) Load the image
    frame = cv2.imread(IMAGE_PATH)
    if frame is None:
        print(f"Failed to load image from {IMAGE_PATH}")
        return

    if USE_PREPROCESSING:
        # Load calibration data
        K, D, DIM = load_calibration_data()
        if K is None or D is None or DIM is None:
            print("Failed to load calibration data.")
            return

        # Preprocess the frame (undistort, etc.)
        frame, _ = preprocess_frame(frame, K, D, DIM)

    # Resize for display
    display_img = cv2.resize(frame, DISPLAY_SIZE, interpolation=cv2.INTER_AREA)
    
    # 2) Load the homography (image->world)
    H_img_to_world = load_homography("coordinate_mapping.json")
    
    # If your matrix is actually world->image, invert it:
    # H_img_to_world = np.linalg.inv(H_world_to_img)  
    # (Uncomment if needed, depending on how you saved it.)

    # 3) Create a named window and set the mouse callback
    cv2.namedWindow("Validation", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(
        "Validation", 
        mouse_callback, 
        param={"image": display_img, "H": H_img_to_world}
    )

    # Show initially
    cv2.imshow("Validation", display_img)
    print("Click on the image to see real-world coordinates (written on the image).")
    print("Press 'q' to quit.")

    # Keep the window open until 'q' is pressed
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()