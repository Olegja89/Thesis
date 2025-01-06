import cv2
import numpy as np
import json

# If you have a config file, you can import VIDEO_PATH from there. For example:
# from config import VIDEO_PATH

def main():
    # -------------------------------------------------------------------------
    # 1. Load the homography data from the JSON file
    # -------------------------------------------------------------------------
    json_file = "coordinate_mapping.json"
    with open(json_file, "r") as f:
        data = json.load(f)
    
    # Homography matrix is 3x3 (pixel -> real-world)
    # We will need the inverse for real-world -> pixel
    homography_matrix = np.array(data["transformation_matrix"], dtype=np.float32)
    
    # -------------------------------------------------------------------------
    # 2. Open the video and read the first frame
    # -------------------------------------------------------------------------
    # Update the video path as needed:
    VIDEO_PATH = "path_to_your_video.mp4"
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret or frame is None:
        print("Could not read the first frame from the video.")
        return
    
    # -------------------------------------------------------------------------
    # 3. Resize the frame as you did in your calibration script
    # -------------------------------------------------------------------------
    display_size = (1920, 1080)  # Adjust to match what you used previously
    image = cv2.resize(frame, display_size, interpolation=cv2.INTER_AREA)
    
    # -------------------------------------------------------------------------
    # 4. Compute the inverse homography (real-world -> pixel)
    # -------------------------------------------------------------------------
    inv_homography = np.linalg.inv(homography_matrix)

    # -------------------------------------------------------------------------
    # 5. Ask user for the grid range and draw the 1x1 meter net
    # -------------------------------------------------------------------------
    print("Specify the range for your grid (in real-world meters).")
    max_x = int(input("Enter maximum X in meters (e.g. 10): "))
    max_y = int(input("Enter maximum Y in meters (e.g. 10): "))
    
    # Draw vertical lines for x = 0..max_x and horizontal lines for y = 0..max_y
    # Each line will be formed by two endpoints in real-world coordinates (homogeneous coords).
    
    # 5.1 Draw vertical grid lines
    for x in range(max_x + 1):
        # Real-world points (homogeneous)
        start_real = np.array([x, 0, 1], dtype=np.float32)
        end_real   = np.array([x, max_y, 1], dtype=np.float32)
        
        # Convert real-world coords to pixel coords using inverse homography
        start_pixel = inv_homography @ start_real
        end_pixel   = inv_homography @ end_real
        
        # Convert from homogeneous to 2D (divide by w)
        if start_pixel[2] != 0:
            start_x = int(start_pixel[0] / start_pixel[2])
            start_y = int(start_pixel[1] / start_pixel[2])
        else:
            continue
        
        if end_pixel[2] != 0:
            end_x = int(end_pixel[0] / end_pixel[2])
            end_y = int(end_pixel[1] / end_pixel[2])
        else:
            continue
        
        cv2.line(image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
    
    # 5.2 Draw horizontal grid lines
    for y in range(max_y + 1):
        start_real = np.array([0, y, 1], dtype=np.float32)
        end_real   = np.array([max_x, y, 1], dtype=np.float32)
        
        start_pixel = inv_homography @ start_real
        end_pixel   = inv_homography @ end_real
        
        if start_pixel[2] != 0:
            start_x = int(start_pixel[0] / start_pixel[2])
            start_y = int(start_pixel[1] / start_pixel[2])
        else:
            continue
        
        if end_pixel[2] != 0:
            end_x = int(end_pixel[0] / end_pixel[2])
            end_y = int(end_pixel[1] / end_pixel[2])
        else:
            continue
        
        cv2.line(image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
    
    # -------------------------------------------------------------------------
    # 6. Show the result
    # -------------------------------------------------------------------------
    cv2.imshow("Grid Check (Homography)", image)
    print("Press any key to close the window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()