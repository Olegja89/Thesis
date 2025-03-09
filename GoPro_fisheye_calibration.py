#This script performs fisheye lens calibration for a GoPro camera using a checkerboard pattern. 
#It detects and refines checkerboard corners from calibration images, computes the camera matrix and distortion coefficients, and saves the calibration data in an .npz file.
#The calibration process uses OpenCV's fisheye model and requires a minimum of five valid images with detected corners.
#The resulting calibration parameters are essential for undistorting images or correcting fisheye distortions in further processing.

import cv2
import numpy as np
import glob
import os

def calibrate_fisheye(
    images_folder,
    grid_size=(6, 8),           # Checkerboard corners (width=9, height=6 commonly)
    square_size=0.019,          # Real-world size of each square in meters (example 2.5 cm)
    output_file='gopro_calibration.npz'
):
    """
    Perform fisheye calibration using images of a checkerboard pattern.
    :param images_folder: Folder containing calibration images.
    :param grid_size: Number of internal corners (columns, rows) in the checkerboard pattern.
    :param square_size: Physical size of each checkerboard square (in meters).
    :param output_file: Output file name to store the calibration data (K, D, DIM).
    """

    # Termination criteria for corner sub-pix refinement
    subpix_criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        1e-6
    )

    # Flags for fisheye calibration
    calibration_flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
        cv2.fisheye.CALIB_CHECK_COND +
        cv2.fisheye.CALIB_FIX_SKEW
    )

    # Prepare known 3D object points for the checkerboard pattern
    # (0,0,0), (1,0,0), (2,0,0) ... but scaled by square_size
    # grid_size = (cols, rows) = (9,6) means 9 corners horizontally, 6 vertically
    objp = np.zeros((1, grid_size[0]*grid_size[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:grid_size[0], 0:grid_size[1]].T.reshape(-1, 2)
    objp *= square_size

    # Lists to store object points and image points
    # for each calibration image
    objpoints = []  # 3D points in real world
    imgpoints = []  # 2D points in image plane

    # Load all images from the folder
    images = glob.glob(os.path.join(images_folder, '*.jpg'))
    images += glob.glob(os.path.join(images_folder, '*.png'))
    images += glob.glob(os.path.join(images_folder, '*.jpeg'))

    if not images:
        print(f"No images found in folder: {images_folder}")
        return

    # For dimension checking
    used_img_shape = None

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"Could not read image {fname}, skipping.")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(
            gray,
            grid_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if ret:
            # Refine corner positions
            cv2.cornerSubPix(gray, corners, (3,3), (-1,-1), subpix_criteria)

            # Store results
            objpoints.append(objp)
            imgpoints.append(corners)

            if used_img_shape is None:
                used_img_shape = gray.shape[::-1]  # (width, height)
            
            # Optionally visualize
            cv2.drawChessboardCorners(img, grid_size, corners, ret)
            # cv2.imshow('Corners', img)
            # cv2.waitKey(100)
        else:
            print(f"Chessboard not found in {fname}")

    # cv2.destroyAllWindows()

    # Now we have collected (objectPoints, imagePoints). Let's calibrate in fisheye mode.
    N_OK = len(objpoints)
    if N_OK < 5:
        print(f"Not enough valid calibration images (found corners in {N_OK}). Need >= 5.")
        return

    # Prepare arrays for fisheye calibration
    objpoints = np.array(objpoints, dtype=np.float32)
    imgpoints = np.array(imgpoints, dtype=np.float32)

    K = np.zeros((3,3))
    D = np.zeros((4,1))
    rvecs = []
    tvecs = []

    # Perform calibration
    rms, _, _, _, _ = cv2.fisheye.calibrate(
        objectPoints=objpoints,
        imagePoints=imgpoints,
        image_size=used_img_shape,
        K=K,
        D=D,
        rvecs=rvecs,
        tvecs=tvecs,
        flags=calibration_flags,
        criteria=subpix_criteria
    )

    # K is our camera matrix; D is our fisheye distortion coefficients
    print("Fisheye calibration done.")
    print(f"Number of images used for calibration: {N_OK}")
    print(f"RMS error: {rms}")
    print("Camera matrix (K):\n", K)
    print("Distortion (D):\n", D)
    
    # "Exact" focal length from K
    # Typically, fx = K[0,0], fy = K[1,1] in a pinhole model.
    # For the fisheye model, K still contains fx, fy in these elements.
    fx = K[0, 0]
    fy = K[1, 1]
    print(f"Estimated focal lengths: fx={fx}, fy={fy}")

    # Save results
    DIM = used_img_shape  # (width, height)
    np.savez(
        output_file,
        K=K,
        D=D,
        DIM=DIM,
        rms=rms,
        focal_length=(fx, fy)
    )
    print(f"Calibration data saved to {output_file}")

if __name__ == '__main__':
    # Example usage:
    # 1) Place your checkerboard images in a folder like "calibration_images"
    # 2) Adjust grid_size to match your checkerboard
    # 3) Adjust square_size if each square is a different dimension
    # 4) Run the script: python calibrate_fisheye.py
    images_folder = 'Images'  # your folder with images
    calibrate_fisheye(
        images_folder=images_folder,
        grid_size=(6, 8),        # or whatever your checkerboard corners are
        square_size=0.019,        # 2 cm squares, for example
        output_file='gopro_calibration_fisheye.npz'
    )