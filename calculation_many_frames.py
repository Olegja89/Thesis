#same method as calculation_model_2points.py, but uses many different combinations of points to find the best
import numpy as np
import csv

# Load data from the CSV file
input_csv = "car_14_transformed.csv"
data = np.genfromtxt(input_csv, delimiter=',', skip_header=1)

# Camera coordinates (first for 40-50kmph, second for 20-30kmph videos)
#cam_coordinates = np.array([-0.21, -8.37, 3.13])
cam_coordinates = np.array([2.04, -3.21, 3.13]) 

# Columns (0-based indexing):
# frame=0, id=1, real_world_x=2, real_world_y=3, width=4, height=5

# Get available frames
frames = np.unique(data[:, 0]).astype(int)
print(f"Available frames: {frames}")
selected_frame = int(input("Choose a frame: "))

# Filter data for the selected frame
selected_data = data[data[:, 0] == selected_frame]

if selected_data.shape[0] != 1:
    print("Selected frame must contain exactly one data point.")
    exit()

# Extract the selected point
selected_x, selected_y, selected_S_real = selected_data[0, 2], selected_data[0, 3], selected_data[0, 5]

# Prepare CSV output file
output_csv = "solved_m_l_per_frame.csv"
with open(output_csv, "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["frame", "m", "l"])

    for frame in frames:
        if frame == selected_frame:
            continue
        
        frame_data = data[data[:, 0] == frame]
        if frame_data.shape[0] != 1:
            continue
        
        # Extract second point
        frame_x, frame_y, frame_S_real = frame_data[0, 2], frame_data[0, 3], frame_data[0, 5]
        
        # Compute angles
        x_rel_1, y_rel_1 = selected_x - cam_coordinates[0], selected_y - cam_coordinates[1]
        x_rel_2, y_rel_2 = frame_x - cam_coordinates[0], frame_y - cam_coordinates[1]
        alpha_1 = np.arctan2(x_rel_1, y_rel_1)
        alpha_2 = np.arctan2(x_rel_2, y_rel_2)
        
        # Compute denominator
        den = np.sin(alpha_2) * np.cos(alpha_1) - np.sin(alpha_1) * np.cos(alpha_2)
        if abs(den) < 1e-12:
            continue  # Skip if angles are too close
        
        # Solve for m and l
        m = (frame_S_real * np.cos(alpha_1) - selected_S_real * np.cos(alpha_2)) / den
        l = (selected_S_real * np.sin(alpha_2) - frame_S_real * np.sin(alpha_1)) / den
        
        # Save to CSV
        writer.writerow([frame, m, l])

print(f"Results saved to {output_csv}")