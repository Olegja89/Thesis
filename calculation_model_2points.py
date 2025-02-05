import numpy as np
import csv

# Parameters
D = 1  # Normalized distance (unused here, but left for reference)
sensor_width = 0.00617
f = 0.00276  # Focal length
image_width = 1920

# Camera coordinates (assumed known)
cam_coordinates = np.array([-0.21, -8.37, 3.13])

# Load data from the CSV file
input_csv = 'car_4_transformed.csv'
data = np.genfromtxt(input_csv, delimiter=',', skip_header=1)

# Columns (0-based indexing):
# frame=0, id=1, real_world_x=2, real_world_y=3, width=4, height=5

# Define frame limits:
frame_start = 0
frame_end = 10000

# Apply frame filter
mask = (data[:, 0] >= frame_start) & (data[:, 0] <= frame_end)
filtered_data = data[mask]

# Check we have at least 2 data points
if filtered_data.shape[0] < 2:
    print("Need at least 2 data points in the specified frame range.")
    exit()

# Extract columns of interest
real_world_x = filtered_data[:, 2]
real_world_y = filtered_data[:, 3]
S = filtered_data[:, 4]  # Width in pixels

# Combine x,y for convenience
coordinates = np.column_stack((real_world_x, real_world_y))

# Calculate distance from the camera (assuming z=0 for objects)
d_values = np.sqrt(
    (coordinates[:, 0] - cam_coordinates[0])**2 +
    (coordinates[:, 1] - cam_coordinates[1])**2 +
    (0 - cam_coordinates[2])**2
)

# Calculate angles alpha
x_rel = coordinates[:, 0] - cam_coordinates[0]
y_rel = coordinates[:, 1] - cam_coordinates[1]
alpha_all = np.arctan2(x_rel, y_rel)
print("x_rel", x_rel)
print("y_rel", y_rel)
print("alpha:", alpha_all)
# Calculate real S array
S_real_all = filtered_data[:, 5]
print("S:", S_real_all)

# --------------------------------------------------
# Pick exactly two data points to solve for l and m
# (Here we just pick the first two. You can pick any
#  that you trust or that have distinct angles.)
# --------------------------------------------------
alpha_1 = alpha_all[0]
alpha_2 = alpha_all[1]
S_real_1 = S_real_all[0]
S_real_2 = S_real_all[1]

# Denominator: sin(alpha2 - alpha1)
den = np.sin(alpha_2) * np.cos(alpha_1) - np.sin(alpha_1) * np.cos(alpha_2)

if abs(den) < 1e-12:
    print("Angles are too close or identical. Cannot solve directly.")
    exit()

# Solve for m (in one line)
m = (S_real_2 * np.cos(alpha_1) - S_real_1 * np.cos(alpha_2)) / den

# Solve for l (in one line)
l = (S_real_1 * np.sin(alpha_2) - S_real_2 * np.sin(alpha_1)) / den

print("Estimated l:", l*1.04)
print("Estimated m:", m*0.887)