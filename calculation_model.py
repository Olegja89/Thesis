import numpy as np
from scipy.optimize import least_squares
import csv

# Parameters
D = 1  # Normalized distance
sensor_width = 0.00617
f = 0.00276  # Focal length
image_width = 1920

# Camera coordinates (assumed known)
cam_coordinates = np.array([2.04, -3.21, 3.13])

# Load data from the CSV file generated by transform_and_export.py
input_csv = 'car_1_transformed.csv'
data = np.genfromtxt(input_csv, delimiter=',', skip_header=1)

# Columns (0-based indexing):
# frame=0, id=1, real_world_x=2, real_world_y=3, width=4, height=5

# Define frame limits:
frame_start = 0
frame_end = 10000

# Apply frame filter
mask = (data[:, 0] >= frame_start) & (data[:, 0] <= frame_end)
filtered_data = data[mask]

if filtered_data.size == 0:
    print("No data in the specified frame range.")
    exit()

real_world_x = filtered_data[:, 2]
real_world_y = filtered_data[:, 3]
S = filtered_data[:, 4]  # Width array

# Combine real_world_x and real_world_y into coordinates
coordinates = np.column_stack((real_world_x, real_world_y))

# Calculate distance from the camera
# Assuming objects are on the ground plane (z=0)
d_values = np.sqrt((coordinates[:, 0] - cam_coordinates[0])**2 +
                   (coordinates[:, 1] - cam_coordinates[1])**2 +
                   (0 - cam_coordinates[2])**2)

x_values = coordinates[:, 0]
y_values = coordinates[:, 1]

# Calculate angle
x_rel = coordinates[:, 0] - cam_coordinates[0]
y_rel = coordinates[:, 1] - cam_coordinates[1]
alpha = np.arctan2(x_rel, y_rel)

# Calculate real S
S_real = filtered_data[:, 5]

# Objective function for least squares
def residuals(params, alpha, S_real):
    l, m = params
    return S_real - (l * np.cos(alpha) + m * np.sin(alpha))

# Bounds for l and m
bounds = ([2, 2], [8, 8])  # Lower and upper bounds for l and m

# Initial guess
initial_guess = [2, 5]

# Solve using constrained least squares
result = least_squares(residuals, initial_guess, bounds=bounds, args=(alpha, S_real))

l, m = result.x

print("Estimated l:", l)
print("Estimated m:", m)