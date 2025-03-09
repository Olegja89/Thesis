#calculates car size based on .csv file exported from car_tracking.py
import numpy as np
from scipy.optimize import least_squares
import csv

# Load data from the CSV file
input_csv = "car_2_transformed.csv"
data = np.genfromtxt(input_csv, delimiter=',', skip_header=1)

# Camera coordinates (first for 20-30kmph, second for 40-50kmph videos)
cam_coordinates = np.array([-0.21, -8.37, 3.13])
#cam_coordinates = np.array([2.04, -3.21, 3.13]) 

# Columns (0-based indexing):
# frame=0, id=1, real_world_x=2, real_world_y=3, width=4, real_width=5

# Extract relevant data
real_world_x = data[:, 2]
real_world_y = data[:, 3]
S_real = data[:, 5]

# Compute angles
x_rel = real_world_x - cam_coordinates[0]
y_rel = real_world_y - cam_coordinates[1]
alpha = np.arctan2(x_rel, y_rel)

# Objective function for least squares
def residuals(params, alpha, S_real):
    l, m = params
    return S_real - (l * np.cos(alpha) + m * np.sin(alpha))

# Bounds for l and m
bounds = ([2, -3], [8, 8])  # Lower and upper bounds for l and m

# Initial guess
initial_guess = [2, 5]

# Solve using constrained least squares
result = least_squares(residuals, initial_guess, bounds=bounds, args=(alpha, S_real))
l, m = result.x

# Prepare CSV output file
output_csv = "solved_m_l_per_frame.csv"
with open(output_csv, "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["l", "m"])
    writer.writerow([l, m])

print(f"Results saved to {output_csv}")
