import numpy as np

# Define your coordinates as a list of tuples
S = np.array([85, 185, 460]) # Sizes of the boxes
coordinates = np.array([(3.9, 52.4), (3.9, 25.7), (3.9, 12)]) # x,y coordinates of the boxes
z = 4 # Height of the camera
D = 1 # Normalized distance
sensor_width = 0.036 
f = 0.036 # Focal length
image_width = 1920

# Calculate distance from the camera
d_values = np.sqrt(coordinates[:, 0]**2 + coordinates[:, 1]**2 + z**2) 
x_values = coordinates[:, 0]
y_values = coordinates[:, 1]

# Calculate angle
alpha = np.arctan2(y_values, x_values)

# Calculate real S
S_real = np.array(S*d_values)

# Least Squares
A = np.column_stack((np.cos(alpha), np.sin(alpha)))
params, residuals, rank, s = np.linalg.lstsq(A, S_real, rcond=None)
l, m = params

l_real=(l*D*sensor_width)/(f*image_width)
m_real=(m*D*sensor_width)/(f*image_width)

print("Estimated l:", l_real)
print("Estimated m:", m_real)