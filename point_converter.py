import numpy as np

def load_points_from_file(filename):
    points = []
    with open(filename, 'r') as file:
        for line in file:
            data = line.strip().split(',')
            if len(data) >= 3:
                label, x, y = data[0], float(data[1]), float(data[2])
                points.append((label, np.array([x, y])))
    return points

def compute_axes(origin, direction_vector):
    road_unit = direction_vector / np.linalg.norm(direction_vector)  # Normalize
    
    # Compute perpendicular vector
    perp_vector = np.array([-road_unit[1], road_unit[0]])
    return road_unit, perp_vector

def transform_point(point, origin, road_unit, perp_vector):
    shifted_point = point - origin  # Shift the point relative to the new origin
    x_new = np.dot(road_unit, shifted_point)  # Projection onto road axis
    y_new = np.dot(perp_vector, shifted_point)  # Projection onto perpendicular axis
    return x_new, y_new

def main():
    filename = input("Enter the filename containing points: ")
    points = load_points_from_file(filename)
    
    if len(points) < 2:
        print("At least 2 points are needed: one for origin and one for direction.")
        return
    
    # Display available points
    print("Available points:")
    for i, (label, _) in enumerate(points):
        print(f"{i}: {label}")
    
    # Ask for user selection
    origin_index = int(input("Enter the index of the point to use as origin: "))
    dir_index = int(input("Enter the index of the point to use as direction reference: "))
    
    origin_label, origin = points[origin_index]
    dir_label, dir_point = points[dir_index]
    
    print(f"Using {origin_label} as the origin and {dir_label} for direction.")
    direction_vector = dir_point - origin
    
    road_unit, perp_vector = compute_axes(origin, direction_vector)
    
    print("Transforming remaining points:")
    transformed_points = []
    
    for i, (label, point) in enumerate(points):
        #if i == origin_index or i == dir_index:
            #continue  # Skip selected reference points
        
        new_coords = transform_point(point, origin, road_unit, perp_vector)
        transformed_points.append((label, new_coords))
        print(f"{label}: X'={new_coords[0]:.4f}, Y'={new_coords[1]:.4f}")
    
    print("\nFinal Transformed Points:")
    for label, (x_prime, y_prime) in transformed_points:
        print(f"{label}: X'={x_prime:.4f}, Y'={y_prime:.4f}")

if __name__ == "__main__":
    main()