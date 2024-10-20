import json
import numpy as np
import cv2

class CoordinateTransformer:
    def __init__(self, mapping_file):
        with open(mapping_file, 'r') as f:
            data = json.load(f)
        
        self.matrix = np.array(data['transformation_matrix'])

    def image_to_world(self, image_point):
        # Convert to homogeneous coordinates
        point_h = np.array([image_point[0], image_point[1], 1])
        
        # Apply transformation
        world_h = self.matrix @ point_h
        
        # Convert back from homogeneous coordinates
        world_point = world_h[:2] / world_h[2]
        
        return world_point

def calculate_real_world_coordinates(boxes, transformer):
    real_world_coords = []
    for box in boxes:
        x, y, w, h = box
        # Calculate the middle-bottom point of the box
        image_point = [x, y + h/2]
        # Transform to real-world coordinates
        world_point = transformer.image_to_world(image_point)
        real_world_coords.append(world_point)
    return real_world_coords

# Function to test the transformation
def test_transformation(mapping_file):
    transformer = CoordinateTransformer(mapping_file)
    with open(mapping_file, 'r') as f:
        data = json.load(f)
    
    image_points = data['image_points']
    real_world_points = data['real_world_points']

    print("Testing coordinate transformation:")
    for img_pt, real_pt in zip(image_points, real_world_points):
        calculated_pt = transformer.image_to_world(img_pt)
        print(f"Image point: {img_pt}")
        print(f"Expected real-world point: {real_pt}")
        print(f"Calculated real-world point: {calculated_pt}")
        print("---")

if __name__ == "__main__":
    test_transformation("coordinate_mapping.json")