# coordinate_transformer.py  
# This module converts image coordinates to real-world coordinates using a homography matrix.  
# It includes:  
# - CoordinateTransformer: Loads transformation matrix from JSON and applies it.  
# - calculate_real_world_coordinates: Computes real-world positions of bounding boxes.  
# - calculate_real_box_width: Estimates real-world object width from bounding box corners.  
# - test_transformation: Verifies transformation accuracy using predefined points (when you run the script itself)
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
    """
    This existing function returns the real-world location of the
    'middle-bottom' point of each bounding box. (Used for speed, etc.)
    """
    real_world_coords = []
    for box in boxes:
        x, y, w, h = box
        # Calculate the tracking point of the box (right now it's middle by X, and 1/4 from the bottom by Y axes)
        image_point = [x, y + h/4]
        # Transform to real-world coordinates
        world_point = transformer.image_to_world(image_point)
        real_world_coords.append(world_point)
    return real_world_coords

def calculate_real_box_width(box, transformer):
    """
    Compute the real-world horizontal distance (width) between the 
    bottom-left and bottom-right corners of the bounding box.
    
    :param box: A tuple/list (x, y, w, h) where (x, y) is the center 
                of the bounding box in image coords, w and h are width/height.
    :param transformer: An instance of CoordinateTransformer.
    :return: A float representing the real-world width (in whatever units 
             your homography is set up for).
    """
    x, y, w, h = box

    # Bottom-left corner in image space
    bottom_left_img = [x - w/2, y + h/2]
    # Bottom-right corner in image space
    bottom_right_img = [x + w/2, y + h/2]

    # Convert both corners to real-world coordinates
    bl_world = transformer.image_to_world(bottom_left_img)
    br_world = transformer.image_to_world(bottom_right_img)

    # Compute Euclidean distance (horizontal real-world width)
    real_width = np.linalg.norm(np.array(br_world) - np.array(bl_world))
    return real_width

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
