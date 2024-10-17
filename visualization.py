import csv
import cv2
import numpy as np
from collections import defaultdict
import colorsys

def generate_colors(n):
    HSV_tuples = [(x * 1.0 / n, 0.5, 0.5) for x in range(n)]
    return list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))

def read_tracking_data(csv_file):
    tracking_data = defaultdict(list)
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header
        for row in csv_reader:
            frame, track_id, x, y = map(float, row[:4])
            tracking_data[int(track_id)].append((int(frame), int(x), int(y)))
    return tracking_data

def create_visualization(tracking_data, output_file, frame_size=(1280, 720), duration=10):
    colors = generate_colors(len(tracking_data))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, 30, frame_size)
    
    # Calculate max_frame correctly
    max_frame = max(max(frame for frame, _, _ in track) for track in tracking_data.values())
    
    for frame in range(max_frame + 1):
        img = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
        
        for (track_id, track), color in zip(tracking_data.items(), colors):
            points = [point for point in track if point[0] <= frame]
            if len(points) > 1:
                cv2.polylines(img, [np.array([(x, y) for _, x, y in points])], False, 
                              (int(color[0]*255), int(color[1]*255), int(color[2]*255)), 2)
            
            if points:
                last_point = points[-1]
                cv2.circle(img, (last_point[1], last_point[2]), 5, 
                           (int(color[0]*255), int(color[1]*255), int(color[2]*255)), -1)
                cv2.putText(img, f"ID: {track_id}", (last_point[1] + 10, last_point[2] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(img, f"Frame: {frame}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(img)
        
        if frame % 30 == 0:  # Update progress every second
            print(f"Processing frame {frame}/{max_frame}")
    
    out.release()
    print(f"Visualization saved to {output_file}")

if __name__ == "__main__":
    csv_file = "tracking_data.csv"
    output_file = "tracking_visualization.mp4"
    frame_size = (1280, 720)
    
    tracking_data = read_tracking_data(csv_file)
    if not tracking_data:
        print("No tracking data found. Please check your CSV file.")
    else:
        create_visualization(tracking_data, output_file, frame_size)