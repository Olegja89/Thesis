import csv
import json
import math
from statistics import mean, stdev
from data_export import CSVExporter

def load_transformation_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    # Extract the transformation matrix
    matrix = data['transformation_matrix']
    return matrix

def apply_homography(x, y, H):
    """
    Apply homography transformation H to the point (x, y).
    H is a 3x3 matrix: [[h11,h12,h13],[h21,h22,h23],[h31,h32,h33]]
    """
    denom = (H[2][0]*x + H[2][1]*y + H[2][2])
    if denom == 0:
        # Avoid division by zero - this would be a malformed homography or point outside domain
        return None, None
    X = (H[0][0]*x + H[0][1]*y + H[0][2]) / denom
    Y = (H[1][0]*x + H[1][1]*y + H[1][2]) / denom
    return X, Y

def remove_outliers(records, std_threshold=2.0):
    """
    Remove outliers based on real_world_x and real_world_y.
    We use a simple standard-deviation based approach.
    :param records: list of dict with keys ['real_world_x', 'real_world_y', ...]
    :param std_threshold: how many standard deviations to allow.
    """
    if not records:
        return records

    # Extract X and Y
    xs = [r['real_world_x'] for r in records]
    ys = [r['real_world_y'] for r in records]

    mean_x, mean_y = mean(xs), mean(ys)
    # If there's only one record, stdev() would fail; handle gracefully:
    std_x = stdev(xs) if len(xs) > 1 else 0
    std_y = stdev(ys) if len(ys) > 1 else 0

    filtered = []
    for r in records:
        x, y = r['real_world_x'], r['real_world_y']
        # Keep record if within the threshold or if stdev is zero
        if std_x == 0 and std_y == 0:
            # Means all points are identical or there's only one point
            filtered.append(r)
        else:
            if (abs(x - mean_x) <= std_threshold * std_x) and \
               (abs(y - mean_y) <= std_threshold * std_y):
                filtered.append(r)

    return filtered

def select_best_frames(records, desired_count=10):
    """
    Select the best (most 'spread') frames if the user wants a certain number of frames.
    We'll illustrate a simple approach:
    1. Sort all frames by real_world_x (ascending).
    2. Then pick frames that maximize coverage across that axis.
    :param records: list of dict with real_world_x, real_world_y, etc.
    :param desired_count: how many frames to pick
    """
    if desired_count <= 0 or len(records) <= desired_count:
        return records  # If the request is 0 or dataset is small, return everything
    
    # Sort by X coordinate
    sorted_records = sorted(records, key=lambda r: r['real_world_x'])

    # A simple approach: pick frames equidistantly in the sorted list
    # so we get coverage along X.
    picked = []
    n = len(sorted_records)
    step = n / (desired_count - 1)  # We'll pick first, last, and so on

    for i in range(desired_count):
        index = int(round(i * step))
        if index >= n:
            index = n - 1
        picked.append(sorted_records[index])

    # Convert to set to avoid duplicates, then back to list
    picked = list({p['frame']: p for p in picked}.values())
    # Sort final picks by frame (or by x, if you prefer)
    picked.sort(key=lambda r: r['frame'])

    return picked

def main():
    tracking_csv = 'tracking_data.csv'       # CSV now contains columns: frame,id,x,y,width,real_width
    mapping_json = 'coordinate_mapping.json' # Homography data

    # Load homography
    H = load_transformation_data(mapping_json)

    # Read all tracking data
    records = []
    with open(tracking_csv, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            row['frame'] = int(row['frame'])
            row['id'] = int(row['id'])
            row['x'] = float(row['x'])
            row['y'] = float(row['y'])
            row['width'] = float(row['width'])
            # CHANGED: read 'real_width' instead of 'height'
            row['real_width'] = float(row['real_width'])
            records.append(row)

    # Ask user for car id
    car_id_str = input("Enter car id: ")
    try:
        car_id = int(car_id_str)
    except ValueError:
        print("Invalid car ID entered. Exiting.")
        return

    # Ask for desired frame count
    desired_count_str = input("Enter desired number of frames (0 for all, default=0): ")
    try:
        desired_count = int(desired_count_str) if desired_count_str.strip() else 0
    except ValueError:
        desired_count = 0  # default

    # Filter records for the chosen car_id
    car_records = [r for r in records if r['id'] == car_id]
    if not car_records:
        print(f"No records found for car id {car_id}.")
        return

    # Compute real-world coordinates for the center (x, y)
    transformed_records = []
    for r in car_records:
        frame = r['frame']
        # CHANGED: No longer use bounding-box height. 
        # We assume x,y is already the bounding box center in the new CSV.
        cx = r['x']
        cy = r['y']

        # Apply homography
        rwx, rwy = apply_homography(cx, cy, H)
        if rwx is None or rwy is None:
            continue

        transformed_records.append({
            'frame': frame,
            'id': car_id,
            'real_world_x': rwx,
            'real_world_y': rwy,
            'width': r['width'],
            # Use the real_width from the CSV
            'real_width': r['real_width']
        })

    # Remove outliers
    cleaned_records = remove_outliers(transformed_records, std_threshold=2.0)

    # If the user asked for a certain # of frames, select them
    final_records = select_best_frames(cleaned_records, desired_count=desired_count)
    if not final_records:
        print("No records remain after filtering or selection.")
        return

    # Prepare output CSV
    # CHANGED: now we export 'width' and 'real_width' (instead of 'height')
    header = ['frame', 'id', 'real_world_x', 'real_world_y', 'width', 'real_width']
    output_filename = f"car_{car_id}_transformed.csv"
    exporter = CSVExporter(output_filename, header)

    # Sort final records by frame before output
    final_records.sort(key=lambda x: x['frame'])

    for r in final_records:
        exporter.write_row([
            r['frame'],
            r['id'],
            r['real_world_x'],
            r['real_world_y'],
            r['width'],
            r['real_width']
        ])

    exporter.close()
    print(f"Data for car id {car_id} exported to {output_filename}")
    print(f"Total frames in output: {len(final_records)}")

if __name__ == "__main__":
    main()