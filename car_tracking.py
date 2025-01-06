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
    std_x, std_y = stdev(xs) if len(xs) > 1 else 0, stdev(ys) if len(ys) > 1 else 0

    filtered = []
    for r in records:
        x, y = r['real_world_x'], r['real_world_y']
        # Keep record if within the threshold
        if std_x == 0 and std_y == 0:
            # Means all points are identical or only one point
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
    2. Then pick the frames that maximize coverage across that axis.
    (You could also consider real_world_y or 2D distances or other heuristics.)
    :param records: list of dict with real_world_x, real_world_y, etc.
    :param desired_count: how many frames to pick
    """
    if desired_count <= 0 or len(records) <= desired_count:
        return records  # If the request is 0 or the dataset is small, return everything
    
    # Sort by X coordinate
    sorted_records = sorted(records, key=lambda r: r['real_world_x'])

    # A simplistic approach: pick frames equidistantly in the sorted list
    # so we get the "biggest differences" along X. If we want the largest coverage,
    # we take the min X and max X for sure, then fill in between.
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
    # Sort final pick by frame or real_world_x
    picked.sort(key=lambda r: r['frame'])

    return picked

def main():
    tracking_csv = 'tracking_data.csv'
    mapping_json = 'coordinate_mapping.json'

    # Load homography
    H = load_transformation_data(mapping_json)

    # Read all tracking data
    records = []
    with open(tracking_csv, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields to floats/ints as needed
            row['frame'] = int(row['frame'])
            row['id'] = int(row['id'])
            row['x'] = float(row['x'])
            row['y'] = float(row['y'])
            row['width'] = float(row['width'])
            row['height'] = float(row['height'])
            records.append(row)

    # Wait for user inputs - car id and desired frame count
    car_id_str = input("Enter car id: ")
    try:
        car_id = int(car_id_str)
    except ValueError:
        print("Invalid car ID entered. Exiting.")
        return

    # Optional: let user pick how many frames they want
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

    # Compute bottom-center in image coords & real-world transform
    transformed_records = []
    for r in car_records:
        frame = r['frame']
        cx = r['x'] + r['width'] / 2.0
        cy = r['y'] + r['height']

        # Apply homography
        rwx, rwy = apply_homography(cx, cy, H)
        # If transformation failed, skip
        if rwx is None or rwy is None:
            continue

        transformed_records.append({
            'frame': frame,
            'id': car_id,
            'real_world_x': rwx,
            'real_world_y': rwy,
            'width': r['width'],
            'height': r['height']
        })

    # Remove outliers
    cleaned_records = remove_outliers(transformed_records, std_threshold=2.0)

    # If the user asked for a certain # of frames, select them
    final_records = select_best_frames(cleaned_records, desired_count=desired_count)

    if not final_records:
        print("No records remain after filtering or selection.")
        return

    # Prepare output CSV
    header = ['frame', 'id', 'real_world_x', 'real_world_y', 'width', 'height']
    output_filename = f"car_{car_id}_transformed.csv"
    exporter = CSVExporter(output_filename, header)

    # Sort final records by frame before output (or keep whatever order you prefer)
    final_records.sort(key=lambda x: x['frame'])

    for r in final_records:
        exporter.write_row([
            r['frame'],
            r['id'],
            r['real_world_x'],
            r['real_world_y'],
            r['width'],
            r['height']
        ])

    exporter.close()
    print(f"Data for car id {car_id} exported to {output_filename}")
    print(f"Total frames in output: {len(final_records)}")

if __name__ == "__main__":
    main()
