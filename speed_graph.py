import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

class SpeedAnalyzer:
    def __init__(self, csv_file, fps=25):
        self.data = pd.read_csv(csv_file)
        self.fps = fps
        self.calculate_speeds()

    def calculate_speeds(self):
        # Sort the dataframe by id and frame
        self.data = self.data.sort_values(['id', 'frame'])
        
        # Calculate displacement and time difference
        self.data['dx'] = self.data.groupby('id')['world_x'].diff()
        self.data['dy'] = self.data.groupby('id')['world_y'].diff()
        self.data['dt'] = self.data.groupby('id')['frame'].diff() / self.fps
        
        # Calculate speed
        self.data['speed'] = np.sqrt(self.data['dx']**2 + self.data['dy']**2) / self.data['dt']
        
        # Replace inf and NaN values with 0 (for the first frame of each object)
        self.data['speed'] = self.data['speed'].replace([np.inf, -np.inf], np.nan).fillna(0)

    def get_object_speed(self, object_id):
        return self.data[self.data['id'] == object_id][['frame', 'speed']].values

    def plot_speed(self, object_id, smoothing=True, window_length=5, polyorder=2):
        object_data = self.data[self.data['id'] == object_id]
        
        if object_data.empty:
            print(f"No data found for object ID {object_id}")
            return
        
        frames = object_data['frame'].values
        speeds = object_data['speed'].values
        
        if smoothing and len(speeds) > window_length:
            speeds = savgol_filter(speeds, window_length, polyorder)
        
        plt.figure(figsize=(10, 6))
        plt.plot(frames, speeds, label=f'Object ID: {object_id}')
        plt.title(f'Speed of Object {object_id} Over Time')
        plt.xlabel('Frame')
        plt.ylabel('Speed (meters/second)')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_trajectory(self, object_id):
        object_data = self.data[self.data['id'] == object_id]
        
        if object_data.empty:
            print(f"No data found for object ID {object_id}")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(object_data['world_x'], object_data['world_y'])
        plt.title(f'Trajectory of Object {object_id}')
        plt.xlabel('X coordinate (meters)')
        plt.ylabel('Y coordinate (meters)')
        plt.grid(True)
        plt.axis('equal')
        plt.show()

def main():
    analyzer = SpeedAnalyzer('world_coordinates.csv')
    
    while True:
        print("\nOptions:")
        print("1. Plot speed graph for an object")
        print("2. Plot trajectory for an object")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            object_id = int(input("Enter object ID: "))
            analyzer.plot_speed(object_id)
        elif choice == '2':
            object_id = int(input("Enter object ID: "))
            analyzer.plot_trajectory(object_id)
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()