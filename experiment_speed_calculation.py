import pandas as pd
import matplotlib.pyplot as plt

def plot_speed_for_car(csv_path):
    # Load data
    df = pd.read_csv(csv_path)
    
    # Ask user for car ID
    car_id = int(input("Enter the Car ID to plot speed: "))
    
    # Filter data for the selected car ID
    car_df = df[df['id'] == car_id]
    
    if car_df.empty:
        print(f"No data found for Car ID {car_id}.")
        return
    
    # Plot Speed vs Time
    plt.figure()
    plt.plot(car_df['frame'], car_df['speed_kmh'], label=f'Car {car_id} Speed')
    plt.xlabel("Time (frames)")
    plt.ylabel("Speed (km/h)")
    plt.title(f"Speed vs Time for Car ID {car_id}")
    plt.legend()
    plt.savefig(f"speed_v_t_car_{car_id}.png")
    
    # Plot Speed vs Distance
    plt.figure()
    plt.plot(car_df['distance_m'], car_df['speed_kmh'], label=f'Car {car_id} Speed')
    plt.xlabel("Distance (m)")
    plt.ylabel("Speed (km/h)")
    plt.title(f"Speed vs Distance for Car ID {car_id}")
    plt.legend()
    plt.savefig(f"speed_v_x_car_{car_id}.png")
    
    plt.show()
    print(f"Plots saved as speed_v_t_car_{car_id}.png and speed_v_x_car_{car_id}.png")

# Example Usage
if __name__ == "__main__":
    FILE_PATH = "world_coordinates.csv"  # Update with actual path
    plot_speed_for_car(FILE_PATH)