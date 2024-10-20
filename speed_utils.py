import numpy as np
from collections import deque

FPS = 25  # Frames per second of the video
SPEED_BUFFER_SIZE = 5  # Number of frames to average speed over

def calculate_speed(prev_pos, curr_pos, time_diff):
    """Calculate speed in km/h given two positions and time difference."""
    distance = np.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
    speed_ms = distance / time_diff if time_diff > 0 else 0
    return speed_ms * 3.6  # Convert m/s to km/h

class SpeedTracker:
    def __init__(self):
        self.trackers = {}

    def update_speed(self, track_id, world_coord, time_diff):
        if track_id not in self.trackers:
            self.trackers[track_id] = deque(maxlen=SPEED_BUFFER_SIZE)
        
        speed_tracker = self.trackers[track_id]
        if speed_tracker:
            speed = calculate_speed(speed_tracker[-1], world_coord, time_diff)
        else:
            speed = 0
        
        speed_tracker.append(world_coord)
        return np.mean([calculate_speed(speed_tracker[i], speed_tracker[i+1], 1/FPS) 
                        for i in range(len(speed_tracker)-1)]) if len(speed_tracker) > 1 else speed

    def get_speeds(self, track_ids, world_coords, time_diff):
        return [self.update_speed(track_id, world_coord, time_diff) 
                for track_id, world_coord in zip(track_ids, world_coords)]