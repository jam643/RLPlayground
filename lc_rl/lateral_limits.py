import numpy as np
import matplotlib.pyplot as plt

def create_lateral_limits(num_points: int, min_width: float, max_width: float):
    # Parameters
    max_interval_length = 60
    transition_steepness = 0.25

    width = np.zeros(num_points)
    end_index = 0

    while(end_index < num_points):
        start_index = end_index
        end_index = min(num_points, start_index + np.random.randint(0, max_interval_length))
        width[start_index:end_index] = np.random.rand()*(max_width - min_width) + min_width

    processed_width = width.copy()

    max_steps = int((max_width-min_width)/transition_steepness) + 1
    for i in range(1, max_steps):
        processed_width[:-i] = np.minimum(processed_width[:-i], width[i:] + np.abs(i) * transition_steepness)
        processed_width[i:] = np.minimum(processed_width[i:], width[:-i] + np.abs(i) * transition_steepness)

    return processed_width

if __name__ == "__main__":
    processed_array = create_lateral_limits(100, 0, 1)

    plt.plot(processed_array)
    plt.title('Lateral Limits')
    plt.xlabel('Index')
    plt.ylabel('Width')
    plt.show()