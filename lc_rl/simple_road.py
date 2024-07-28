from dataclasses import dataclass
import numpy as np


def generate_standard_lookahead_profile():
    series = []
    current_value = 0
    increment = 1.0
    group_count = 5

    while True:
        for _ in range(group_count):
            current_value += increment
            series.append(current_value)
            if len(series) >= 20:
                return series
        increment *= 2


@dataclass
class SimpleRoad:
    baseline_poses: np.ndarray
    left_width: np.ndarray
    right_width: np.ndarray
    path_ds: float

    def get_max_station(self):
        return self.path_ds * (len(self.baseline_poses) - 1)

    def get_curvature_lookahead(self, station: float, lookahead_profile: np.ndarray):
        # Extract curvature from baseline poses
        curvatures = self.baseline_poses[:, 3]

        lookahead_curvatures = []
        for i in range(len(lookahead_profile)):
            start_station = station + (lookahead_profile[i - 1] if i > 0 else 0)
            end_station = station + lookahead_profile[i]

            start_index = int(np.floor(start_station / self.path_ds))
            end_index = int(np.floor(end_station / self.path_ds))

            if end_index >= len(curvatures):
                end_index = len(curvatures) - 1

            curvature_segment = curvatures[start_index:end_index + 1]
            mean_curvature = np.mean(curvature_segment)
            lookahead_curvatures.append(mean_curvature)

        return np.array(lookahead_curvatures)

    def get_pose_at_station(self, station):
        # Interpolate path state
        continuous_index = station / self.path_ds
        path_from_index = int(np.floor(continuous_index))
        prog = continuous_index - path_from_index
        from_pose = self.baseline_poses[path_from_index, :]
        to_pose = self.baseline_poses[path_from_index + 1, :]
        return from_pose * (1 - prog) + to_pose * prog

    def get_width_at_station(self, station):
        # Interpolate width
        continuous_index = station / self.path_ds
        path_from_index = int(np.floor(continuous_index))
        prog = continuous_index - path_from_index
        from_left_width = self.left_width[path_from_index]
        from_right_width = self.right_width[path_from_index]
        to_left_width = self.left_width[path_from_index + 1]
        to_right_width = self.right_width[path_from_index + 1]
        left_width = from_left_width * (1 - prog) + to_left_width * prog
        right_width = from_right_width * (1 - prog) + to_right_width * prog
        return left_width, right_width



    def get_width_lookahead(self, station: float, lookahead_profile: np.ndarray):
        lookahead_left_widths = []
        lookahead_right_widths = []
        for i in range(len(lookahead_profile)):
            start_station = station + (lookahead_profile[i - 1] if i > 0 else 0)
            end_station = station + lookahead_profile[i]

            start_index = int(np.floor(start_station / self.path_ds))
            end_index = int(np.floor(end_station / self.path_ds))

            if end_index >= len(self.left_width):
                end_index = len(self.left_width) - 1

            left_width_segment = self.left_width[start_index:end_index + 1]
            right_width_segment = self.right_width[start_index:end_index + 1]

            mean_left_width = np.mean(left_width_segment)
            mean_right_width = np.mean(right_width_segment)

            lookahead_left_widths.append(mean_left_width)
            lookahead_right_widths.append(mean_right_width)

        return np.array(lookahead_left_widths), np.array(lookahead_right_widths)


if __name__ == "__main__":
    # Example usage:
    baseline_poses = np.array([
        [0, 0, 0, 0],
        [1, 1, 0, 0.1],
        [2, 2, 0, 0.2],
        [3, 3, 0, 0.3]
    ])
    left_width = np.array([1, 1.1, 1.2, 1.3])
    right_width = np.array([1, 0.9, 0.8, 0.7])
    path_ds = 1.0

    road = SimpleRoad(baseline_poses, left_width, right_width, path_ds)
    station = 0
    lookahead_profile = np.array([1, 2, 3])
    curvatures = road.get_curvature_lookahead(station, lookahead_profile)
    left_widths, right_widths = road.get_width_lookahead(station, lookahead_profile)
    print("Curvatures:", curvatures)
    print("Left widths:", left_widths)
    print("Right widths:", right_widths)

    lookahead_profile = generate_standard_lookahead_profile()
    print(len(lookahead_profile))
    print(lookahead_profile)
