from dataclasses import dataclass
import numpy as np
from abc import ABC, abstractmethod
from lateral_limits import create_lateral_limits
from dubins_py import get_multi_waypoint_from_np, is_path_valid
from enum import Enum

def generate_standard_lookahead_profile():
    series = []
    current_value = 0
    increment = 1.0
    group_count = 4

    while True:
        for _ in range(group_count):
            current_value += increment
            series.append(current_value)
            if len(series) >= 20:
                return series
        increment *= 2.3


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

        # Interpolate x and y positions
        interpolated_pose = from_pose * (1 - prog) + to_pose * prog

        # Handle angle wraparound for the heading
        from_angle = from_pose[2]
        to_angle = to_pose[2]

        # Calculate the difference and wrap around if necessary
        angle_diff = to_angle - from_angle
        if angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        elif angle_diff < -np.pi:
            angle_diff += 2 * np.pi

        interpolated_angle = from_angle + prog * angle_diff

        # Assign the interpolated angle back to the pose
        interpolated_pose[2] = interpolated_angle

        return interpolated_pose

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

    def get_lines_for_visualization(self):
        baseline_x = self.baseline_poses[:, 0]
        baseline_y = self.baseline_poses[:, 1]
        headings = self.baseline_poses[:, 2]

        # Calculate offsets for right and left boundary lines
        right_x = baseline_x - self.right_width * np.cos(headings + np.pi / 2)
        right_y = baseline_y - self.right_width * np.sin(headings + np.pi / 2)
        left_x = baseline_x + self.right_width * np.cos(headings + np.pi / 2)
        left_y = baseline_y + self.left_width * np.sin(headings + np.pi / 2)

        return baseline_x, baseline_y, left_x, left_y, right_x, right_y


class RoadGenerator(ABC):
    @abstractmethod
    def get_road(self, seed) -> np.ndarray:
        pass


class FixedScenario(Enum):
    StraightRoad = 1
    TurnStraightAndTurn = 2
    TurnStraightAndWideTurn = 3
    StraightRoadWithNarrowGap = 4

class FixedScenarioRoadGenerator(RoadGenerator):

    def __init__(self, scenario: FixedScenario, ds: float=1.0):
        self.scenario = scenario
        self.ds = ds

    def get_road(self, seed) -> SimpleRoad:
        if self.scenario == FixedScenario.StraightRoad:
            waypoints = np.zeros((2, 4), dtype=float)
            waypoints[1, 0] = 3000
            waypoints[:, 3] = 20
            waypoints[:, 2] = np.pi/2.0
            path = get_multi_waypoint_from_np(waypoints, self.ds)
            width = np.ones(len(path)) * 2
            return SimpleRoad(path_ds=1.0, baseline_poses=path, left_width=width, right_width=width)
        if self.scenario == FixedScenario.TurnStraightAndTurn or self.scenario == FixedScenario.TurnStraightAndWideTurn:
            waypoints = np.zeros((6, 4), dtype=float)
            waypoints[0, 0] = 500
            waypoints[0, 1] = -50
            waypoints[0, 2] = -np.pi / 2.0

            waypoints[1, 0] = 0
            waypoints[1, 1] = -50
            waypoints[1, 2] = -np.pi / 2.0

            waypoints[2, 0] = 0
            waypoints[2, 1] = 0
            waypoints[2, 2] = np.pi / 2.0

            waypoints[3, 0] = 500
            waypoints[3, 1] = 0
            waypoints[3, 2] = np.pi / 2.0

            waypoints[4, 0] = 500
            waypoints[4, 1] = 50
            waypoints[4, 2] = -np.pi / 2.0

            waypoints[5, 0] = 0
            waypoints[5, 1] = 50
            waypoints[5, 2] = -np.pi / 2.0

            waypoints[:, 3] = 20
            path = get_multi_waypoint_from_np(waypoints, self.ds)
            width = np.ones(len(path)) * 2
            if self.scenario == FixedScenario.TurnStraightAndWideTurn:
                mid = int(len(width) / 2)
                width[mid:] = 5
            return SimpleRoad(path_ds=1.0, baseline_poses=path, left_width=width, right_width=width)


# THIS TAKES 5 seconds for 1000 calls.
class DubinsRoadGenerator(RoadGenerator):
    @dataclass
    class DubinsRoadParams:
        num_way_points: int = 4
        min_radius: float = 10
        max_radius: float = 50
        min_width: float = 1.75
        max_width: float = 5
        position_range: float = 700
        check_intersection: bool = False
        intersection_dist: float = 10
        path_ds: float = 1.0

    def __init__(self, params=DubinsRoadParams()):
        self.params = params

    def get_road(self, seed) -> SimpleRoad:
        while (True):
            waypoints = np.random.rand(self.params.num_way_points, 4)
            waypoints[:, 0:1] *= self.params.position_range
            waypoints[:, 2] *= 2 * np.pi
            waypoints[:, 3] = np.random.uniform(self.params.min_radius, self.params.max_radius,
                                                self.params.num_way_points)
            path = get_multi_waypoint_from_np(waypoints, self.params.path_ds)
            if not self.params.check_intersection or is_path_valid(path=path,
                                                                   min_distance=self.params.intersection_dist,
                                                                   step=self.params.path_ds):
                width = create_lateral_limits(num_points=len(path), min_width=self.params.min_width,
                                              max_width=self.params.max_width)

                # note: right now using same width left and right
                road = SimpleRoad(path_ds=self.params.path_ds, baseline_poses=path, left_width=width, right_width=width)
                return road


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
