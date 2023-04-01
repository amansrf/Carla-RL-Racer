from ROAR.planning_module.mission_planner.mission_planner import (
    MissionPlanner,
)
from pathlib import Path
import logging
from typing import List, Optional
from ROAR.utilities_module.data_structures_models import Transform, Location, Rotation
from collections import deque
from ROAR.agent_module.agent import Agent
import numpy as np
from scipy.interpolate import interp1d


class WaypointFollowingMissionPlanner(MissionPlanner):
    """
    A mission planner that takes in a file that contains x,y,z coordinates, formulate into carla.Transform
    """

    def run_in_series(self) -> deque:
        """
        Regenerate waypoints from file
        Find the waypoint that is closest to the current vehicle location.
        return a mission plan starting from that waypoint

        Args:
            vehicle: current state of the vehicle

        Returns:
            mission plan that start from the current vehicle location
        """
        super(WaypointFollowingMissionPlanner, self).run_in_series()
        return self.produce_mission_plan()

    def __init__(self, agent: Agent):
        super().__init__(agent=agent)
        self.logger = logging.getLogger(__name__)
        self.file_path: Path = Path(self.agent.agent_settings.waypoint_file_path)
        self.mission_plan = self.produce_mission_plan()
        self._mission_plan_backup = self.mission_plan.copy()
        self.logger.debug("Path Following Mission Planner Initiated.")

    def produce_mission_plan(self) -> deque:
        """
        Generates a list of waypoints based on the input file path
        :return a list of waypoint
        """
        raw_path: List[List[float]] = self._read_data_file()
        length = self.agent.agent_settings.num_laps * len(raw_path)
        mission_plan = deque(maxlen=length)
        for coord in np.tile(raw_path, (self.agent.agent_settings.num_laps, 1)):
            if len(coord) == 3 or len(coord) == 6:
                mission_plan.append(self._raw_coord_to_transform(coord))
        self.logger.debug(f"Computed Mission path of length [{len(mission_plan)}]")

        return mission_plan

    def interpolate_waypoints(self, dense_list, thres = 0.05):
        dist = 0
        waypoints = []
        pre_pt = dense_list[0]
        cur_pt = None
        waypoints.append(pre_pt)
        for i in range(1, len(dense_list)):
            cur_pt = dense_list[i]
            dist = np.linalg.norm (np.array(pre_pt) - np.array(cur_pt))
            if dist >= 0.5:
                waypoints.append(cur_pt)
                pre_pt = cur_pt
        waypoints = np.array(waypoints)
        wptsx = waypoints[:,0]  
        wptsy = waypoints[:,1]  
        wptsz = waypoints[:,2]  
        index = np.arange(0, len(waypoints)).T
        fx = interp1d(index, wptsx, kind='cubic')
        fy = interp1d(index, wptsy, kind='cubic')
        fz = interp1d(index, wptsz, kind='cubic')
        new_waypoints = None
        for i in range(1, len(waypoints)):
            pt1 = waypoints[i - 1]
            pt2 = waypoints[i]
            dist = np.linalg.norm (pt1 - pt2)
            index = np.arange(i - 1, i, thres / (dist + 0))
            intrpx = fx(index)
            intrpy = fy(index)
            intrpz = fz(index)
            new_points = np.stack((intrpx, intrpy, intrpz), axis=0).T
            # self.waypoints_list_check(new_points)
            if new_waypoints is None:
                new_waypoints = new_points
            else:
                new_waypoints = np.vstack((new_waypoints, new_points))


        remove_index = []
        for i in range(1, len(new_waypoints)):
            pt1 = new_waypoints[i-1]
            pt2 = new_waypoints[i]
            dist = np.linalg.norm (pt1 - pt2)
            if dist <= 1e-3:
                remove_index.append(i)
        return np.delete(new_waypoints, remove_index, 0)
    
        dist = -1
        previous_pt = waypoints[0]
        output_list = []
        output_list.append(previous_pt)
        for i in range(1, len(waypoints)):
            current_pt = waypoints[i]
            dist = np.linalg.norm (current_pt - previous_pt)
            if dist >= dis_thres:
                output_list.append(current_pt)
                previous_pt = current_pt
                dist = -1
        return output_list
    
    def waypoints_list_check(self, waypoints):
        max_dist =  -1
        min_dist = 10000
        dist_list = []
        for i in range(1, len(waypoints)):
            pt1 = waypoints[i - 1]
            pt2 = waypoints[i]
            dist = np.linalg.norm (np.array(pt1) - np.array(pt2))
            dist_list.append(dist)
            if dist >= max_dist:
                max_dist = dist
            if dist <= min_dist:
                min_dist = dist
            
        print(min_dist, max_dist, len(waypoints))
        return dist_list

    def produce_single_lap_mission_plan(self):
        raw_path: List[List[float]] = self._read_data_file()
        raw_path = self.interpolate_waypoints(np.array(raw_path))
        # self.waypoints_list_check(waypoints=raw_path)
        mission_plan = deque(maxlen=len(raw_path))
        for coord in raw_path:
            if len(coord) == 3 or len(coord) == 6:
                mission_plan.append(self._raw_coord_to_transform(coord))
        self.logger.info(f"Computed Mission path of length [{len(mission_plan)}]")
        return mission_plan

    def _read_data_file(self) -> List[List[float]]:
        """
        Read data file and generate a list of (x, y, z) where each of x, y, z is of type float
        Returns:
            List of waypoints in format of [x, y, z]
        """
        result = []
        with open(self.file_path.as_posix(), "r") as f:
            for line in f:
                result.append(self._read_line(line=line))
        return result

    def _raw_coord_to_transform(self, raw: List[float]) -> Optional[Transform]:
        """
        transform coordinate to Transform instance

        Args:
            raw: coordinate in form of [x, y, z, pitch, yaw, roll]

        Returns:
            Transform instance
        """
        if len(raw) == 3:
            return Transform(
                location=Location(x=raw[0], y=raw[1], z=raw[2]),
                rotation=Rotation(pitch=0, yaw=0, roll=0),
            )
        elif len(raw) == 6:
            return Transform(
                location=Location(x=raw[0], y=raw[1], z=raw[2]),
                rotation=Rotation(roll=raw[3], pitch=raw[4], yaw=raw[5]),
            )
        else:
            self.logger.error(f"Point {raw} is invalid, skipping")
            return None

    def _read_line(self, line: str) -> List[float]:
        """
        parse a line of string of "x,y,z" into [x,y,z]
        Args:
            line: comma delimetered line

        Returns:
            [x, y, z]
        """
        try:
            x, y, z = line.split(",")
            x, y, z = float(x), float(y), float(z)
            return [x, y, z]
        except:
            x, y, z, roll, pitch, yaw = line.split(",")
            return [float(x), float(y), float(z), float(roll), float(pitch), float(yaw)]

    def restart(self):
        self.mission_plan = self._mission_plan_backup.copy()