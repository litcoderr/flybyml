from __future__ import annotations

import time
import math
from threading import Thread
from queue import Queue

from util import offset_coord, ft_to_me, me_to_ft, haversine_distance_and_bearing
from airport import Runway
from state.state import PlaneState


class Command:
    def __init__(self, state: PlaneState, desired_heading: float, desired_altitude: float):
        """
        state: PlaneState
        desired_heading: heading in degrees
        desired_altitude: altitude in meters
        """
        self.time = time.time()
        self.state = state
        self.desired_heading = desired_heading
        self.desired_altitude = desired_altitude
    
    def commence(self):
        print(f"heading: {self.desired_heading} alt: {me_to_ft(self.desired_altitude)}")
        # TODO play audio
        pass


class Stage:
    def __init__(self, tgt_rwy: Runway):
        self.tgt_rwy = tgt_rwy
        # calculate final and base coordinates
        self.base_alt_ft = 3000
        self.final = self.calc_final()
        self.base = self.calc_base()

        self.vicinity_threshold = 800  # radius in meters to determine acquired
        self.alert_period = 10  # alert period of command in seconds
        self.command = None
    
    def transition(self) -> Stage:
        raise NotImplementedError()
    
    def update(self, state: PlaneState) -> bool:
        """
        update based on state
        returns:
            boolean if this stage is acquired
        """
        raise NotImplementedError()
    
    def calc_final(self):
        return offset_coord(
            lat1 = self.tgt_rwy.lat,
            lon1 = self.tgt_rwy.lon,
            bearing = math.radians(self.tgt_rwy.bearing) + math.pi,
            distance = ft_to_me(self.base_alt_ft / math.tan(math.radians(3)))
        )

    def calc_base(self):
        base_right = offset_coord(
            lat1 = self.final[0],
            lon1 = self.final[1],
            bearing = math.radians(self.tgt_rwy.bearing) + (math.pi / 2),
            distance = 5000
        )
        base_left = offset_coord(
            lat1 = self.final[0],
            lon1 = self.final[1],
            bearing = math.radians(self.tgt_rwy.bearing) - (math.pi / 2),
            distance = 5000
        )
        return (base_left, base_right)


class Final(Stage):
    def __init__(self, tgt_rwy: Runway):
        super().__init__(tgt_rwy)
    
    def update(self, state: PlaneState) -> bool:
        # TODO
        return False


class Base(Stage):
    def __init__(self, tgt_rwy: Runway):
        super().__init__(tgt_rwy)
    
    def transition(self) -> Final:
        return Final(self.tgt_rwy)

    def update(self, state: PlaneState) -> bool:
        # TODO
        pass


class Downwind(Stage):
    def __init__(self, tgt_rwy: Runway):
        super().__init__(tgt_rwy)
    
    def transition(self) -> Downwind:
        return Base(self.tgt_rwy)

    def update(self, state: PlaneState) -> bool:
        acquired = False
        base_left, base_right = self.base
        left_dist, left_bearing = haversine_distance_and_bearing(
            lat1 = state.pos.lat,
            lon1 = state.pos.lon,
            ele1 = 0,
            lat2 = base_left[0],
            lon2 = base_left[1],
            ele2 = 0
        )
        right_dist, right_bearing = haversine_distance_and_bearing(
            lat1 = state.pos.lat,
            lon1 = state.pos.lon,
            ele1 = 0,
            lat2 = base_right[0],
            lon2 = base_right[1],
            ele2 = 0
        )
        # calculate target heading
        if left_dist < right_dist:
            tgt_heading = left_bearing
            if left_dist <= self.vicinity_threshold:
                acquired = True
        else:
            tgt_heading = right_bearing
            if right_dist <= self.vicinity_threshold:
                acquired = True
        
        # calculate target altitude (m)
        tgt_alt = self.tgt_rwy.elev + ft_to_me(self.base_alt_ft)

        if not acquired:
            if self.command is None:
                self.command = Command(state, tgt_heading, tgt_alt) 
                self.command.commence()
            elif time.time() - self.command.time >= self.alert_period:
                self.command = Command(state, tgt_heading, tgt_alt) 
                self.command.commence()
        return acquired


class ATC(Thread):
    def __init__(self, queue: Queue, tgt_rwy: Runway):
        super().__init__()
        self.daemon = True
        self.queue = queue

        self.tgt_rwy = tgt_rwy
        self.stage: Stage = Downwind(self.tgt_rwy)
    
    def run(self):
        print(f"Apt: {self.tgt_rwy.apt_id} Runway: {self.tgt_rwy.rwy_id}")
        while True:
            msg = self.queue.get()
            if not msg["is_running"]:
                break
            if time.time() - msg["timestamp"] > 1:
                continue
            
            state: PlaneState = msg["state"]
            if self.stage.update(state):
                self.stage = self.stage.transition()
