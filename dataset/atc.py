from __future__ import annotations

import time
import math
from threading import Thread
from queue import Queue
from pydub.playback import play

from util import offset_coord, ft_to_me, me_to_ft, haversine_distance_and_bearing
from airport import Runway
from state.state import PlaneState
from dataset.audio import play_fly_heading, play_altitude, play_downwind, play_base, play_final, \
                          nominal, left, far_left, right, far_right, glidpath, altitude, low, too_low, high, too_high


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
        play_fly_heading(str(int(self.desired_heading)).zfill(3))
        play_altitude(me_to_ft(self.desired_altitude), me_to_ft(self.desired_altitude - self.state.pos.alt))


class LandingCallout:
    def __init__(self, state: PlaneState, tgt_rwy: Runway):
        """
        state: PlaneState
        tgt_rwy: Runway
        """
        self.time = time.time()
        self.state = state
        self.tgt_rwy = tgt_rwy

        self.nominal_degree = 1
        self.correction_degree = 5
        self.nominal_alt = 10 # feet
        self.correction_alt = 50 # feet
    
    def commence(self):
        dist_rwy, heading_rwy = haversine_distance_and_bearing(
            lat1 = self.state.pos.lat,
            lon1 = self.state.pos.lon,
            ele1 = 0,
            lat2 = self.tgt_rwy.lat,
            lon2 = self.tgt_rwy.lon,
            ele2 = 0
        )
        tgt_alt = self.tgt_rwy.elev + dist_rwy * math.tan(math.radians(3))

        # altitude difference in meters
        alt_diff = self.state.pos.alt - tgt_alt
        
        # heading difference in radians
        heading_diff = math.radians(heading_rwy) - math.radians(self.tgt_rwy.bearing)

        # heading callout
        if math.degrees(abs(heading_diff)) < self.nominal_degree:
            play(glidpath)
            play(nominal)
            play_fly_heading(str(int(heading_rwy)).zfill(3))
        else:
            if heading_diff < 0:  # on the right side of the runway
                if math.degrees(abs(heading_diff)) <= self.correction_degree:
                    play(right)
                else:
                    play(far_right)
            else:  # on the left side of the runway
                if math.degrees(abs(heading_diff)) <= self.correction_degree:
                    play(left)
                else:
                    play(far_left)
        
        # altitude callout
        if me_to_ft(abs(alt_diff)) < self.nominal_alt:
            play(altitude)
            play(nominal)
        else:
            if alt_diff < 0:  # low
                if me_to_ft(abs(alt_diff)) <= self.correction_alt:
                    play(low)
                else:
                    play(too_low)
            else:  # high
                if me_to_ft(abs(alt_diff)) <= self.correction_alt:
                    play(high)
                else:
                    play(too_high)


class Stage:
    def __init__(self, tgt_rwy: Runway):
        self.tgt_rwy = tgt_rwy
        # calculate final and base coordinates
        self.base_alt_ft = 3000
        self.final = self.calc_final()
        self.base = self.calc_base()

        self.vicinity_threshold = 1000  # radius in meters to determine acquired
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
            distance = 10000
        )
        base_left = offset_coord(
            lat1 = self.final[0],
            lon1 = self.final[1],
            bearing = math.radians(self.tgt_rwy.bearing) - (math.pi / 2),
            distance = 10000
        )
        return (base_left, base_right)


class Final(Stage):
    def __init__(self, tgt_rwy: Runway):
        super().__init__(tgt_rwy)
        self.alert_period = 5
        self.has_commenced_heading = False

        play_final()
    
    def update(self, state: PlaneState) -> bool:
        if not self.has_commenced_heading:
            dist_rwy, heading_rwy = haversine_distance_and_bearing(
                lat1 = state.pos.lat,
                lon1 = state.pos.lon,
                ele1 = 0,
                lat2 = self.tgt_rwy.lat,
                lon2 = self.tgt_rwy.lon,
                ele2 = 0
            )
            tgt_alt = self.tgt_rwy.elev + dist_rwy * math.tan(math.radians(3))

            Command(state, heading_rwy, tgt_alt).commence()
            self.has_commenced_heading = True

        if self.command is None:
            self.command = LandingCallout(state, self.tgt_rwy) 
            self.command.commence()
        elif time.time() - self.command.time >= self.alert_period:
            self.command = LandingCallout(state, self.tgt_rwy) 
            self.command.commence()

        return False


class Base(Stage):
    def __init__(self, tgt_rwy: Runway):
        super().__init__(tgt_rwy)
        play_base()
    
    def transition(self) -> Final:
        return Final(self.tgt_rwy)

    def update(self, state: PlaneState) -> bool:
        acquired = False
        dist, tgt_heading = haversine_distance_and_bearing(
            lat1 = state.pos.lat,
            lon1 = state.pos.lon,
            ele1 = 0,
            lat2 = self.final[0],
            lon2 = self.final[1],
            ele2 = 0
        )
        # this is just the approximation. needs to fly lower
        dist_rwy, _ = haversine_distance_and_bearing(
            lat1 = state.pos.lat,
            lon1 = state.pos.lon,
            ele1 = 0,
            lat2 = self.tgt_rwy.lat,
            lon2 = self.tgt_rwy.lon,
            ele2 = 0
        )
        if dist <= self.vicinity_threshold:
            acquired = True

        # calculate target altitude (m)
        tgt_alt = self.tgt_rwy.elev + dist_rwy * math.tan(math.radians(3))

        if not acquired:
            if self.command is None:
                self.command = Command(state, tgt_heading, tgt_alt) 
                self.command.commence()
            elif time.time() - self.command.time >= self.alert_period:
                self.command = Command(state, tgt_heading, tgt_alt) 
                self.command.commence()
        return acquired


class Downwind(Stage):
    def __init__(self, tgt_rwy: Runway):
        super().__init__(tgt_rwy)
        play_downwind()
    
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
