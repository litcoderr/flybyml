from typing import List, Tuple

import random
from pathlib import Path

from util import haversine_distance_and_bearing, random_point_within_radius, ft_to_me
from xplane_airports.AptDat import AptDat, RowCode


class Runway:
    def __init__(self, apt_id: str, rwy_id: str, lat: float, lon: float, elev: float, bearing: float, width: float, length: float):
        self.apt_id = apt_id
        self.rwy_id = rwy_id
        self.lat = lat
        self.lon = lon
        self.elev = elev
        self.bearing = bearing
        self.width = width
        self.length = length

# filter landable runway
landable_rwy: List[Runway] = []
limitations = { # in meters
    'width': 45,
    'length': 2200
}

def sample_tgt_rwy_and_position(radius: float = 25000) -> Tuple[Runway, float, float]:
    """
    radius: radius of a circular region to randomly pick from (meters)
    """
    if len(landable_rwy) == 0:
        APT_DAT_PATH = Path("C:\\Program Files (x85)\\Steam\\steamapps\\common\\X-Plane 12\\Global Scenery\\Global Airports\\Earth nav data\\apt.dat")
        # read all apt data
        print("reading airports meta data ...")
        apts = AptDat(APT_DAT_PATH)
        print("finished reading airports meta data")

        for apt in apts:
            apt_id = apt.id
            apt_elevation = ft_to_me(apt.elevation_ft_amsl) # now in meters
            for rwy in apt._runway_lines():
                # check if rwy is on land and has either asphalt or concrete covering
                if not(rwy[0] == RowCode.LAND_RUNWAY and (int(rwy[2]) == 1 or int(rwy[2]) == 2)):
                    continue
                rwy_width = float(rwy[1])  # width in meters

                rwy_id_1 = rwy[8]
                rwy_lat_1 = float(rwy[9])
                rwy_lon_1 = float(rwy[10])
                rwy_id_2 = rwy[17]
                rwy_lat_2 = float(rwy[18])
                rwy_lon_2 = float(rwy[19])

                rwy_length, rwy_bearing = haversine_distance_and_bearing(rwy_lat_1, rwy_lon_1, apt_elevation,
                                                                            rwy_lat_2, rwy_lon_2, apt_elevation)

                if rwy_width >= limitations['width'] and rwy_length >= limitations["length"]:
                    landable_rwy.append(Runway(
                        apt_id, rwy_id_1, rwy_lat_1, rwy_lon_1, apt_elevation, rwy_bearing, rwy_width, rwy_length
                    ))
                    landable_rwy.append(Runway(
                        apt_id, rwy_id_2, rwy_lat_2, rwy_lon_2, apt_elevation, (rwy_bearing+180)%360, rwy_width, rwy_length
                    ))

    rwy = random.sample(landable_rwy, 1)[0]
    init_lat, init_lon = random_point_within_radius(rwy.lat, rwy.lon, radius)
    return rwy, init_lat, init_lon
