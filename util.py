import math
import random
import numpy as np

from weather import Weather, ChangeMode, \
    CloudBaseMsl, CloudTopMsl, CloudCoverage, CloudType, \
    Precipitation, RunwayWetness, Temperature, \
    WindMsl, WindDirection, WindSpeed, WindTurbulence, WindShearDirection, WindShearMaxSpeed


def kts_to_mps(kts):
    """
    knots to meters per second
    """
    return kts * 0.514444

def mps_to_kts(mps):
    """
    meters per second to knots
    """
    return mps * 1.94384

def mps_to_fpm(mps):
    """
    meters per second to feet per minute
    """
    return mps * 196.85

def fpm_to_mps(fpm):
    """
    feet per minute to meters per second
    """
    return fpm / 196.85

def me_to_ft(me):
    """
    meters to feets
    """
    return me * 3.28084

def ft_to_me(ft):
    """
    feets to meters
    """
    return ft * 0.3048

def haversine_distance_and_bearing(lat1: float, lon1: float, ele1: float, lat2: float, lon2: float, ele2: float):
    """
    lat1 / lon1 / lat2 / lon2: (degree)
    ele1 / ele2: elevation from mean sea level(m)
    returns:
        distance: (m)
        bearing: (degree)
    """
    # Earth radius in kilometers
    R = 6371.0

    # Converting coordinates to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Differences in coordinates
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    # Haversine formula for horizontal distance
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance_horizontal_km = R * c

    # Convert horizontal distance to meters
    distance_horizontal_m = distance_horizontal_km * 1000

    # Elevation difference in meters
    elevation_difference = ele2 - ele1

    # Total distance considering elevation
    total_distance_m = math.sqrt(distance_horizontal_m**2 + elevation_difference**2)

    # Formula for initial bearing
    x = math.sin(dlon) * math.cos(lat2_rad)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - (math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon))
    initial_bearing_rad = math.atan2(x, y)
    
    # Convert bearing from radians to degrees
    initial_bearing_deg = math.degrees(initial_bearing_rad)
    compass_bearing = (initial_bearing_deg + 360) % 360

    return total_distance_m, compass_bearing


def random_point_within_radius(lat: float, lon: float, radius_meters: float):
    # Earth's radius in kilometers
    R = 6371.0

    # Convert radius from meters to kilometers
    radius_km = radius_meters / 1000

    # Convert latitude and longitude from degrees to radians
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)

    # Random distance within the radius
    random_distance = random.uniform(0, radius_km)

    # Random bearing in all directions (0 to 360 degrees)
    random_bearing = random.uniform(0, 2 * math.pi)

    # New latitude and longitude in radians
    new_lat_rad = math.asin(math.sin(lat_rad) * math.cos(random_distance / R) +
                            math.cos(lat_rad) * math.sin(random_distance / R) * math.cos(random_bearing))

    new_lon_rad = lon_rad + math.atan2(math.sin(random_bearing) * math.sin(random_distance / R) * math.cos(lat_rad),
                                       math.cos(random_distance / R) - math.sin(lat_rad) * math.sin(new_lat_rad))

    # Convert new latitude and longitude back to degrees
    new_lat = math.degrees(new_lat_rad)
    new_lon = math.degrees(new_lon_rad)

    return new_lat, new_lon

def offset_coord(lat1, lon1, bearing, distance):
    """
    args:
        lat1: degree
        lon1: degree
        bearing: radian
        distance: meters
    returns:
        lat2: degree
        lon2: degree
    """
    R = 6371e3  # Earth radius in meters
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)

    lat2 = math.asin(math.sin(lat1) * math.cos(distance / R) +
                     math.cos(lat1) * math.sin(distance / R) * math.cos(bearing))

    lon2 = lon1 + math.atan2(math.sin(bearing) * math.sin(distance / R) * math.cos(lat1),
                             math.cos(distance / R) - math.sin(lat1) * math.sin(lat2))
    lat2 = math.degrees(lat2)
    lon2 = math.degrees(lon2)
    return lat2, lon2

def sample_weather(apt_elev: float = ft_to_me(300)) -> Weather:
    """
    apt_elev: airport elevation in meters
    returns sampled Weather object
    """
    # sample cloud
    cloud_base_msl = apt_elev + random.uniform(ft_to_me(1500), ft_to_me(4000))
    cloud_top_msl = random.uniform(cloud_base_msl + ft_to_me(5000), cloud_base_msl + ft_to_me(8000))
    cloud_coverage = random.uniform(0, 1)
    cloud_type = float(np.random.choice(np.arange(0, 4), p=[0.4, 0.4, 0.2, 0.0]))

    # sample precipitation
    precipitation = random.uniform(0, 1)
    temperature = random.uniform(-50, 50)
    if precipitation < 0.1:
        runway_wetness = 0
    else:
        if temperature < 0:
            runway_wetness = 6 * precipitation - random.uniform(-1, 0)
        else:
            runway_wetness = 7 + 6 * precipitation - random.uniform(-1, 0)

    wind_msl = random.uniform(apt_elev + ft_to_me(50), cloud_top_msl)
    wind_direction = random.uniform(0, 360)
    wind_speed = float(np.random.choice(np.arange(0, 10, 2), p=[0.4, 0.4, 0.2, 0, 0]))
    wind_turbulence = float(np.random.choice(np.arange(0, 1, 0.2), p=[0.6, 0.35, 0.05, 0, 0]))
    wind_shear_direction = random.uniform(wind_direction-5, wind_direction+5)
    if wind_shear_direction < 0:
        wind_shear_direction += 360
    wind_shear_direction %= 360
    wind_shear_max_speed = float(np.random.choice(np.arange(0,5,1), p=[0.5, 0.3, 0.15, 0.05, 0]))

    weather = Weather(
        change_mode = ChangeMode(random.randint(0, 6)),

        cloud_base_msl = CloudBaseMsl([cloud_base_msl, 0, 0]),
        cloud_top_msl = CloudTopMsl([cloud_top_msl, 0, 0]),
        cloud_coverage = CloudCoverage([cloud_coverage, 0, 0]),
        cloud_type = CloudType([cloud_type, 0, 0]),

        precipitation = Precipitation(precipitation),
        runway_wetness = RunwayWetness(runway_wetness),
        temperature = Temperature(temperature),

        wind_msl = WindMsl([wind_msl, wind_msl + ft_to_me(1100),0,0,0,0,0,0,0,0,0,0,0]),
        wind_direction = WindDirection([wind_direction,wind_direction,0,0,0,0,0,0,0,0,0,0,0]),
        wind_speed = WindSpeed([wind_speed,wind_speed,0,0,0,0,0,0,0,0,0,0,0]),
        wind_turbulence = WindTurbulence([wind_turbulence,wind_turbulence,0,0,0,0,0,0,0,0,0,0,0]),
        wind_shear_direction = WindShearDirection([wind_shear_direction,wind_shear_direction,0,0,0,0,0,0,0,0,0,0,0]),
        wind_shear_max_speed = WindShearMaxSpeed([wind_shear_max_speed,wind_shear_max_speed,0,0,0,0,0,0,0,0,0,0,0])
    )
    return weather