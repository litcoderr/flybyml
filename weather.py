from typing import List


class WeatherProperty:
    def __init__(self, dref: str, value):
        """
        dref: dataref for xplane api
        value: value corresponding to dref
        """
        self.dref = dref
        self.value = value
    # TODO should implement sampling method

class ChangeMode(WeatherProperty):
    def __init__(self, value: int):
        """
        value: integer
            0 = Rapidly Improving,
            1 = Improving,
            2 = Gradually Improving,
            3 = Static,
            4 = Gradually Deteriorating,
            5 = Deteriorating,
            6 = Rapidly Deteriorating
        """
        super().__init__(dref="sim/weather/region/change_mode",
                         value=value)

"""
Clouds
- at most 3 cloud layers
"""
class CloudBaseMsl(WeatherProperty):
    def __init__(self, value: List[float]):
        """
        Base altitude (MSL in meters) of 3 cloud layers
        """
        super().__init__(dref="sim/weather/region/cloud_base_msl_m",
                         value=value)

class CloudTopMsl(WeatherProperty):
    def __init__(self, value: List[float]):
        """
        Top altitude (MSL in meters) of 3 cloud layers
        """
        super().__init__(dref="sim/weather/region/cloud_tops_msl_m",
                         value=value)

class CloudCoverage(WeatherProperty):
    def __init__(self, value: List[float]):
        """
        Coverage of 3 cloud layers (0-1 range)
        How dense the clouds are
        """
        super().__init__(dref="sim/weather/region/cloud_coverage_percent",
                         value=value)

class CloudType(WeatherProperty):
    def __init__(self, value: List[float]):
        """
        Type of 3 cloud layers.
        Intermediate float values can be accepted
        0 = Cirrus
        1 = Stratus
        2 = Cumulus
        3 = Cumulo-nimbus
        """
        super().__init__(dref="sim/weather/region/cloud_type",
                         value=value)

"""
Precipitation and Temperature
"""
class Precipitation(WeatherProperty):
    def __init__(self, value: float):
        """
        Degree of rain/snow falling (0-1 range)
        """
        super().__init__(dref="sim/weather/region/rain_percent",
                         value=value)

class RunwayWetness(WeatherProperty):
    def __init__(self, value: float):
        """
        Degree of how wet the runway is (0-15 range)
        Dry = 0,
        wet(1-3),
        puddly(4-6),
        snowy(7-9),
        icy(10-12),
        snowy/icy(13-15)
        """
        super().__init__(dref="sim/weather/region/runway_friction",
                         value=value)

class Temperature(WeatherProperty):
    def __init__(self, value: float):
        """
        Temperature at MSL(Mean Sea Level) (degree Celsius)
        """
        super().__init__(dref="sim/weather/region/sealevel_temperature_c",
                         value=value)

"""
Wind
- at most 13 wind layers
Note that Xplane doesn't let you explicitly set the exact value of each property.
Xplane modifies the values based on physics engine.
"""
class WindMsl(WeatherProperty):
    def __init__(self, value: List[float]):
        """
        MSL altitude of 13 wind layers (MSL meters)
        wind layer should be atleast 1000ft apart
        """
        super().__init__(dref="sim/weather/region/wind_altitude_msl_m",
                         value=value)

class WindDirection(WeatherProperty):
    def __init__(self, value: List[float]):
        """
        Direction of 13 wind layers (0-360 degrees)
        """
        super().__init__(dref="sim/weather/region/wind_direction_degt",
                         value=value)

class WindSpeed(WeatherProperty):
    def __init__(self, value: List[float]):
        """
        Speed of 13 wind layers (>=0 m/s)
        """
        super().__init__(dref="sim/weather/region/wind_speed_msc",
                         value=value)

class Weather:
    def __init__(self,
                 change_mode: ChangeMode,
                 cloud_base_msl: CloudBaseMsl,
                 cloud_top_msl: CloudTopMsl,
                 cloud_coverage: CloudCoverage,
                 cloud_type: CloudType,
                 precipitation: Precipitation,
                 runway_wetness: RunwayWetness,
                 temperature: Temperature,
                 wind_msl: WindMsl,
                 wind_direction: WindDirection,
                 wind_speed: WindSpeed):
        self.change_mode = change_mode
        # clouds
        self.cloud_base_msl = cloud_base_msl
        self.cloud_top_msl = cloud_top_msl
        self.cloud_coverage = cloud_coverage
        self.cloud_type = cloud_type
        self.precipitation = precipitation
        self.runway_wetness = runway_wetness
        self.temperature = temperature
        self.wind_msl = wind_msl
        self.wind_direction = wind_direction
        self.wind_speed = wind_speed
