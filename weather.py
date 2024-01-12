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
        """
        super().__init__(dref="sim/weather/region/cloud_coverage_percent",
                         value=value)


class Weather:
    def __init__(self,
                 change_mode: ChangeMode,
                 cloud_base_msl: CloudBaseMsl,
                 cloud_top_msl: CloudTopMsl,
                 cloud_coverage: CloudCoverage):
        self.change_mode = change_mode
        # clouds
        self.cloud_base_msl = cloud_base_msl
        self.cloud_top_msl = cloud_top_msl
        self.cloud_coverage = cloud_coverage
    
    def __iter__(self):
        for attr, value in self.__dict__.iteritems():
            yield attr, value
