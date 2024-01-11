class WeatherProperty:
    def __init__(self, dref: str, value):
        """
        dref: dataref for xplane api
        value: value corresponding to dref
        """
        self.dref = dref
        self.value = value

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
            6 = Rapidly Deterioratin
        """
        super().__init__(dref="sim/weather/region/change_mode",
                         value=value)

class Weather:
    def __init__(self,
                 change_mode: ChangeMode):
        self.change_mode = change_mode
        # TODO
