class Position:
    def __init__(self, lat: float, lon: float, alt: float):
        self.lat = lat
        self.lon = lon
        self.alt = alt
    
    def __str__(self):
        return f"[Position] lat: {self.lat:.1f} | lon: {self.lon:.1f} | alt(m): {self.alt:.1f}"

class AirportPosition(Position):
    def __init__(self, id, lat, lon, alt):
        super().__init__(lat, lon, alt)
        self.id = id

class Gimpo(AirportPosition):
    """
    This is a position of Gimpo Airport Rwy 32R
    """
    def __init__(self):
        super().__init__(id='RKSS', lat=37.547897612, lon=126.806916813, alt=18.9268988028)
