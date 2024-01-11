class Position:
    def __init__(self, lat: float, lon: float, alt: float):
        """
        alt: in meters
        """
        self.lat = lat
        self.lon = lon
        self.alt = alt
    
    def __str__(self):
        return f"[Position] lat: {self.lat:.1f} | lon: {self.lon:.1f} | alt(m): {self.alt:.1f}"
