from .interface import Aircraft


class B738(Aircraft):
    def __init__(self):
        super().__init__(f_path='Aircraft/Laminar Research/Boeing 737-800/b738.acf')