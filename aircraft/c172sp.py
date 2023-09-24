from .interface import Aircraft


class C172SP(Aircraft):
    def __init__(self):
        super().__init__(f_path='Aircraft/Laminar Research/Cessna 172 SP/Cessna_172SP.acf')
