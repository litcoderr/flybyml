from .interface import Aircraft


class B738(Aircraft):
    def __init__(self):
        super().__init__(f_path='Aircraft/B737-800X_XP12_early_access/B737-800X/b738_4k.acf')