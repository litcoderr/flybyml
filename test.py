from xp import XP
import time

from status.pos import Gimpo
from aircraft import C172SP

xp = XP()

airport = Gimpo()
aircraft = C172SP()
xp.set_posi(aircraft, airport, 0, 320, 0)

while True:
    time.sleep(0.16)
    try:
        pos, att, gear = xp.get_posi()
        airspeed = xp.get_indicated_airspeed()
        print(f'{pos} {att} {gear} {airspeed}')
    except:
        print("plane loading...")
