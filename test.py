from xp import XP
import time

xp = XP()

while True:
    time.sleep(0.06)
    pos, att, gear = xp.get_posi()
    airspeed = xp.get_indicated_airspeed()
