from action import getPos, setPos
from pos import Gimpo
from att import CessnaInit

import time
from dref import hasCrashed, setIndicatedAirspeed, getIndicatedAirspeed, sendDref, getDref

pos, att, gear = getPos()
print(pos)
print(att)
pos.alt = 100

setPos(pos, att, 1)

while True:
    print(hasCrashed())
    print(getIndicatedAirspeed())
    time.sleep(1)
