from action import getPos, setPos
from pos import Gimpo
from att import CessnaInit

import time
from dref import hasCrashed, setIndicatedAirspeed, getIndicatedAirspeed, sendDref, getDref


pos, att, gear = getPos()
pos.alt += 1000
setPos(pos, att, 1)

i = 0
while True:

    time.sleep(0.1)
    pos, att, gear = getPos()
    speed = getIndicatedAirspeed()
    print(f'{i} {pos} {att} {speed}')
    i+=1
