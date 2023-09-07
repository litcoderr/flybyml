from action import getPos, setPos
from pos import Gimpo
from att import CessnaInit

"""
pos, att, gear = getPos()
print(pos)
print(att)
"""

pos = Gimpo()
att = CessnaInit(320)
setPos(pos, att, 1)
