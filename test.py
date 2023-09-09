from connect import XP
import time

xp = XP(verbose=True)

lat = xp.get_latitude()
print(float(lat))

while True:
    # TODO test getting dref
    time.sleep(20)
    print("woke")
    lat = xp.get_latitude()
    print(lat)
