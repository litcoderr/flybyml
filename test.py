from connect import XP
import time

xp = XP(verbose=False)

i = 0
while(True):
    time.sleep(0.5)
    posi = xp.get_posi()
    print(f'[{i}] {posi}')
    i+=1
