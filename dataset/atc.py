import time
from threading import Thread
from queue import Queue

from airport import Runway
from state.state import PlaneState


class ATC(Thread):
    def __init__(self, queue: Queue, tgt_rwy: Runway):
        super().__init__()
        self.daemon = True
        self.queue = queue

        self.tgt_rwy = tgt_rwy
    
    def run(self):
        while True:
            msg = self.queue.get()
            if not msg["is_running"]:
                break
            if time.time() - msg["timestamp"] > 1:
                continue
            
            state: PlaneState = msg["state"]
            print(state)
            # TODO implement finding optimal path for landing and atc callouts
