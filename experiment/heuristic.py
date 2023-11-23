import math
from tkinter import Tk, Label, Entry, Button
from threading import Thread

from environment import XplaneEnvironment
from agent import AgentInterface
from state.plane import PlaneState
from controls import Controls
from aircraft.c172sp import C172SP
from state.pos import Gimpo

class Bug:
    def __init__(self, pitch, roll, spd):
        self.pitch = pitch
        self.roll = roll
        self.spd = spd

class HeuristicAgent(AgentInterface):
    def __init__(self):
        super().__init__(aircraft=C172SP())

        self.bug = Bug(0, 0, 60)
        self.control_buffer = Controls( # maintains previous control input
            elev = 0,
            ail = 0,
            rud = 0,
            thr = 0.8,
            gear = 1,
            flaps = 0
        )
        self.state_buffer = None
    
    def calc_elev(self, state: PlaneState):
        # hyper parameters
        K_p = 0.01
        K_d = 0.1

        # calculate proportional value
        error = self.bug.pitch - state.att.pitch
        proportional = K_p * error

        # calculate derivative value
        previous_error = self.bug.pitch - self.state_buffer.att.pitch
        derivative = K_d * (error - previous_error)

        rate = proportional + derivative

        elev = self.control_buffer.elev + rate 
        if elev > 1:
            elev = 1
        elif elev < -1:
            elev = -1
        return elev

    def calc_ail(self, state: PlaneState):
        # hyper parameters
        K_p = 0.01
        K_d = 0.1

        # calculate proportional value
        error = self.bug.roll - state.att.roll
        proportional = K_p * error

        # calculate derivative value
        previous_error = self.bug.roll - self.state_buffer.att.roll
        derivative = K_d * (error - previous_error)

        rate = proportional + derivative

        ail = self.control_buffer.ail + rate 
        if ail > 1:
            ail = 1
        elif ail < -1:
            ail = -1
        return ail
    
    def calc_thr(self, state: PlaneState):
        # hyper parameters
        K_p = 0.005
        K_d = 0.15

        # calculate proportional value
        error = self.bug.spd - state.spd
        proportional = K_p * error

        # calculate derivative value
        previous_error = self.bug.spd - self.state_buffer.spd
        derivative = K_d * (error - previous_error)

        rate = proportional + derivative

        thr = self.control_buffer.thr + rate 
        if thr > 1:
            thr = 1
        elif thr < 0:
            thr = 0
        return thr
    
    def sample_action(self, state: PlaneState) -> Controls:
        if self.state_buffer is None:
            self.state_buffer = state
        else:
            controls = Controls(
                elev = self.calc_elev(state),
                ail = self.calc_ail(state),
                rud = 0,
                thr = self.calc_thr(state),
                gear = 1,
                flaps = 0
            )
            self.state_buffer = state
            self.control_buffer = controls
        return self.control_buffer

class GUI(Tk):
    def __init__(self, agent: HeuristicAgent):
        super().__init__()
        self.agent = agent

        self.pitch_label = Label(self, text='pitch')
        self.pitch_label.grid(row=0, column=0)
        self.pitch_entry = Entry(self)
        self.pitch_entry.grid(row=0, column=1)
        self.pitch_set_btn = Button(self, text="set", command=self.set_pitch)
        self.pitch_set_btn.grid(row=0, column=2)
        
        self.roll_label = Label(self, text='roll')
        self.roll_label.grid(row=1, column=0)
        self.roll_entry = Entry(self)
        self.roll_entry.grid(row=1, column=1)
        self.roll_set_btn = Button(self, text="set", command=self.set_roll)
        self.roll_set_btn.grid(row=1, column=2)

        self.spd_label = Label(self, text='spd')
        self.spd_label.grid(row=2, column=0)
        self.spd_entry = Entry(self)
        self.spd_entry.grid(row=2, column=1)
        self.spd_set_btn = Button(self, text="set", command=self.set_spd)
        self.spd_set_btn.grid(row=2, column=2)
    
    def set_pitch(self):
        pitch = float(self.pitch_entry.get())
        self.agent.bug.pitch = pitch

    def set_roll(self):
        roll = float(self.roll_entry.get())
        self.agent.bug.roll = roll

    def set_spd(self):
        spd = float(self.spd_entry.get())
        self.agent.bug.spd = spd

if __name__ == "__main__":
    airport = Gimpo()
    agent = HeuristicAgent()
    env = XplaneEnvironment(agent, airport, frame_interval=0.1)

    def simulation():
        state = env.reset(heading=320, alt=1000, spd=30)
        while True:
            state = env.step(state)

    simulation_thread = Thread(target=simulation, daemon=True)
    simulation_thread.start()

    gui = GUI(agent)
    gui.mainloop()
