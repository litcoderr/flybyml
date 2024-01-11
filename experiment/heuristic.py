# TODO out of date. needs to be reviewed

from tkinter import Tk, Label, Entry, Button, Frame
from threading import Thread

from environment import XplaneEnvironment
from agent import AgentInterface
from state.state import PlaneState
from controls import Controls
from aircraft.c172sp import C172SP
from util import ft_to_me, me_to_ft, mps_to_fpm, fpm_to_mps

class Bug:
    def __init__(self, pitch, roll, spd, hdg, alt, vs):
        self.pitch = pitch
        self.roll = roll
        self.spd = spd
        self.hdg = hdg
        self.alt = alt
        self.vs = vs

class Mode:
    # thrust mode
    SPD = 'SPD' # maintain speed

    # lateral mode
    ROLL = 'ROLL' # maintain roll
    HDG = 'HDG' # maintain heading

    # vertical mode
    PITCH = 'PITCH' # maintain pitch
    ALTACQ = 'ALTACQ' # altitude acquire
    ALT = 'ALT' # altitude mode
    VS = 'V/S' # vertical speed mode

    def __init__(self, thrust, lateral, vertical):
        self.thrust = thrust
        self.lateral = lateral
        self.vertical = vertical

class HeuristicAgent(AgentInterface):
    def __init__(self):
        super().__init__(aircraft=C172SP())

        self.bug = Bug(0, 0, 60, 320, ft_to_me(3000), 0)
        self.mode = Mode(Mode.SPD, Mode.ROLL, Mode.PITCH)
        self.control_buffer = Controls( # maintains previous control input
            elev = 0,
            ail = 0,
            rud = 0,
            thr = 0.8,
            gear = 1,
            flaps = 0,
            trim = 0
        )
        self.state_buffer = PlaneState()
    
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
    
    def calc_target_bank(self, state: PlaneState):
        # HDG mode
        if self.mode.lateral == Mode.HDG:
            def heading_error(desired, current):
                error = desired - current 
                if error < -180:
                    return error + 360
                elif error > 180:
                    return error - 360
                else:
                    return error
            # hyper parameters
            roll_threshold = 30
            K_p = 0.0115
            K_d = 1.07

            # calculate proportional value
            error = heading_error(self.bug.hdg, state.att.yaw)
            proportional = K_p * error

            # calculate derivative value
            previous_error = heading_error(self.bug.hdg, self.state_buffer.att.yaw)
            derivative = K_d * (error - previous_error)

            rate = proportional + derivative

            roll = self.bug.roll + rate 
            if roll > roll_threshold:
                roll = roll_threshold
            elif roll < -roll_threshold:
                roll = -roll_threshold
            self.bug.roll = roll
    
    def calc_target_pitch(self, state: PlaneState):
        altacq_threshold = ft_to_me(100)
        if self.mode.vertical == Mode.VS:
            converging_dir = ((self.bug.alt - state.pos.alt) * state.vert_spd) >= 0
            if abs(self.bug.alt - state.pos.alt) < altacq_threshold and converging_dir:
                self.mode.vertical = Mode.ALTACQ

        if self.mode.vertical == Mode.ALTACQ:
            if abs(self.bug.alt - state.pos.alt) > altacq_threshold or \
                (abs(self.bug.alt - state.pos.alt) < ft_to_me(10) and state.vert_spd < fpm_to_mps(10)):
                self.bug.vs = 0
                self.mode.vertical = Mode.ALT
            else:
                vs_threshold = 2000
                # hyper parameters
                K_p = 0.0015
                K_d = 0.25

                # calculate proportional value
                error = self.bug.alt - state.pos.alt
                proportional = K_p * error

                # calculate derivative value
                previous_error = self.bug.alt - self.state_buffer.pos.alt
                derivative = K_d * (error - previous_error)

                rate = proportional + derivative

                vs = self.bug.vs + rate 
                if vs > vs_threshold:
                    vs = vs_threshold
                elif vs < -vs_threshold:
                    vs = -vs_threshold
                self.bug.vs = vs

        # update pitch in terms of vertical speed
        if self.mode.vertical != Mode.PITCH:
            vs_threshold = 15
            # hyper parameters
            K_p = 0.037
            K_d = 0.3

            # calculate proportional value
            error = self.bug.vs - state.vert_spd
            proportional = K_p * error

            # calculate derivative value
            previous_error = self.bug.vs - self.state_buffer.vert_spd
            derivative = K_d * (error - previous_error)

            rate = proportional + derivative

            vs = self.bug.pitch + rate 
            if vs > vs_threshold:
                vs = vs_threshold
            elif vs < -vs_threshold:
                vs = -vs_threshold
            self.bug.pitch = vs
    
    def highlevel_update(self, state: PlaneState):
        self.calc_target_bank(state)
        self.calc_target_pitch(state)
    
    def sample_action(self, state: PlaneState) -> Controls:
        if self.state_buffer is None:
            self.state_buffer = state
        else:
            self.highlevel_update(state)
            controls = Controls(
                elev = self.calc_elev(state),
                ail = self.calc_ail(state),
                rud = 0,
                thr = self.calc_thr(state),
                gear = 1,
                flaps = 0,
                trim = 0
            )
            self.state_buffer = state
            self.control_buffer = controls
        return self.control_buffer

class ModeViewer(Frame):
    def __init__(self, parent, agent: HeuristicAgent):
        Frame.__init__(self, parent)
        self.parent = parent
        self.agent = agent

        mode_row = 0
        self.thrust_mode_label = Label(self, text=self.agent.mode.thrust)
        self.thrust_mode_label.grid(row=mode_row, column=0)
        self.lateral_mode_label = Label(self, text=self.agent.mode.lateral)
        self.lateral_mode_label.grid(row=mode_row, column=1)
        self.vertical_mode_label = Label(self, text=self.agent.mode.vertical)
        self.vertical_mode_label.grid(row=mode_row, column=2)
        self.update()

    def update(self):
        self.thrust_mode_label['text'] = self.agent.mode.thrust
        self.lateral_mode_label['text'] = self.agent.mode.lateral
        self.vertical_mode_label['text'] = self.agent.mode.vertical
        self.after(50, self.update)

class ModeSelector(Frame):
    def __init__(self, parent, agent: HeuristicAgent):
        Frame.__init__(self, parent)

        self.agent = agent
        row_offset = 0
        self.pitch_label = Label(self, text=f'pitch: {self.agent.state_buffer.att.pitch:.1f}', width=20, anchor='w')
        self.pitch_label.grid(row=row_offset+0, column=0)
        self.pitch_entry = Entry(self)
        self.pitch_entry.grid(row=row_offset+0, column=1)
        self.pitch_set_btn = Button(self, text="set", command=self.set_pitch)
        self.pitch_set_btn.grid(row=row_offset+0, column=2)
        
        self.roll_label = Label(self, text=f'roll: {self.agent.state_buffer.att.roll:.1f}', width=20, anchor='w')
        self.roll_label.grid(row=row_offset+1, column=0)
        self.roll_entry = Entry(self)
        self.roll_entry.grid(row=row_offset+1, column=1)
        self.roll_set_btn = Button(self, text="set", command=self.set_roll)
        self.roll_set_btn.grid(row=row_offset+1, column=2)

        self.spd_label = Label(self, text=f'spd: {int(self.agent.state_buffer.spd)}', width=20, anchor='w')
        self.spd_label.grid(row=row_offset+2, column=0)
        self.spd_entry = Entry(self)
        self.spd_entry.grid(row=row_offset+2, column=1)
        self.spd_set_btn = Button(self, text="set", command=self.set_spd)
        self.spd_set_btn.grid(row=row_offset+2, column=2)

        self.hdg_label = Label(self, text=f'hdg: {str(int(self.agent.state_buffer.att.yaw)).zfill(3)}', width=20, anchor='w')
        self.hdg_label.grid(row=row_offset+3, column=0)
        self.hdg_entry = Entry(self)
        self.hdg_entry.grid(row=row_offset+3, column=1)
        self.hdg_set_btn = Button(self, text="SEL", command=self.set_hdg)
        self.hdg_set_btn.grid(row=row_offset+3, column=2)

        self.alt_label = Label(self, text=f'alt: {int(me_to_ft(self.agent.state_buffer.pos.alt))}', width=20, anchor='w')
        self.alt_label.grid(row=row_offset+4, column=0)
        self.alt_entry = Entry(self)
        self.alt_entry.grid(row=row_offset+4, column=1)
        self.alt_set_btn = Button(self, text="set", command=self.set_alt)
        self.alt_set_btn.grid(row=row_offset+4, column=2)

        self.vs_label = Label(self, text=f'v/s: {int(mps_to_fpm(self.agent.state_buffer.vert_spd))}', width=20, anchor='w')
        self.vs_label.grid(row=row_offset+5, column=0)
        self.vs_entry = Entry(self)
        self.vs_entry.grid(row=row_offset+5, column=1)
        self.vs_set_btn = Button(self, text="set", command=self.set_vs)
        self.vs_set_btn.grid(row=row_offset+5, column=2)
        self.update()

    def set_pitch(self):
        pitch = float(self.pitch_entry.get())
        self.agent.bug.pitch = pitch
        self.agent.mode.vertical = Mode.PITCH

    def set_roll(self):
        roll = float(self.roll_entry.get())
        self.agent.bug.roll = roll
        self.agent.mode.lateral = Mode.ROLL

    def set_spd(self):
        spd = float(self.spd_entry.get())
        self.agent.bug.spd = spd
        self.agent.mode.thrust = Mode.SPD

    def set_hdg(self):
        hdg = float(self.hdg_entry.get())
        self.agent.bug.hdg = hdg
        self.agent.mode.lateral = Mode.HDG

    def set_alt(self):
        alt = ft_to_me(float(self.alt_entry.get()))
        self.agent.bug.alt = alt
    
    def set_vs(self):
        vs = fpm_to_mps(float(self.vs_entry.get()))
        self.agent.bug.vs = vs
        self.agent.mode.vertical = Mode.VS

    def update(self):
        self.pitch_label['text'] = f'pitch: {self.agent.state_buffer.att.pitch:.1f}'
        self.roll_label['text'] = f'roll: {self.agent.state_buffer.att.roll:.1f}'
        self.spd_label['text'] = f'spd: {int(self.agent.state_buffer.spd)}'
        self.hdg_label['text'] = f'hdg: {str(int(self.agent.state_buffer.att.yaw)).zfill(3)}'
        self.alt_label['text'] = f'alt: {int(me_to_ft(self.agent.state_buffer.pos.alt))}'
        self.vs_label['text'] = f'v/s: {int(mps_to_fpm(self.agent.state_buffer.vert_spd))}'
        self.after(50, self.update)

class GUI(Tk):
    def __init__(self, agent: HeuristicAgent):
        super().__init__()
        self.mode_viewer = ModeViewer(self, agent)
        self.mode_viewer.pack()

        self.mode_selector = ModeSelector(self, agent)
        self.mode_selector.pack()


if __name__ == "__main__":
    agent = HeuristicAgent()
    env = XplaneEnvironment(agent, frame_interval=0.1)

    def simulation():
        state = env.reset(lat=32, lon=32, alt=1000, heading=320, spd=30)
        while True:
            state = env.step(state)

    simulation_thread = Thread(target=simulation, daemon=True)
    simulation_thread.start()

    gui = GUI(agent)
    gui.mainloop()
