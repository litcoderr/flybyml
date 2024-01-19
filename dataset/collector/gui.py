from tkinter import Tk, Label, Button

from api import API
from airport import Runway


class StarterGui(Tk):
    def __init__(self, api: API, runway: Runway):
        super().__init__()
        self.title = "Start Data Collection"
        self.api = api
        
        self.airport = Label(self, text=f'airport: {runway.apt_id}', width=100, anchor='w')
        self.airport.grid(row=0, column=0)

        self.runway = Label(self, text=f'runway: {runway.rwy_id}', width=100, anchor='w')
        self.runway.grid(row=1, column=0)

        self.start_btn = Button(self, text="Start", command=self.start, width=100)
        self.start_btn.grid(row=2, column=0)
    
    def start(self):
        self.api.resume()
        self.destroy()
