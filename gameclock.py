from mtTkinter import *
from threading import Thread, Lock
from datetime import datetime, timedelta
from time import sleep
import os

class GameClock(Thread):
    def __init__(self, scorebox):
        Thread.__init__(self)
        self.active = True
        self.l = Lock()
        self.win = scorebox

        #base time on clock
        self.base = timedelta(0,0)

        #Time from last start point
        self.ref = datetime.now()

        self.clock_running = False

        self.warn_thresh = None
        self.alarm_thresh = None
        
        self.warn_fired = False
        self.alarm_fired = False
        self.alarm_callback = None
        
        self._update_display()

    def _reset_alarms(self):
        self.warn_fired = False
        self.alarm_fired = False

    def _warn_alarm(self):
        if not self.warn_fired:
            self.warn_fired = True
            self.win.set_clock_color("orange")
            os.system("beep -f 1000 -l 50 &>/dev/null &")

    def _alarm_alarm(self):
        if not self.alarm_fired:
            self.alarm_fired = True
            self.win.set_clock_color("red")
            os.system("beep -f 2000 -r 5 &>/dev/null &")
            if self.alarm_callback is not None:
                self.alarm_callback()

    def _update_display(self):
        cdt = self._get_delta()
        tot_secs = int(cdt.total_seconds())
        ms = cdt.microseconds / 1e3
        mins = tot_secs / 60
        secs = tot_secs - mins * 60
        self.win.set_clock_text("%02d:%02d" % (mins, secs), "%03d" % ms)

        if self.alarm_thresh is not None and tot_secs >= self.alarm_thresh:
            self._alarm_alarm()
        elif self.warn_thresh is not None and tot_secs >= self.warn_thresh:
            self._warn_alarm()

    def clock_start(self):
        with self.l:
            if self.clock_running:
                return
            self.clock_running = True
            self.ref = datetime.now()
            self.win.set_clock_color("black")

    def clock_reset(self):
        with self.l:
            self.base = timedelta(0,0)
            self.ref = datetime.now()
            self._reset_alarms()
            if self.clock_running:
                self.win.set_clock_color("black")

    def _get_delta(self):
        time = self.base

        if self.clock_running:
            dt = datetime.now() - self.ref
            time += dt

        return time

    def clock_stop(self):
        with self.l:
            if not self.clock_running:
                return
            self.base = self._get_delta()
            self.clock_running = False
            self.win.set_clock_color("blue")
        
    def kill(self):
        self.active=False

    def run(self):
        while self.active:
            with self.l:
                self._update_display()
            sleep(0.073)
