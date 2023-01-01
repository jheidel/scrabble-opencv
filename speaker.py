from threading import Thread, Condition
import os
from time import sleep

class Speaker(Thread):
    def __init__(self):
        Thread.__init__(self)
        self.daemon = True
        self.q = []
        self.c = Condition()
        self.exit = False

    def say(self, phrase):
        with self.c:
            self.q.insert(0, phrase)
            self.c.notify()

    def beep(self):
        os.system("beep -f 500 -l 200; beep -f 125 -l 400 &>/dev/null")

    def _speak(self, txt):
        if txt == "beep":
            self.beep()
        else:
            os.system("espeak \"%s\" 2>/dev/null >/dev/null" % txt)

    def kill(self):
        with self.c:
            self.exit = True
            self.c.notify()

    def run(self):
        while True:
            txt = ""
            with self.c:
                while len(self.q) == 0:
                    if self.exit:
                        print("Speaker terminating")
                        return
                    self.c.wait()
                txt = self.q.pop()
            self._speak(txt)
            sleep(0.3)

                    

    

