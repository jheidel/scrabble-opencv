from threading import Thread, Condition
import os
from time import sleep

class Speaker(Thread):
    def __init__(self):
        Thread.__init__(self)
        self.q = []
        self.c = Condition()

    def say(self, phrase):
        with self.c:
            self.q.insert(0, phrase)
            self.c.notify()

    def _speak(self, txt):
        os.system("espeak \"%s\" 2>/dev/null >/dev/null" % txt)

    def run(self):
        while True:
            txt = ""
            with self.c:
                while len(self.q) == 0:
                    self.c.wait()
                txt = self.q.pop()
            self._speak(txt)
            sleep(0.3)

                    

    

