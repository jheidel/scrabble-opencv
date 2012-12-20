from mtTkinter import *
from threading import Thread, Lock


class ScoreBox(Thread):
    def __init__(self, d):
        Thread.__init__(self)
        self.player_list = d
        self.strvars = {}
        self.letters_left = None
        self.rnd = None
        self.started = False

    def wait_for_started(self):
        while not self.started:
            pass

    def set_score(self, player, score):
        self.wait_for_started()
        self.strvars[player].set(str(score))

    def set_letters(self, letters):
        self.wait_for_started()
        self.letters_left.set(" %d" % letters)
    
    def set_rnd(self, rnd):
        self.wait_for_started()
        self.rnd.set("#%d" % rnd)

    def update_scores(self, new_dict):
        for (k,v) in zip(new_dict.keys(), new_dict.values()):
            self.set_score(k,v)

    def kill(self):
        self.wait_for_started()
        self.master.quit()
        self.master.destroy()

    def highlight(self, player):
        i = 0
        for p in self.player_list:
            if p == player:
                color = "red"
            else:
                color = "black"
            Label(self.master, text=str(p), fg=color, font=("Helvetica", 47)).grid(row=0, column=i)
            i += 1

    def set_clock_text(self, txt, small_txt):
        self.wait_for_started()
        self.clock_txt.set(txt)
        
    def set_clock_color(self, color):
        while self.master is None:
            pass
        Label(self.master, fg=color, textvariable=self.clock_txt, font=("Helvetica", 35)).grid(row=4, column=len(self.player_list) - 1, sticky=W, padx=10, pady=10)

    def run(self):
        self.master = Tk()
        self.master.title("Scrabble Scores")
    
        #Make the titles
        i = 0
        for p in self.player_list:
            Label(self.master, text=str(p), font=("Helvetica", 47)).grid(row=0, column=i)
            i += 1
    
        #Make the variable scores
        i = 0
        for p in self.player_list:
            v = StringVar()
            self.strvars[p] = v
            v.set(str(0))
            Label(self.master, textvariable=v, font=("Helvetica", 95)).grid(row=1, column=i, padx=50)
            i += 1

        i-=1

        Label(self.master, text="Bag letters", font=("Helvetica", 20)).grid(row=2, column=0, sticky=W, padx=10)
        self.letters_left = StringVar()
        Label(self.master, textvariable=self.letters_left, font=("Helvetica", 40)).grid(row=3, column=0, sticky=W, padx=10)
        Label(self.master, text="Round", font=("Helvetica", 20)).grid(row=2, column=i, sticky=E, padx = 10)
        self.rnd = StringVar()
        self.rnd.set("#1")
        Label(self.master, textvariable=self.rnd, font=("Helvetica", 40)).grid(row=3, column=i, sticky=E, padx=10)

        self.clock_txt = StringVar() 


        self.started = True

        self.master.mainloop()
        self.master.quit()
        self.master.destroy()
        

