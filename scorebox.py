from Tkinter import *
from threading import Thread, Lock


class ScoreBox(Thread):
    def __init__(self, d):
        Thread.__init__(self)
        self.score_dict = dict(d)
        self.strvars = {}

    def set_score(self, player, score):
        self.strvars[player].set(str(score))

    def update_scores(self, new_dict):
        for (k,v) in zip(new_dict.keys(), new_dict.values()):
            self.set_score(k,v)

    def kill(self):
        self.master.quit()

    def run(self):
        self.master = Tk()
        self.master.title("Scrabble Scores")
    
        #Make the titles
        i = 0
        for p in self.score_dict.keys():
            Label(self.master, text=str(p), font=("Helvetica", 47)).grid(row=0, column=i)
            i += 1
    
        #Make the variable scores
        i = 0
        for p in self.score_dict.keys():
            v = StringVar()
            self.strvars[p] = v
            v.set(str(self.score_dict[p]))
            Label(self.master, textvariable=v, font=("Helvetica", 95)).grid(row=1, column=i, padx=50)
            i += 1

        self.master.mainloop()
        self.master.quit()
        self.master.destroy()
        

