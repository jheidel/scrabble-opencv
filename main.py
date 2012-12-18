import os
from vision import ScrabbleVision
from scoreboard import Scoreboard
from board import Board

def ask(s):
    return str(raw_input(str(s) + "\n> "))

def speak(s):
    os.system("echo \"%s\" | espeak &> /dev/null &" % s)

scoreboard = Scoreboard() 

#Find out our players
player_count = int(ask("How many players?"))

for i in range(1, player_count+1):
    x = ask("What is Player %d's name?"% i)    


print "Starting scrabble vision..."
sv = ScrabbleVision()
sv.start()
while not sv.started:
    pass
print "Scrabble vision started. Ready."

#The game board
game_board = Board()


while True:

    #TODO Inform player turn

    print "Begin turn" 
    rsp = ask("Push enter to register move or type \"done\" to indicate the game is over").lower().strip()
    
    if rsp == "done":
        print "Game ended!"
        break
    else:
        
        #Process board and differences
        new_board = sv.get_current_board() 
        diffs = Board.differences(game_board, new_board)

        print str(diffs)
       
        new_words = set()

        for (x,y,c) in diffs:
            wh = new_board.get_word(x,y,True)
            wv = new_board.get_word(x,y,False)
            if wh is not None:
                new_words.add(wh)
            if wv is not None:
                new_words.add(wv)

        print str(new_words)

        for (wrd, pos, hz) in new_words:
            print "New word: %s" % wrd
                

        game_board.add_diffs(diffs) #Update game board w/ the changes



print "Totalling scores"
#TODO: total scores












