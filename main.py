import os
from vision import ScrabbleVision
from scoreboard import Scoreboard
from board import Board
from speaker import Speaker

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

print "Starting speaker..."
voice = Speaker()
voice.start()

#The game board
game_board = Board()

voice.say("Starting game!")

while True:

    #TODO Inform player turn

    print "-- Begin Jeff's turn --" 
    voice.say("Jeff's turn!")
    rsp = ask("Push enter to register move or type \"done\" to indicate the game is over").lower().strip()
    
    if rsp == "done":
        print "Game ended!"
        voice.say("Game has ended!")
        break
    else:
        
        #Process board and differences
        new_board = sv.get_current_board() 
        diffs = Board.differences(game_board, new_board)

        #TODO: Check for letters that have gone None 

        new_words = set()

        for (x,y,c) in diffs:
            wh = new_board.get_word(x,y,True)
            wv = new_board.get_word(x,y,False)
            if wh is not None:
                new_words.add(wh)
            if wv is not None:
                new_words.add(wv)

        words_with_scores = map(lambda x: (x, new_board.score_word(x, diffs)), new_words) 

        total_score = 0
        strs = []
        for ((wrd, pos, hz), score) in words_with_scores:
            print "New word: %s -- %d points" % (wrd, score)
            strs.append("-- %s -- for %d point%s" % (wrd, score, "s" if score > 1 else ""))
            total_score += score

        extra_str = ""
        if len(diffs) >= 7:
            print "All letter bonus: +50 points"
            extra_str = ", and used all letters for 50 more points, "
            total_score += 50

        print "==Total score for turn: %d points" % total_score
       
        voice.say("Jeff plays %s %s%s." % (", and ".join(strs), extra_str, ("for a total of %d points" % total_score) if len(strs) > 1 or extra_str != "" else ""))
        
        #TODO: dictionary check


        
        rsp = ask("Commit changes? (enter \"no\" to retry, anything else to continue)").lower().strip()
        if rsp == "no":
            print "Changes aborted. Please retry."
            voice.say("Turn has been undone.")
        else:
            #Save changes to game
        
            game_board.add_diffs(diffs) #Update game board w/ the changes



print "Totalling scores"
#TODO: total scores












