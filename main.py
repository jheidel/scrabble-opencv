import os
from vision import ScrabbleVision
from scoreboard import Scoreboard
from board import Board
from speaker import Speaker
import twl
import signal, sys

def ask(s):
    return str(raw_input(str(s) + "\n> "))

#Find out our players
player_count = int(ask("How many players?"))
player_list = []
for i in range(1, player_count+1):
    x = ask("What is Player %d's name?"% i)    
    player_list.append(x.strip())


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
scoreboard = Scoreboard(player_list)


#Register interrupt handler
def signal_handler(signal, frame):
    print "\nProgram terminating!"
    voice.kill()
    sv.kill()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


#TODO: Load from pickled data if desired


voice.say("Starting game!")

while True:

    #TODO Inform player turn
    cur_player = scoreboard.get_player_turn()

    print "-- Begin %s's turn --" % cur_player 
    voice.say("%s's turn!" % cur_player)
    rsp = ask("Push enter to register move or type \"done\" to indicate the game is over").lower().strip()
    
    if rsp == "done":
        print "Game ended!"
        voice.say("Game has ended!")
        break
    else:
        
        #Process board and differences
        new_board = sv.get_current_board() 
        new_board.merge(game_board)
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

        words_with_scores = map(lambda x: (new_board.score_word(x, diffs),x), new_words) 
        words_with_scores.sort(reverse=True)

        total_score = 0
        strs = []
        for (score, (wrd, pos, hz)) in words_with_scores:
            print "New word: %s -- %d points" % (wrd, score)
            strs.append("-- %s -- for %d point%s" % (wrd, score, "s" if score > 1 else ""))
            total_score += score

        extra_str = ""
        if len(diffs) >= 7:
            print "All letter bonus: +50 points"
            extra_str = ", and used all letters for 50 more points, "
            total_score += 50

        print "==Total score for turn: %d points" % total_score

        if total_score == 0:
            voice.say("%s skips this turn." % cur_player)
        else:
            voice.say("%s plays %s %s%s." % (cur_player, ", and ".join(strs), extra_str, ("for a total of %d points" % total_score) if len(strs) > 1 or extra_str != "" else ""))

        
        not_wrds = []
        for (score, (wrd, pos, hz)) in words_with_scores:
            if not twl.check(wrd):
                print "WARN: \"%s\" not in dictionary." % wrd
                not_wrds.append(wrd)
        if len(not_wrds) > 0:
            voice.say("WARNING! The word%s %s %s not in the dictionary." % ("s" if len(not_wrds) > 1 else "", " and ".join(map(lambda x: "-- %s -- " % x, not_wrds)), "are" if len(not_wrds) > 1 else "is"))

         
        rsp = ask("Commit changes? (enter \"no\" to retry, anything else to continue)").lower().strip()
        if "n" in rsp:
            print "Changes aborted. Please retry."
            voice.say("Turn has been undone.")
        else:
            #Save changes to game state
            game_board.add_diffs(diffs) #Update game board w/ the changes
            round_completed = scoreboard.add_move(cur_player, total_score, words_with_scores)
            if round_completed:
                voice.say("End of round %d." % (scoreboard.turn_round - 1))
                leader, points = scoreboard.get_scores()[0]
                voice.say("%s is in the lead with %d points." % (leader, points))

            #TODO: Pickle away game state

        


    #TODO: End-game condition


print "Totalling scores"
#TODO: total scores




signal.pause() #Wait for ending signal











