import os
from vision import ScrabbleVision
from scoreboard import Scoreboard
from board import Board
from speaker import Speaker
import twl
import signal, sys
from scoring import *
import pickle

def ask(s):
    return str(raw_input(str(s) + "\n> "))

print "Starting scrabble vision..."
sv = ScrabbleVision()
sv.start()
while not sv.started:
    pass
print "Scrabble vision started. Ready."

print "Starting speaker..."
voice = Speaker()
voice.start()

PICKLE_FILENAME = "game.state"

if len(sys.argv) == 2:
    filename = sys.argv[1] 
    (scoreboard, game_board) = pickle.load( open(filename, "rb") )
    print "Game recovered from file"
    voice.say("Resuming game!")
else:
    #Find out our players
    player_count = int(ask("How many players?"))
    player_list = []
    for i in range(1, player_count+1):
        x = ask("What is Player %d's name?"% i)    
        player_list.append(x.strip())
    
    game_board = Board()
    scoreboard = Scoreboard(player_list)

    voice.say("Starting game!")


#Register interrupt handler
def signal_handler(signal, frame):
    print "\nProgram terminating!"
    voice.kill()
    sv.kill()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


player_out = None
no_letters_warned = False

while True:

    #TODO Inform player turn
    cur_player = scoreboard.get_player_turn()

    print "-- Begin %s's turn --" % cur_player 
    print "Letters %s Players boards: %s" % (scoreboard.tile_count, scoreboard.tiles)
    voice.say("%s's turn!" % cur_player)
    rsp = ask("Push enter to register move").lower().strip()
    
    if rsp == "lookup":
        pass
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
            tiles_left = scoreboard.subtract_tiles(cur_player, len(diffs))
            game_board.add_diffs(diffs) #Update game board w/ the changes
            round_completed = scoreboard.add_move(cur_player, total_score, words_with_scores)
            if tiles_left == 0:
                #Game over!
                print "Game finished! %s is out of letters!" % cur_player
                player_out = cur_player
                voice.say("%s is out of letters. The game is over." % cur_player)
                break
            elif round_completed:
                voice.say("End of round %d." % (scoreboard.turn_round - 1))
                leader, points = scoreboard.get_scores()[0]
                voice.say("%s is in the lead with %d points." % (leader, points))
                print "Letters remaining in bag: %d" % scoreboard.get_tiles_in_bag()
                #voice.say("There are %d letters left in the bag." % scoreboard.get_tiles_in_bag())
            if scoreboard.get_tiles_in_bag() == 0 and (not no_letters_warned):
                no_letters_warned = True
                print "No more letters!"
                voice.say("There are no more letters in the bag.")

            #Pickle away game state in case of crash
            pickle.dump( (scoreboard, game_board) , open(PICKLE_FILENAME, "wb"))

    #TODO: End-game condition

#Perform end-game out of letter checks
for p in scoreboard.player_list:
    if p != player_out: 
        voice.say("Which letters does %s have left?" % p)
        letter_count = len(scoreboard.tiles[p])
        r = ask("Which %s does %s have left? (input as a list separated by commas)" % (("%d letters" % letter_count) if letter_count > 1 else "letter",p))
        letters = map(lambda x: x.strip().lower(), r.split(','))
        total_points = sum(map(get_letter_points, letters))
        
        #Give these points to the player who went out
        scoreboard.add_adjustment(p, -1 * total_points)
        scoreboard.add_adjustment(player_out, total_points)
        print "%d points transferred from %s to %s" % (total_points, p, player_out)

final_scores = scoreboard.get_scores()
winner, winning_score = final_scores[0]

print "-------------------"

print "== %s Wins! ==" % winner
print "%s has %d points" % (winner, winning_score)
voice.say("%s is the winner with a final score of %d points." % (winner, winning_score))

for player, points in final_scores[1:]:
    print "%s has %d points" % (player, points)
    voice.say("%s finished with %d points." % (player,points))

print "-------------------"



signal.pause() #Wait for ending signal











