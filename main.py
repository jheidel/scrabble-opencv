import os
import pickle
import signal
import sys

from scoreboard import Scoreboard
from scoring import *
import configs
from dictionary import DictLookup

import twl

import board
import gameclock
import scorebox
import source
import speaker
import vision
import webserver

def ask(s):
    return str(raw_input(str(s) + "\n> "))

vision_source = source.FileSource()

print "Starting scrabble vision..."
sv = vision.ScrabbleVision(source=vision_source)
sv.start()
while not sv.started:
    pass
print "Scrabble vision started. Ready."

print "Starting speaker..."
voice = speaker.Speaker()
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
    
    game_board = board.Board()
    scoreboard = Scoreboard(player_list)

    voice.say("Starting game!")


#Create scorebox
sbox = scorebox.ScoreBox(scoreboard.player_list)
sbox.start()
sbox.set_letters(scoreboard.get_tiles_in_bag())
sbox.set_rnd(scoreboard.turn_round)
sbox.update_scores(scoreboard.points)

#Game clock
clock = gameclock.GameClock(sbox)
clock.start()

#STart up the webserver
serve = webserver.ScrabbleServer(game_board, scoreboard)
serve.start()

#Register interrupt handler
def signal_handler(signal, frame):
    print "\nProgram terminating!"
    voice.kill()
    sv.kill()
    sbox.kill()
    clock.kill()
    serve.kill()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


player_out = None
prev_leader = None
no_letters_warned = False
last_skip = dict(map(lambda x: (x,False), scoreboard.player_list))

while True:
    reload(configs)
    clock.warn_thresh = int(configs.WARN_TIME * 60) if configs.WARN_TIME > 0 else None
    clock.alarm_thresh = int(configs.ALARM_TIME * 60) if configs.ALARM_TIME > 0 else None

    cur_player = scoreboard.get_player_turn()

    print "====== SCORES ======"
    scores = scoreboard.get_scores()
    for player, points in scores:
        print "%s: %d points" % (player, points)
    print "===================="

    print "-- Begin %s's turn --" % cur_player 
    voice.say("%s's turn!" % cur_player)
    sbox.highlight(cur_player)
    
    serve.refresh()

    clock.clock_start()
    
    while True:
        rsp = ask("Push enter to register move").lower().strip()
        splitted = map(lambda x: x.lower().strip(), rsp.split(' '))

        if splitted[0] == "check" or splitted[0] == "lookup":
            wrd = splitted[1]
            in_dict = twl.check(wrd)
            print "The word %s %s in the dictionary." % (wrd, "is" if in_dict else "IS NOT")
        elif splitted[0] == "define":
            DictLookup(splitted[1].strip().lower())
        elif splitted[0] == "pause":
            print "Turn clock paused"
            clock.clock_stop()
        elif splitted[0] == "resume":
            print "Resuming turn clock"
            clock.clock_start()
        elif splitted[0] == "accept":
            #Hack to accept a current game board state
            #used at least once to use this program for verification
            #purposes in an already-started scrabble game
            new_board = sv.get_current_board() 
            game_board = new_board
            continue
        elif splitted[0] == "":
            break
        else:
            print "Command not recognized."

    clock.clock_stop()

    #Process board and differences
    new_board = sv.get_current_board() 
    new_board.merge(game_board)
    diffs = board.Board.differences(game_board, new_board)

    if not game_board.verify_diffs(diffs):
        #The letters played are invalid
        print "!! Invalid set of letters played. The given letters are not a single, valid"
        print "word. Please check the camera view to make sure all letters are being detected"
        print "properly and that there are no stray letters being picked up."
        print "For reference: the following diffs were detected:"
        print str(diffs)
        voice.say("Error. Invalid move. Check vision.")
        continue

    new_words = set()

    for (x,y,c) in diffs:
        wh = new_board.get_word(x,y,True)
        wv = new_board.get_word(x,y,False)
        if wh is not None:
            new_words.add(wh)
        if wv is not None:
            new_words.add(wv)

    #Prompt that the resolver function requires
    def blank_prompt(x,y):
        voice.say("Blank detected. Please input letter.")
        r = ask("Blank detected at position (%d,%d). What letter would you like to assign?" % (x,y))
        return str(r).strip().lower()

    board.Board.blank_resolver(diffs, new_words, new_board, blank_prompt) 
    #Blanks must be resolved at this point for the diffs and new_words' strings must be fixed
    #And new board must be updated 
    
    words_with_scores = map(lambda x: (new_board.score_word(x, diffs),x), new_words) 
    words_with_scores.sort(reverse=True)

    not_wrds = []
    for (score, (wrd, pos, hz)) in words_with_scores:
        if not twl.check(wrd):
            print "WARN: \"%s\" not in dictionary." % wrd
            not_wrds.append(wrd)
    if len(not_wrds) > 0:
        voice.say("beep")
        voice.say("WARNING! The word%s %s %s not in the dictionary." % ("s" if len(not_wrds) > 1 else "", " and ".join(map(lambda x: "-- %s -- " % x, not_wrds)), "are" if len(not_wrds) > 1 else "is"))
        cnt = ask("Do you wish to continue?")
        if "n" in cnt:
            continue

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
        voice.say("%s skips turn." % cur_player)
        last_skip[cur_player] = True
    else:
        last_skip[cur_player] = False
        voice.say("%s plays %s %s%s." % (cur_player, ", and ".join(strs), extra_str, ("for a total of %d points" % total_score) if len(strs) > 1 or extra_str != "" else ""))

    
    rsp = ask("Commit changes? (enter \"no\" to retry, anything else to continue)").lower().strip()
    if "n" in rsp:
        print "Changes aborted. Please retry."
        voice.say("Turn has been undone.")
    else:
        #Save changes to game state
        tiles_left = scoreboard.subtract_tiles(cur_player, len(diffs))
        game_board.add_diffs(diffs) #Update game board w/ the changes
        round_completed = scoreboard.add_move(cur_player, total_score, words_with_scores)
        finished = False
        clock.clock_reset()
        if tiles_left == 0:
            #Game over!
            print "Game finished! %s is out of letters!" % cur_player
            player_out = cur_player
            voice.say("%s is out of letters. The game is over." % cur_player)
            finished = True
        elif all(last_skip.values()) and scoreboard.get_tiles_in_bag() == 0: #All players have skipped @ end of game
            print "All players have skipped. Game is over."
            voice.say("All players have skipped their turns. The game is over.")
            finished = True
        elif round_completed:
            voice.say("End of round %d." % (scoreboard.turn_round - 1))
            leader, points = scoreboard.get_scores()[0]
            if leader != prev_leader:
                prev_leader = leader
                voice.say("%s takes the lead with %d points." % (leader, points))
            #voice.say("There are %d letters left in the bag." % scoreboard.get_tiles_in_bag())
        if scoreboard.get_tiles_in_bag() == 0 and (not no_letters_warned):
            no_letters_warned = True
            print "No more letters!"
            voice.say("There are no more letters in the bag.")

        print "Letters remaining in bag: %d" % scoreboard.get_tiles_in_bag()
        sbox.update_scores(scoreboard.points)              
        sbox.set_rnd(scoreboard.turn_round)
        sbox.set_letters(scoreboard.get_tiles_in_bag())

        #Pickle away game state in case of crash
        pickle.dump( (scoreboard, game_board) , open(PICKLE_FILENAME, "wb"))
        
        if finished:
            break

sbox.highlight(None)
clock.clock_stop()
clock.clock_reset()
serve.refresh()

#Perform end-game out of letter checks
if not all(last_skip.values()): #Game didn't end due to all-skip condition
    for p in scoreboard.player_list:
        if p != player_out: 
            letter_count = scoreboard.tiles[p]
            query = ("Which %s does %s have left?" % (("%d letters" % letter_count) if letter_count > 1 else "letter",p))
            voice.say(query)
            r = ask("%s (input as a list separated by commas)" % query)
            letters = map(lambda x: x.strip().lower(), r.split(','))
            total_points = sum(map(get_letter_points, letters))
            
            #Give these points to the player who went out
            scoreboard.add_adjustment(p, -1 * total_points)
            scoreboard.add_adjustment(player_out, total_points)
            print "%d points transferred from %s to %s" % (total_points, p, player_out)
            voice.say("%s get %d points from %s." % (player_out, total_points, p))
            sbox.update_scores(scoreboard.points)              

sbox.update_scores(scoreboard.points)              
final_scores = scoreboard.get_scores()
winner, winning_score = final_scores[0]
sbox.highlight(winner)
serve.refresh()

print "-------------------"

print "== %s Wins! ==" % winner
print "%s has %d points" % (winner, winning_score)
voice.say("%s is the winner with a final score of %d points." % (winner, winning_score))

for player, points in final_scores[1:]:
    print "%s has %d points" % (player, points)
    voice.say("%s finished with %d points." % (player,points))

print "-------------------"



sayings = ["Have a nice day.",
           "Good game.",
           "Enjoy the rest of your day.",
           "yo-lo",
           "Good bye!",
           "Good job %s." % winner,
           "Thanks for playing.",
           "Thank you.",
           "Thank you and have a pleasant day."]

from random import randint

#Say one of the random end-game sayings
voice.say(sayings[randint(0, len(sayings)-1)])

signal.pause() #Wait for ending signal (ctrl+C)











