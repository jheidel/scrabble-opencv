from scoring import *
import twl  #for auto blank resolution
import configs


class Board:
    SIZE = 15
    def __init__(self):
        self.board = [None for x in range(0,Board.SIZE**2)]

    def set(self, x,y,c):
        if x < 0 or x > Board.SIZE-1 or y < 0 or y > Board.SIZE-1:
            raise ValueError
        self.board[y*Board.SIZE + x] = c

    def get(self, x,y):
        if x < 0 or x > Board.SIZE-1 or y < 0 or y > Board.SIZE-1:
            return None
        return self.board[y*Board.SIZE + x]

    #Returns a deep copy of self
    def copy(self):
        b = Board()
        b.board = list(self.board)
        return b

    #Gets the nearest points around (x,y) that are non none
    def get_nearest_not_none(self, x,y, num):
        matches = []

        def search_ring(n):
            ring_matches = []
            i = x - n
            j = y - n
            def check_and_append(a,b):
                p = self.get(a,b)
                if p is not None:
                    ring_matches.append(p)

            #Check top of box
            while i < x + n:
                check_and_append(i,j)
                i += 1
            #Check right side of box
            while j < y + n:
                check_and_append(i,j)
                j += 1
            #Check bottom of box
            while i > x - n:
                check_and_append(i,j)
                i -= 1
            #Check left side of box
            while j > y - n:
                check_and_append(i,j)
                j -= 1
            return ring_matches
        
        cring = 1
        while len(matches) < num and cring < 14:
            matches += search_ring(cring)
            cring += 1
        
        return matches[:num]


    #Returns differences between two boards in (x,y, new content from b2) tuples
    @classmethod
    def differences(cls, old, new):
        diffs = []
        for i in range(0,Board.SIZE):
            for j in range(0,Board.SIZE):
                if old.get(i,j) != new.get(i,j):
                    diffs.append((i,j,new.get(i,j)))
        return diffs

    #Takes a set of diffs and makes sure they're a valid move (i.e all letters in a single line)
    def verify_diffs(self, diffs):
        points = list(map(lambda p: (p[0],p[1]), diffs))
        x_p = list(map(lambda p: p[0], points))
        y_p = list(map(lambda p: p[1], points))

        if len(diffs) == 0:
            #Skip trun
            return True

        if len(set(x_p)) == 1: #All elements are lined up vertically
            o_p = y_p
            from_board = map(lambda m: m[0], filter(lambda p: p[1] is not None, [(y, self.get(x_p[0], y)) for y in range(0,Board.SIZE)]))
        elif len(set(y_p)) == 1: #All elements are lined up horizontally
            o_p = x_p
            from_board = map(lambda m: m[0], filter(lambda p: p[1] is not None, [(x, self.get(x, y_p[0])) for x in range(0,Board.SIZE)]))
        else:
            #Elements are not all in one line
            return False
    
        #Make sure all the values in o_p are continuous
        o_p.sort()
      
        #Get anything on the board between the min and max letters we placed
        board_relevant = list(filter(lambda x: x > o_p[0] and x < o_p[-1], from_board))
        print("Board relevant: " + str(board_relevant))
        
        #get row from board
        o_p += board_relevant
        o_p.sort()

        #Make sure all values are continuous
        paired = zip(o_p, o_p[1:])
        continuous_values = all(map(lambda p: p[1] == p[0]+1, paired))

        if not continuous_values:
            return False

        #Verify that at least one letter is next to an existing letter (unless board is empty)
        def next_to_existing_letter(p):
            return (self.get(p[0]+1,p[1]) is not None) or (self.get(p[0]-1, p[1]) is not None) or (self.get(p[0], p[1]-1) is not None) or (self.get(p[0], p[1]+1) is not None)

        def board_is_empty():
            return all(map(lambda x: x is None, self.board))

        if not any(map(next_to_existing_letter, points)):
            #Check if the board is empty
            if not board_is_empty():
                return False
       
        #All checks pass
        return True

    #Adds a set of differences to this board
    def add_diffs(self, diffs):
        for (x,y,c) in diffs:
            self.set(x,y,c)

    #Copies all non-none values from the other board to this board
    def merge(self, other):
        for i in range(0, Board.SIZE):
            for j in range(0, Board.SIZE):
                if other.get(i,j) != None:
                    self.set(i,j, other.get(i,j))

    #Gets the word containing the letter at element at x,y
    def get_word(self, x, y, horizontal=True):
        
        #Backtrack to the start of the word
        sx, sy = x, y
        while self.get(sx, sy) != None:
            if horizontal:
                sx -= 1
            else:
                sy -= 1
        
        if horizontal:
            sx += 1
        else:
            sy += 1


        word_start = (sx, sy)
        word = ""

        #Read out the word
        while self.get(sx, sy) != None:
            word += str(self.get(sx, sy))
            if horizontal:
                sx += 1
            else:
                sy += 1

        if len(word) < 2:
            return None

        return (word, word_start, horizontal)

    #Brute force test of all blanks against dictionary in an attempt to resolve it ourselves
    @classmethod
    def auto_resolve_blanks(cls, diffs, new_board, word_set):
        count = 0
        x = 0
        y = 0
        for (i,j,v) in diffs:
            if v == '-':
                x = i
                y = j
                count += 1
        if count != 1:
            return  #Auto resolver can only resolve exactly one blank at the moment

        letters = letter_points.keys()
        test_board = new_board.copy()
        possible_letters = []

        #Test word set with all possible letters
        for l in letters:
            #Make testing change in the test board
            test_board.set(x,y,l)
            new_words = []

            #Get set of new words resulting from this change
            for (word, (a,b), horizontal) in word_set:
                if '-' in word:
                    (new_wrd,j,k) = test_board.get_word(a,b,horizontal)
                    new_words.append(new_wrd)

            if all(map(twl.check, new_words)):
                possible_letters.append(l) #All new words pass dictionary check

        print("Possible letters for blank: %s" % str(possible_letters))

        if len(possible_letters) == 1:
            l = possible_letters[0]
            print("Auto resolving blank to %s" % l)
            diffs.remove((x,y,'-'))
            blnk = Blank(l)
            diffs.append((x,y,blnk))
            new_board.set(x,y,blnk)

    @classmethod
    def blank_resolver(cls, diffs, word_set, new_board, blank_prompter):
        #This method must: 
        #Blanks must be resolved at this point for the diffs and word set' strings must be fixed
        #And new board must be updated 
        
        #Attempt to automatically resolve blanks using our dictionary
        Board.auto_resolve_blanks(diffs, new_board, word_set)

        #Resolve all remaining blanks in our diff set
        for (i,j,v) in list(diffs):
            if v == '-':
                blnk = Blank(blank_prompter(i,j))
                diffs.remove((i,j,v))
                diffs.append((i,j,blnk))
                new_board.set(i,j,blnk)

        #Fix our detected word set
        for (word, pt, horizontal) in word_set:
            if '-' in word:
                (x,y) = pt
                new_word = new_board.get_word(x,y,horizontal)
                word_set.remove((word,pt,horizontal))
                word_set.add(new_word)

    #Scores a word given the new board and the list of diffs
    def score_word(self, t, diffs):
        (word, word_start, horizontal) = t

        def is_new(x,y):
            for (i,j,c) in diffs:
                if x == i and y == j:
                    return True
            return False

        c_mult = 1
        score = 0
        sx, sy = word_start
        while self.get(sx, sy) != None:
            #Score letter!
            l = self.get(sx, sy)
            lm = 1

            if is_new(sx, sy):
                #Multipliers are applicable for this letter
                lm *= get_letter_mult(sx,sy)
                c_mult *= get_word_mult(sx,sy)

            score += lm * get_letter_points(l)

            if horizontal:
                sx += 1
            else:
                sy += 1

        return score * c_mult


class AveragedBoard:
  """
  AveragedBoard represents a grid of letters with hysteresis.
  """

  def __init__(self):
    self.reset()

  def reset(self):
    self._board_letters = [[] for x in range(0,15**2)]

  def _acc(self, x, y):
    return self._board_letters[y*15 +x]

  def observe(self, x, y, c):
    """
    Observe letter `c` at position x,y
    """
    a = self._acc(x, y)
    a.insert(0,c)
    while len(a) > configs.BOARD_LETTER_HISTORY_SIZE:
      a.pop()

  def get(self, x, y):
    """
    Get the average letter at position x,y
    """
    a = self._acc(x,y)
    d = {}
    for l in a:
      if l not in d:
        d[l] = 1
      else:
        d[l] = d[l] + 1

    dd = list(zip(d.values(), d.keys()))
    dd.sort(reverse=True, key=lambda x: x[0])

    if len(dd) == 0:
      return None
    
    if dd[0][1] == None and len(dd) >= 2:
      nc = dd[0][0]
      ncf = float(nc) / len(a)
      if ncf > configs.BLANK_REQ_PERCENT:
        return None
      else:
        dd.remove(dd[0])

    return dd[0][1]
