from scoring import *

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

    #Returns differences between two boards in (x,y, new content from b2) tuples
    @classmethod
    def differences(cls, old, new):
        diffs = []
        for i in range(0,Board.SIZE):
            for j in range(0,Board.SIZE):
                if old.get(i,j) != new.get(i,j):
                    diffs.append((i,j,new.get(i,j)))
        return diffs

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
            word += self.get(sx, sy)
            if horizontal:
                sx += 1
            else:
                sy += 1

        if len(word) < 2:
            return None

        return (word, word_start, horizontal)
    
    #Scores a word given the new board and the list of diffs
    def score_word(self, (word, word_start, horizontal), diffs):

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

