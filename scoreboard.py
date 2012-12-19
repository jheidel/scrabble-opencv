TILES_IN_GAME = 20 #100 with blanks
TILES_PER_PLAYER = 7

class Scoreboard:
    def __init__(self, player_list):
        
        self.points = {}
        self.move_history = {}
        self.score_history = {}
        self.player_list = player_list
        self.tiles = {}
        self.adjustments = {}

        self.tile_count = TILES_IN_GAME 

        for p in player_list:
            self.points[p] = 0
            self.tiles[p] = TILES_PER_PLAYER
            self.tile_count -= TILES_PER_PLAYER
            self.score_history[p] = []
            self.move_history[p] = []
            self.adjustments[p] = []

        self.turn = 0
        self.turn_round = 1

    #Adds move to scoreboard returns True if a round was just completed
    def add_move(self, player, points, move):
        self.points[player] += points
        self.score_history[player].append(points)
        self.move_history[player].append(move)
        self.turn += 1

        if self.turn % len(self.player_list) == 0:
            self.turn_round += 1
            return True
        else:
            return False

    #Adds an adjustment to this player (useful for end-of-game remaining letters)
    def add_adjustment(self, player, points):
        self.adjustments[player].append(points)
        self.points[player] += points

    #Adjust game state to indicate the following number of tiles have been played by the player
    #Returns the number of tiles in the posession of the player (game over when 0)
    def subtract_tiles(self, player, tiles):
        self.tiles[player] -= tiles
        adjusted_need = min(tiles, self.tile_count)
        
        self.tiles[player] += adjusted_need
        self.tile_count -= adjusted_need

        return self.tiles[player]

    def get_tiles_in_bag(self):
        return self.tile_count


    def get_scores(self):
        a = zip(self.points.values(), self.points.keys())
        a.sort(reverse=True)
        return [(x,y) for (y,x) in a]

    def get_player_turn(self):
        return self.player_list[self.turn % len(self.player_list)]

