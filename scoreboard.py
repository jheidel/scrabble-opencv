
class Scoreboard:
    def __init__(self, player_list):
        
        self.points = {}
        self.move_history = {}
        self.score_history = {}
        self.player_list = player_list

        for p in player_list:
            self.points[p] = 0
            self.score_history[p] = []
            self.move_history[p] = []

        self.turn = 0
        self.turn_round = 1

    #Adds move to scoreboard returns True if a round was just completed
    def add_move(self, player, points, move):
        self.points[player] += points
        self.score_history[player].append(points)
        self.move_history[player].append(move)
        self.turn += 1

        if self.turn % len(self.player_list) == 0:
            return True
            self.turn_round += 1
        else:
            return False

    def get_scores(self):
        a = zip(self.points.values(), self.points.keys())
        a.sort(reverse=True)
        return [(x,y) for (y,x) in a]

    def get_player_turn(self):
        return self.player_list[self.turn % len(self.player_list)]

