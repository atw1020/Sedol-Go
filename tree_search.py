"""

Author: Arthur Wesley & Camden Ross

"""

import sente
# from sente import
import keras
import numpy as np
import tensorflow
from copy import deepcopy
from keras import layers
from mcts import mcts


class gameState():
    policy_model = keras.models.load_model('policy network')
    value_model = keras.models.load_model('value network')

    def __init__(self):
        '''

        :param board: sente game passed into the tree search
        '''
        self.game = sente.Game()
        self.moves = []

    def getPossibleActions(self):
        game_array = self.game.numpy()
        game_array = np.expand_dims(game_array, axis=0)
        likely_moves = self.policy_model.predict(game_array)
        top_ten = np.argpartition(likely_moves[0], -10)[-10:]
        final_move = []

        for i in top_ten:
            x = int(i/19)
            y = i % 19
            if self.game.is_legal(x, y):
                if sente.Move(x, y, sente.stone.BLACK) not in self.moves and sente.Move(x, y, sente.stone.WHITE) not in self.moves:
                    final_move.append((x, y, self.game.get_active_player()))

        return final_move

    def takeAction(self, move):
        newState = self
        real_move = sente.Move(move[0], move[1], move[2])
        newState.game.play(real_move)
        self.moves.append(sente.Move(move[0], move[1], move[2]))
        return newState

    def isTerminal(self):
        print(self.game)
        return self.game.is_over()

    def getReward(self):
        game_array = self.game.numpy()
        game_array = np.expand_dims(game_array, axis=0)
        return self.value_model.predict(game_array)


def main():
    state = gameState()
    monte_carlo = mcts(timeLimit=1000)
    while True:
        best_action = monte_carlo.search(initialState=state)
        print(best_action)



if __name__ == "__main__":
    main()
