import logging
from tqdm import tqdm
import numpy as np


log = logging.getLogger(__name__)

class Arena():
    """
    Arena for AlphaTensor:
    Two agents each try to factorize the same tensor.
    The one with fewer steps (or smaller residual) wins.
    """

    def __init__(self, player1, player2, game, display=None):
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def playSingle(self, player):
        """
        让一个选手单独跑一次分解，返回：
          - 剩余残差
          - 步数
        """
        board = self.game.getInitBoard()
        curPlayer = 1
        steps = 0
        while self.game.getGameEnded(board, curPlayer) == 0:
            action = player(board)
            board, curPlayer = self.game.getNextState(board, curPlayer, action)
            steps += 1
        residual = np.sum(board)
        return residual, steps

    def playGame(self, verbose=False):
        """
        选手1和选手2各自跑一次，看谁更优
        """
        residual1, steps1 = self.playSingle(self.player1)
        residual2, steps2 = self.playSingle(self.player2)

        if verbose:
            print(f"[Player1] Steps: {steps1}, Residual: {residual1}")
            print(f"[Player2] Steps: {steps2}, Residual: {residual2}")

        # 比较分解步骤（残差都接近0的前提下）
        if steps1 < steps2:
            return 1
        elif steps1 > steps2:
            return -1
        else:
            return 0

    def playGames(self, num, verbose=False):
        """
        执行 num 场比较
        """
        oneWon = 0
        twoWon = 0
        draws = 0

        for _ in tqdm(range(num), desc="Arena.playGames"):
            result = self.playGame(verbose=verbose)
            if result == 1:
                oneWon += 1
            elif result == -1:
                twoWon += 1
            else:
                draws += 1

        return oneWon, twoWon, draws
