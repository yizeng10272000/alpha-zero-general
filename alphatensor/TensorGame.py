import numpy as np

class TensorGame():
    def __init__(self, tensor_shape=(2, 2, 2)):
        """
        初始化目标张量，这里用最简单的 2x2x2 张量做示例
        """
        self.tensor = self.init_tensor(tensor_shape)
        self.remaining_tensor = self.tensor.copy()

    def init_tensor(self, shape):
        """
        返回初始张量
        """
        return np.ones(shape)

    def getInitBoard(self):
        """
        初始状态
        """
        return self.remaining_tensor

    def getBoardSize(self):
        """
        张量形状
        """
        return self.remaining_tensor.shape

    def getActionSize(self):
        """
        动作空间大小
        这里用固定值：7 种候选 rank-1 张量（示例）
        """
        return 7

    def getNextState(self, board, player, action):
        """
        给定状态、玩家、动作，返回下一个状态（张量减去 rank-1）
        注意：player 对 AlphaTensor 没意义，直接返回同一个 player
        """
        board = board.copy()
        board -= self.getRank1Tensor(action)
        board = np.clip(board, 0, None)
        return board, player

    def getRank1Tensor(self, action):
        """
        返回预定义的 rank-1 张量（示例）
        """
        return np.ones_like(self.remaining_tensor) * 0.1

    def getValidMoves(self, board, player):
        """
        有效动作：这里简单示例全部都有效
        """
        return np.ones(self.getActionSize(), dtype=np.uint8)

    def getGameEnded(self, board, player):
        """
        判断是否结束
        """
        if np.allclose(board, 0, atol=1e-2):
            return 1  # 成功分解
        else:
            return 0  # 未结束

    def getCanonicalForm(self, board, player):
        """
        标准形式：对 AlphaTensor 没有视角翻转，直接返回
        """
        return board

    def getSymmetries(self, board, pi):
        """
        用于棋盘对称性增强：对张量分解无对称性，直接返回原样
        """
        return [(board, pi)]

    def stringRepresentation(self, board):
        """
        用字节表示状态
        """
        return board.tobytes()
