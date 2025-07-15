import numpy as np


class TensorGame:
    def __init__(self, tensor_shape=(2, 2, 2), rank_budget=10, vector_bank_size=2):
        self.tensor_shape = tensor_shape
        self.rank_budget = rank_budget
        self.vector_bank_size = vector_bank_size

        self.target_tensor = self.init_tensor(tensor_shape)
        self.reset()

        self.action_size = self.vector_bank_size ** len(self.tensor_shape)

    def init_tensor(self, shape):
        return np.random.rand(*shape)

    def reset(self):
        self.remaining_tensor = self.target_tensor.copy()
        self.steps_taken = 0

    def getInitBoard(self):
        self.reset()
        return self.remaining_tensor

    def getBoardSize(self):
        return self.tensor_shape

    def getActionSize(self):
        return self.action_size

    def decode_action(self, action):
        idxs = []
        a = action
        for _ in range(len(self.tensor_shape)):
            idxs.append(a % self.vector_bank_size)
            a //= self.vector_bank_size
        return tuple(reversed(idxs))

    def get_vector_bank(self, dim_size):
        bank = []
        for i in range(self.vector_bank_size):
            vec = np.zeros(dim_size)
            vec[i % dim_size] = 1.0
            bank.append(vec)
        return bank

    def getRank1Tensor(self, action):
        idxs = self.decode_action(action)
        vecs = []
        for dim_size, vec_idx in zip(self.tensor_shape, idxs):
            bank = self.get_vector_bank(dim_size)
            vecs.append(bank[vec_idx])
        rank1 = np.einsum('i,j,k->ijk', vecs[0], vecs[1], vecs[2])
        return rank1

    def getNextState(self, board, player, action):
        board = board.copy()
        rank1 = self.getRank1Tensor(action)
        board -= rank1
        self.steps_taken += 1
        # 关键修改：玩家轮换，返回 -player
        return board, -player

    def getValidMoves(self, board, player):
        return np.ones(self.action_size, dtype=np.uint8)

    def getGameEnded(self, board, player):
        norm = np.linalg.norm(board)
        if norm < 1e-3:
            return 1  # 当前玩家赢
        elif self.steps_taken >= self.rank_budget:
            return -1  # 超步数限制，当前玩家输
        else:
            return 0

    def getCanonicalForm(self, board, player):
        # AlphaZero要求从当前玩家视角，board无变动即可
        return board

    def getSymmetries(self, board, pi):
        # 无对称变换
        return [(board, pi)]

    def stringRepresentation(self, board):
        return board.tobytes()
