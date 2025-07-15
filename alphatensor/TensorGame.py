import numpy as np


class TensorGame:
    def __init__(self, tensor_shape=(2, 2, 2), rank_budget=10, vector_bank_size=2):
        """
        目标张量: 这里用 2x2x2 张量示例
        rank_budget: 最多 rank-1 分解次数
        vector_bank_size: 每个维度多少个候选向量
        """
        self.tensor_shape = tensor_shape
        self.rank_budget = rank_budget
        self.vector_bank_size = vector_bank_size

        self.target_tensor = self.init_tensor(tensor_shape)
        self.reset()

        self.action_size = self.vector_bank_size ** len(self.tensor_shape)

    def init_tensor(self, shape):
        """
        示例: 随机张量 or 可以换成矩阵乘法张量
        """
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
        """
        action: 单个整数索引，解码为 (u_idx, v_idx, w_idx)
        """
        idxs = []
        a = action
        for _ in range(len(self.tensor_shape)):
            idxs.append(a % self.vector_bank_size)
            a //= self.vector_bank_size
        return tuple(reversed(idxs))

    def get_vector_bank(self, dim_size):
        """
        向量库: 真实可训练时是连续向量，这里示例用单位向量组合
        """
        bank = []
        for i in range(self.vector_bank_size):
            vec = np.zeros(dim_size)
            vec[i % dim_size] = 1.0
            bank.append(vec)
        return bank

    def getRank1Tensor(self, action):
        """
        生成外积: u ⊗ v ⊗ w
        """
        idxs = self.decode_action(action)

        # 分别对每个维度用自己的 basis
        vecs = []
        for dim_size, vec_idx in zip(self.tensor_shape, idxs):
            bank = self.get_vector_bank(dim_size)
            vecs.append(bank[vec_idx])

        rank1 = np.einsum('i,j,k->ijk', vecs[0], vecs[1], vecs[2])
        return rank1

    def getNextState(self, board, player, action):
        """
        残差张量 - rank-1 外积
        """
        board = board.copy()
        rank1 = self.getRank1Tensor(action)
        board -= rank1

        self.steps_taken += 1

        return board, player

    def getValidMoves(self, board, player):
        """
        全部合法
        """
        return np.ones(self.action_size, dtype=np.uint8)

    def getGameEnded(self, board, player):
        """
        Frobenius 范数足够小 or 用完 rank_budget
        """
        norm = np.linalg.norm(board)
        if norm < 1e-3:
            return 1  # 成功
        elif self.steps_taken >= self.rank_budget:
            return 1  # 到 budget 强制结束
        else:
            return 0

    def getCanonicalForm(self, board, player):
        return board

    def getSymmetries(self, board, pi):
        return [(board, pi)]

    def stringRepresentation(self, board):
        return board.tobytes()
