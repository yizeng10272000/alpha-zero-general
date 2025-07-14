import numpy as np
import itertools

class TensorGame():
    def __init__(self, tensor_shape=(2, 2, 2), rank_limit=7):
        self.tensor_shape = tensor_shape
        self.rank_limit = rank_limit
        self.target_tensor = self.get_target_tensor()
        self.reset()

        # -----------------------------
        # Define a small candidate pool
        # -----------------------------
        self.vector_pool = [
            np.array([1, 0]),
            np.array([0, 1]),
            np.array([1, 1]),
            np.array([1, -1]),
        ]

        # All possible (U,V,W) index combinations
        self.action_space = list(itertools.product(
            range(len(self.vector_pool)),
            range(len(self.vector_pool)),
            range(len(self.vector_pool))
        ))

    def get_target_tensor(self):
        # Same as before: simple matrix multiplication tensor
        T = np.zeros((2, 2, 2))
        T[0, 0, 0] = 1
        T[0, 1, 1] = 1
        T[1, 0, 1] = 1
        T[1, 1, 0] = 1
        return T

    def reset(self):
        self.remaining_tensor = np.copy(self.target_tensor)
        self.decomposition = []
        self.steps = 0

    def getInitBoard(self):
        self.reset()
        return self.remaining_tensor

    def getBoardSize(self):
        return self.tensor_shape

    def getActionSize(self):
        return len(self.action_space)

    def getNextState(self, board, player, action):
        U, V, W = self.decode_action(action)
        rank1_tensor = np.einsum('i,j,k->ijk', U, V, W)
        new_board = board - rank1_tensor
        self.decomposition.append((U, V, W))
        self.steps += 1
        return new_board, player

    def getValidMoves(self, board, player):
        # For now all combinations are valid
        return np.ones(self.getActionSize(), dtype=np.int8)

    def getGameEnded(self, board, player):
        err = np.linalg.norm(board)
        if err < 1e-5:
            return 1  # success
        elif self.steps >= self.rank_limit:
            return -1  # fail
        else:
            return 0  # continue

    def getCanonicalForm(self, board, player):
        return board

    def stringRepresentation(self, board):
        return board.tobytes()

    def getSymmetries(self, board, pi):
        return [(board, pi)]

    def decode_action(self, action):
        idx_U, idx_V, idx_W = self.action_space[action]
        U = self.vector_pool[idx_U]
        V = self.vector_pool[idx_V]
        W = self.vector_pool[idx_W]
        return U, V, W
