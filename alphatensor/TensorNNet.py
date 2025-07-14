import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np

class TensorNNet(nn.Module):
    def __init__(self, game, args):
        super(TensorNNet, self).__init__()
        self.board_x, self.board_y, self.board_z = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        self.fc1 = nn.Linear(self.board_x * self.board_y * self.board_z, 128)
        self.fc2 = nn.Linear(128, 128)

        self.fc_pi = nn.Linear(128, self.action_size)
        self.fc_v = nn.Linear(128, 1)

    def forward(self, s):
        s = s.view(-1, self.board_x * self.board_y * self.board_z)  # flatten
        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))

        pi = self.fc_pi(s)
        v = self.fc_v(s)

        return F.log_softmax(pi, dim=1), torch.tanh(v)

class NNetWrapper:
    def __init__(self, game):
        self.nnet = TensorNNet(game, args=None)
        self.board_x, self.board_y, self.board_z = game.getBoardSize()
        self.action_size = game.getActionSize()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nnet.to(self.device)

    def train(self, examples):
        optimizer = optim.Adam(self.nnet.parameters(), lr=0.001)
        batch_size = 64
        epochs = 10

        for epoch in range(epochs):
            self.nnet.train()
            batch_idx = 0
            while batch_idx < len(examples):
                sample_ids = np.random.randint(len(examples), size=batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))

                boards = torch.FloatTensor(np.array(boards)).to(self.device)
                target_pis = torch.FloatTensor(np.array(pis)).to(self.device)
                target_vs = torch.FloatTensor(np.array(vs)).to(self.device)

                out_pi, out_v = self.nnet(boards)

                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)

                total_loss = l_pi + l_v

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                batch_idx += batch_size

    def predict(self, board):
        self.nnet.eval()
        board = torch.FloatTensor(board.astype(np.float64))
        board = board.view(1, self.board_x, self.board_y, self.board_z).to(self.device)
        with torch.no_grad():
            pi, v = self.nnet(board)
        return torch.exp(pi).cpu().numpy()[0], v.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.mkdir(folder)
        torch.save({'state_dict': self.nnet.state_dict()}, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model in path {filepath}")
        checkpoint = torch.load(filepath, map_location=self.device)
        self.nnet.load_state_dict(checkpoint['state_dict'])
