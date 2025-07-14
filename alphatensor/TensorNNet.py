import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class TensorNNet(nn.Module):
    def __init__(self, game, args=None):
        super(TensorNNet, self).__init__()
        self.board_x, self.board_y, self.board_z = game.getBoardSize()
        self.action_size = game.getActionSize()

        self.conv = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.board_x * self.board_y * self.board_z, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.policy_head = nn.Linear(128, self.action_size)
        self.value_head = nn.Sequential(
            nn.Linear(128, 1),
            nn.Tanh()
        )

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, s):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, dtype=torch.float32)

        # Adjust the input shape to (batch_size, board_x*board_y*board_z)
        if len(s.shape) == 3:  # For example (2,2,2)
            s = s.view(-1, self.board_x * self.board_y * self.board_z)
        elif len(s.shape) == 2 and s.shape[1] != self.board_x * self.board_y * self.board_z:
            s = s.view(-1, self.board_x * self.board_y * self.board_z)
        elif len(s.shape) == 1:
            s = s.unsqueeze(0).view(-1, self.board_x * self.board_y * self.board_z)

        s = s.to(self.device)
        x = self.conv(s)
        pi = self.policy_head(x)
        v = self.value_head(x)
        return torch.softmax(pi, dim=1), v

    def train_step(self, examples):
        self.train()
        total_loss = 0
        for board, target_pi, target_v in examples:
            board = torch.tensor(board, dtype=torch.float32).to(self.device)
            target_pi = torch.tensor(target_pi, dtype=torch.float32).to(self.device)
            target_v = torch.tensor(target_v, dtype=torch.float32).to(self.device)

            # Make sure the input shape is correct
            if len(board.shape) == 3:
                board = board.view(-1, self.board_x * self.board_y * self.board_z)

            self.optimizer.zero_grad()
            out_pi, out_v = self.forward(board)
            loss_pi = -torch.sum(target_pi * torch.log(out_pi + 1e-8))
            loss_v = nn.functional.mse_loss(out_v.view(-1), target_v)
            loss = loss_pi + loss_v
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(examples)

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        import os
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.makedirs(folder)
        torch.save({
            'state_dict': self.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        import os
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model found at {filepath}")
        checkpoint = torch.load(filepath, map_location=self.device)
        self.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])


class NNetWrapper():
    def __init__(self, game, args=None):
        self.nnet = TensorNNet(game, args)
        self.board_x, self.board_y, self.board_z = game.getBoardSize()
        self.action_size = game.getActionSize()

    def train(self, examples):
        return self.nnet.train_step(examples)

    def predict(self, board):
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet.forward(board)
        return pi.cpu().numpy()[0], v.cpu().numpy()[0][0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        self.nnet.save_checkpoint(folder, filename)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        self.nnet.load_checkpoint(folder, filename)
