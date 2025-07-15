import os
import torch
import torch.nn as nn
import torch.optim as optim


class NNetWrapper:
    def __init__(self, game):
        self.board_x, self.board_y, self.board_z = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.lr = 0.001

        self.nnet = TensorNNet(self.board_x, self.board_y, self.board_z, self.action_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nnet.to(self.device)

    def train(self, examples):
        optimizer = optim.Adam(self.nnet.parameters(), lr=self.lr)
        self.nnet.train()

        for epoch in range(10):
            for board, pi, v in examples:
                board = torch.tensor(board, dtype=torch.float32).unsqueeze(0).to(self.device)
                target_pi = torch.tensor(pi, dtype=torch.float32).unsqueeze(0).to(self.device)
                target_v = torch.tensor([v], dtype=torch.float32).to(self.device)

                out_pi, out_v = self.nnet(board)
                loss_pi = -torch.sum(target_pi * torch.log(out_pi + 1e-8))
                loss_v = (target_v - out_v).pow(2).mean()

                loss = loss_pi + loss_v

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def predict(self, board):
        board = torch.tensor(board, dtype=torch.float32).unsqueeze(0).to(self.device)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)
        return pi.cpu().numpy()[0], v.item()

    def save_checkpoint(self, folder='.', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.makedirs(folder)
        torch.save({'state_dict': self.nnet.state_dict()}, filepath)

    def load_checkpoint(self, folder='.', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model found at {filepath}")
        checkpoint = torch.load(filepath, map_location=self.device)
        self.nnet.load_state_dict(checkpoint['state_dict'])


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(channels)

    def forward(self, x):
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return torch.relu(out)


class TensorNNet(nn.Module):
    def __init__(self, x, y, z, action_size, num_channels=64, num_blocks=5):
        super(TensorNNet, self).__init__()
        self.conv1 = nn.Conv3d(1, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(num_channels)

        self.resblocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_blocks)
        ])

        self.fc1 = nn.Linear(num_channels * x * y * z, 256)
        self.fc_pi = nn.Linear(256, action_size)
        self.fc_v = nn.Linear(256, 1)

    def forward(self, s):
        s = s.unsqueeze(1) if len(s.shape) == 4 else s  # [batch, 1, x, y, z]
        s = torch.relu(self.bn1(self.conv1(s)))
        for block in self.resblocks:
            s = block(s)
        s = s.view(s.size(0), -1)
        s = torch.relu(self.fc1(s))

        pi = torch.softmax(self.fc_pi(s), dim=1)
        v = torch.tanh(self.fc_v(s))
        return pi, v
