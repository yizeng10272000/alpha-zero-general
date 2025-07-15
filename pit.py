import Arena
from MCTS import MCTS
from alphatensor.TensorGame import TensorGame as Game
from alphatensor.NNet import NNetWrapper as NNet

import numpy as np
from utils import *
import csv
import os

# 初始化张量分解任务
g = Game()

# 选手1：训练好的模型1
n1 = NNet(g)
n1.load_checkpoint('./temp', 'checkpoint_1.pth.tar')
args1 = dotdict({'numMCTSSims': 10, 'cpuct': 1.0})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

# 选手2：训练好的模型2
n2 = NNet(g)
n2.load_checkpoint('./temp', 'checkpoint_2.pth.tar')
args2 = dotdict({'numMCTSSims': 10, 'cpuct': 1.0})
mcts2 = MCTS(g, n2, args2)
n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

# 创建 Arena（对比两个模型）
arena = Arena.Arena(n1p, n2p, g)

oneWon, twoWon, draws = arena.playGames(10, verbose=True)

print(f"Model 1 wins: {oneWon}")
print(f"Model 2 wins: {twoWon}")
print(f"Draws: {draws}")

csv_file = "arena_results.csv"
write_header = not os.path.exists(csv_file)

with open(csv_file, "a", newline='') as csvfile:
    writer = csv.writer(csvfile)
    if write_header:
        writer.writerow(["Model1_wins", "Model2_wins", "Draws"])
    writer.writerow([oneWon, twoWon, draws])
