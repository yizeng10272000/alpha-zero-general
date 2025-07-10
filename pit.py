import Arena
from MCTS import MCTS
from othello.OthelloGame import OthelloGame
from othello.OthelloPlayers import *
from othello.pytorch.NNet import NNetWrapper as NNet

import numpy as np
from utils import *
import csv
import os

mini_othello = False  # Play in 6x6 instead of the normal 8x8.
human_vs_cpu = False

if mini_othello:
    g = OthelloGame(6)
else:
    g = OthelloGame(8)

rp = RandomPlayer(g).play
gp = GreedyOthelloPlayer(g).play
hp = HumanOthelloPlayer(g).play

# here to change the model for comparision

# nnet players
n1 = NNet(g)
if mini_othello:
    n1.load_checkpoint('./pretrained_models/othello/pytorch/','6x100x25_best.pth.tar')
else:
    n1.load_checkpoint('./pretrained_models/othello/pytorch/','8x8_100checkpoints_best.pth.tar')
args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

if human_vs_cpu:
    player2 = hp
else:
    n2 = NNet(g)
    n2.load_checkpoint('./pretrained_models/othello/pytorch/', '8x8_100checkpoints_best.pth.tar')
    args2 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
    mcts2 = MCTS(g, n2, args2)
    n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

    player2 = n2p

arena = Arena.Arena(n1p, player2, g, display=OthelloGame.display)

oneWon, twoWon, draws = arena.playGames(2, verbose=True)

print(f"Player 1 wins: {oneWon}")
print(f"Player 2 wins: {twoWon}")
print(f"Draws: {draws}")

csv_file = "arena_results.csv"
write_header = not os.path.exists(csv_file)

with open(csv_file, "a", newline='') as csvfile:
    writer = csv.writer(csvfile)
    if write_header:
        writer.writerow(["Player1_wins", "Player2_wins", "Draws"])
    writer.writerow([oneWon, twoWon, draws])
