import logging
import numpy as np
from collections import deque

log = logging.getLogger(__name__)

class Coach:
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = nnet.__class__(game)  # Old model, copy the nnet structure during initialization
        self.pnet.load_state_dict(self.nnet.state_dict())  # Copy Weight
        self.args = args
        self.trainExamplesHistory = []  # Training sample history
        self.skipFirstSelfPlay = False
        self.iteration = 0

        # import MCTS
        from MCTS import MCTS  
        self.mcts = MCTS(self.game, self.nnet, self.args)

        self.updateThreshold = getattr(args, 'updateThreshold', 0.55)  # The new model acceptance threshold is 0.55 by default
        self.checkpoint = getattr(args, 'checkpoint', './checkpoints')

    def learn(self):
        for i in range(1, self.args.numIters+1):
            self.iteration = i
            log.info(f"Starting Iter #{i} ...")

            # 1. Generating training data through self-play
            iterationTrainExamples = []
            for _ in range(self.args.numEps):
                iterationTrainExamples += self.executeEpisode()

            self.trainExamplesHistory.append(iterationTrainExamples)

            # Control the length of historical data
            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                self.trainExamplesHistory.pop(0)

            # 2. Training a new network
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            self.nnet.train(trainExamples)

            # 3. The old model competes with the new model to decide whether to accept the new model
            log.info("PITTING AGAINST PREVIOUS VERSION")
            nwins, pwins, draws = self.pitModels()

            log.info(f"NEW/PREV WINS : {nwins} / {pwins} ; DRAWS : {draws}")

            win_ratio = float(nwins) / max(1, (nwins + pwins))
            if win_ratio < self.updateThreshold:
                log.info("REJECTING NEW MODEL")
                # Loading old model weights
                self.nnet.load_checkpoint(folder=self.checkpoint, filename=self.getCheckpointFile(i-1))
            else:
                log.info("ACCEPTING NEW MODEL")
                self.nnet.save_checkpoint(folder=self.checkpoint, filename=self.getCheckpointFile(i))
                self.pnet.load_state_dict(self.nnet.state_dict())

        log.info("Training completed.")

    def executeEpisode(self):
        trainExamples = []
        board = self.game.getInitBoard()
        player = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, player)
            temp = 1 if episodeStep < self.args.tempThreshold else 0
            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            symmetries = self.game.getSymmetries(canonicalBoard, pi)
            for b, p in symmetries:
                trainExamples.append([b, player, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, player = self.game.getNextState(board, player, action)
            ended = self.game.getGameEnded(board, player)
            if ended != 0:
                return [(x[0], x[2], ended*((-1)**(x[1]!=player))) for x in trainExamples]

    def pitModels(self):
        # The old model and the new model played numGames games, with statistics of wins, losses and draws
        numGames = self.args.arenaCompare
        nwins, pwins, draws = 0, 0, 0
        from Arena import Arena

        # Wrap nnet and pnet into callable functions for Arena
        def nnet_player(board):
            pi, _ = self.nnet.getActionProb(board, temp=0)
            return np.argmax(pi)

        def pnet_player(board):
            pi, _ = self.pnet.getActionProb(board, temp=0)
            return np.argmax(pi)

        arena = Arena(nnet_player, pnet_player, self.game)

        for _ in range(numGames):
            result = arena.playGame()
            if result == 1:
                nwins += 1
            elif result == -1:
                pwins += 1
            else:
                draws += 1

        return nwins, pwins, draws

    def getCheckpointFile(self, iteration):
        return f'checkpoint_{iteration}.pth.tar'
