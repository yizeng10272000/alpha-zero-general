import logging
import coloredlogs

from Coach import Coach
from alphatensor.TensorGame import TensorGame as Game
from alphatensor.NNet import NNetWrapper as nn
from utils import dotdict

log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')  # DEBUG 可调更详细

args = dotdict({
    'numIters': 5,
    'numEps': 10,
    'tempThreshold': 10,
    'updateThreshold': 0.55,
    'maxlenOfQueue': 10000,
    'numMCTSSims': 25,
    'arenaCompare': 5,
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('temp','checkpoint.pth.tar'),
    'numItersForTrainExamplesHistory': 5,
})


def main():
    log.info('Loading game...')
    g = Game()

    log.info('Loading NNet...')
    nnet = nn(g)

    if args.load_model:
        log.info('Loading checkpoint...')
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.info('No checkpoint loaded!')

    log.info('Loading Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading train examples...")
        c.loadTrainExamples()

    log.info('Starting training...')
    c.learn()


if __name__ == "__main__":
    main()
