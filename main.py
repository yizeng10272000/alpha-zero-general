import logging
import coloredlogs

from Coach import Coach
from alphatensor.TensorGame import TensorGame as Game
from alphatensor.NNet import NNetWrapper as nn
from utils import dotdict

log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')  # DEBUG 可调更详细

args = dotdict({
    'numIters': 50,                   # 总训练轮数
    'numEps': 50,                    # 每轮自对弈游戏数（生成训练样本）
    'tempThreshold': 10,             # 控制探索温度阈值
    'updateThreshold': 0.5,         # 新模型胜率超过该值才更新
    'maxlenOfQueue': 10000,          # 训练样本队列最大长度
    'numMCTSSims': 100,              # 每次MCTS模拟次数
    'arenaCompare': 20,               # 旧模型对新模型的对局数
    'cpuct': 0.5,                      # MCTS中探索参数
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
