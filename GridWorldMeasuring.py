from python_helper import Constant as c
from python_helper import ObjectHelper, StringHelper, RandomHelper, log, SettingHelper

from reinforcement_learning import value as valueModule
from reinforcement_learning import MonteCarloEpisodeAgent, RandomAgent, Agent, Action, Environment, Episode, History, State, Reward, List, Tuple, Set, Dictionary, Id
from reinforcement_learning import trainningModule, DataCollector

from GridWorld import SquareGridEnvironmentImpl, HumanAgentImpl

from plot import printGraph

SettingHelper.updateActiveEnvironment(SettingHelper.LOCAL_ENVIRONMENT)
log.loadSettings()


MAX_REWARD: float = 1.0
MIN_REWARD: float = 0.0
WIN_REWARD: float = MAX_REWARD
DRAW_REWARD: float = MIN_REWARD
DEFAULT_REWARD: float = MIN_REWARD

AGENT_KEY: str = 'X'
ENVIRONMENT_KEY = 'GridWorld'
AGENT_SUFIX = f'{AGENT_KEY}'
REWARD_SIMBOL: str = 'R'

# BOARD_SIZE: list = List([3, 3])
# MAX_EPISODE_HISTORY_LENGHT: int = 2 * BOARD_SIZE[0] * BOARD_SIZE[1]
# TARGET_POSITIONS: List = List([[2, 2]])
#
# TOTAL_TRAINNING_ITERATIONS: int = 200
# TRAINNING_BATCH_SIZE: int = 20
# MEASURING_BATCH_SIZE: int = 2

BOARD_SIZE: list = List([13, 14])
MAX_EPISODE_HISTORY_LENGHT: int = 2 * BOARD_SIZE[0] * BOARD_SIZE[1]
TARGET_POSITIONS: List = List([[11, 11], [9, 11]])

TOTAL_TRAINNING_ITERATIONS: int = 10
TRAINNING_BATCH_SIZE: int = 10
MEASURING_BATCH_SIZE: int = 6

DEFAULT_EXPLORATION: float = 1 #0.09
DEFAULT_RETENTION: float = 0.9

agents = {
    # AGENT_KEY: HumanAgentImpl(AGENT_KEY),
    AGENT_KEY: MonteCarloEpisodeAgent(
        AGENT_KEY,
        exploration=DEFAULT_EXPLORATION,
        retention=DEFAULT_RETENTION
    )
}

environment: Environment = SquareGridEnvironmentImpl(
    WIN_REWARD,
    DRAW_REWARD,
    DEFAULT_REWARD,
    ENVIRONMENT_KEY,
    boardSize=BOARD_SIZE,
    targetPositions=TARGET_POSITIONS
)


class DataCollectorImpl(DataCollector):

    def newMeasurementData(self):
        self.measurementData = Dictionary({
            'winCount': 0,
            'loseCount': 0,
            'measurementEpisodeLenList': []
        })

    def updateMeasurementData(self, measurementEpisode: Episode):
        winner: str = self.getWinner(measurementEpisode)
        if AGENT_KEY == winner:
            self.measurementData['winCount'] += 1
        else:
            self.measurementData['loseCount'] += 1
        self.measurementData['measurementEpisodeLenList'].append(len(measurementEpisode.history))

    def getTrainningBatchResult(self, measurementData: dict) -> dict:
        self.data[trainningIteration] = {
            'winCount': self.measurementData['winCount']
            , 'loseCount': self.measurementData['loseCount']
            , 'drawCount': MEASURING_BATCH_SIZE - self.measurementData['winCount'] - self.measurementData['loseCount']
            , 'measurementEpisodeLenList': self.measurementData['measurementEpisodeLenList']
        }

    def getWinner(self, measurementEpisode: Episode) -> str:
        return measurementEpisode.environment._getWinner(
            measurementEpisode.environment.getState(),
            measurementEpisode.environment.getCurrentAgentKey()
        )


dataCollector = DataCollectorImpl()


results: dict = trainningModule.runTrainning(
    environment,
    AGENT_KEY,
    agents,
    TOTAL_TRAINNING_ITERATIONS,
    TRAINNING_BATCH_SIZE,
    MEASURING_BATCH_SIZE,
    dataCollector,
    maxEpisodeHistoryLenght=MAX_EPISODE_HISTORY_LENGHT,
    verifyEachIterationOnTrainningBatch=False,
    showBoardStatesOnTrainningBatch=False,
    verifyEachIterationOnMeasuringBatch=False,
    showBoardStatesOnMeasuringBatch=False,
    runLastEpisode=True,
    showBoardStatesOnLastEpisode=True
)

# log.prettyPython(log.debug, 'results', results, logLevel=log.DEBUG)

printGraph('Grid world', results, agents[AGENT_KEY])
