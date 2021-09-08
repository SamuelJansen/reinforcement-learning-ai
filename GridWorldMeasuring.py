from python_helper import Constant as c
from python_helper import ObjectHelper, StringHelper, RandomHelper, log, SettingHelper

from reinforcement_learning import value as valueModule
from reinforcement_learning import MonteCarloEpisodeAgent, RandomAgent, Agent, Action, Environment, Episode, History, State, Reward, List, Tuple, Set, Dictionary, Id
from reinforcement_learning import trainningModule

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
    initialState=None,
    boardSize=BOARD_SIZE,
    targetPositions=TARGET_POSITIONS
)


def getWinner(environment: Environment) -> str:
    return environment._getWinner(environment.getState(), environment.getCurrentAgentKey())


def newMeasurementData() -> dict:
    return {
        'winCount': 0,
        'loseCount': 0,
        'measurementEpisodeLenList': []
    }


def updateMeasurementData(measurementData: dict, measurementEpisode: Episode):
    winner: str = getWinner(measurementEpisode.environment)
    if AGENT_KEY == winner:
        measurementData['winCount'] += 1
    else:
        measurementData['loseCount'] += 1
    measurementData['measurementEpisodeLenList'].append(len(measurementEpisode.history))


def getTrainningBatchResult(measurementData: dict) -> dict:
    return {
        'winCount': measurementData['winCount']
        , 'loseCount': measurementData['loseCount']
        , 'drawCount': MEASURING_BATCH_SIZE - measurementData['winCount'] - measurementData['loseCount']
        , 'measurementEpisodeLenList': measurementData['measurementEpisodeLenList']
    }


results: dict = trainningModule.runTrainning(
    environment,
    AGENT_KEY,
    agents,
    TOTAL_TRAINNING_ITERATIONS,
    TRAINNING_BATCH_SIZE,
    MEASURING_BATCH_SIZE,
    verifyEachIterationOnTrainningBatch=False,
    showBoardStatesOnTrainningBatch=False,
    verifyEachIterationOnMeasuringBatch=False,
    showBoardStatesOnMeasuringBatch=False,
    runLastGame=True,
    showBoardStatesOnLastGame=True,
    maxEpisodeHistoryLenght=MAX_EPISODE_HISTORY_LENGHT,
    newMeasurementData=newMeasurementData,
    updateMeasurementData=updateMeasurementData,
    getTrainningBatchResult=getTrainningBatchResult
)

# log.prettyPython(log.debug, 'results', results, logLevel=log.DEBUG)

printGraph('Grid world', results, agents[AGENT_KEY])
