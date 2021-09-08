from python_helper import Constant as c
from python_helper import ObjectHelper, StringHelper, RandomHelper, log, SettingHelper

from reinforcement_learning import value as valueModule
from reinforcement_learning import MonteCarloEpisodeAgent, RandomAgent, Action, Environment, Episode, History, State, Reward, List, Tuple, Set, Dictionary, Id
from reinforcement_learning import trainningModule

from TicTacToe import TicTacToeEnvironmentImpl

from plot import printGraph

SettingHelper.updateActiveEnvironment(SettingHelper.LOCAL_ENVIRONMENT)
log.loadSettings()


PLAYER_X_KEY: str = 'X'
PLAYER_O_KEY: str = 'O'
ENVIRONMENT_KEY = 'TicTacToe'

BOARD_SIZE: int = 3
MAX_REWARD: float = 1.0
WIN_REWARD: float = MAX_REWARD
DRAW_REWARD: float = MAX_REWARD ###- MAX_REWARD * 0.75
DEFAULT_REWARD: float = 0.0
MAX_EPISODE_HISTORY_LENGHT: int = None

TOTAL_TRAINNING_ITERATIONS: int = 50
TRAINNING_BATCH_SIZE: int = 100
MEASURING_BATCH_SIZE: int = 30

# TOTAL_TRAINNING_ITERATIONS: int = 20
# TRAINNING_BATCH_SIZE: int = 4
# MEASURING_BATCH_SIZE: int = 2

DEFAULT_EXPLORATION: float = 1 # 0.09
DEFAULT_RETENTION: float = 0.9


# agents = {
#     PLAYER_X_KEY: RandomAgent(PLAYER_X_KEY),
#     PLAYER_O_KEY: MonteCarloEpisodeAgent(
#         PLAYER_O_KEY,
#         exploration=DEFAULT_EXPLORATION,
#         retention=DEFAULT_RETENTION
#     )
# }
agents = {
    PLAYER_X_KEY: MonteCarloEpisodeAgent(
        PLAYER_X_KEY,
        exploration=DEFAULT_EXPLORATION,
        retention=DEFAULT_RETENTION
    ),
    PLAYER_O_KEY: MonteCarloEpisodeAgent(
        PLAYER_O_KEY,
        exploration=DEFAULT_EXPLORATION,
        retention=DEFAULT_RETENTION
    )
}

environment: Environment = TicTacToeEnvironmentImpl(
    agents[PLAYER_X_KEY],
    agents[PLAYER_O_KEY],
    WIN_REWARD,
    DRAW_REWARD,
    DEFAULT_REWARD,
    ENVIRONMENT_KEY,
    boardSize=BOARD_SIZE,
    initialState=None
)


def getWinner(environment: Environment) -> str:
    return environment._getWinner(environment.getState())


def newMeasurementData() -> dict:
    return {
        'winCount': 0,
        'loseCount': 0,
        'measurementEpisodeLenList': []
    }


def updateMeasurementData(measurementData: dict, measurementEpisode: Episode):
    winner: str = getWinner(measurementEpisode.environment)
    if PLAYER_O_KEY == winner:
        measurementData['winCount'] += 1
    elif PLAYER_X_KEY == winner:
        measurementData['loseCount'] += 1
    # print(f'    ---> episode winner: {winner}, state: {environment.state}')
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
    PLAYER_O_KEY,
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

# for agent in agents.values():
#     agent.printActionTable()
# log.prettyPython(log.debug, 'results', results, logLevel=log.DEBUG)

printGraph('tic tac toe', results, agents[PLAYER_O_KEY])
