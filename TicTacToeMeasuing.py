from python_helper import Constant as c
from python_helper import ObjectHelper, StringHelper, RandomHelper, log, SettingHelper
from reinforcement_learning import value as valueModule

from reinforcement_learning import MonteCarloEpisodeAgent, RandomAgent, Action, Environment, Episode, History, State, Reward, List, Tuple, Set, Dictionary, Id
from TicTacToe import TicTacToeEnvironmentImpl
from TicTacToeTrainning import runTest

from plot import printGraph

SettingHelper.updateActiveEnvironment(SettingHelper.LOCAL_ENVIRONMENT)
log.loadSettings()


BOARD_SIZE: int = 3
MAX_REWARD: float = 1.0
WIN_REWARD: float = MAX_REWARD
DRAW_REWARD: float = MAX_REWARD ###- MAX_REWARD * 0.75
DEFAULT_REWARD: float = 0.0

PLAYER_X_VALUE: str = 'X'
PLAYER_O_VALUE: str = 'O'

TOTAL_TRAINNING_ITERATIONS: int = 50
TRAINNING_BATCH: int = 100
MEASURING_BATCH: int = 30

# TOTAL_TRAINNING_ITERATIONS: int = 2000
# TRAINNING_BATCH: int = 4
# MEASURING_BATCH: int = 2

DEFAULT_EXPLORATION: float = 1 # 0.09
EXPLORATION_REDUCING_RATIO: float = valueModule.getExplorationReducingRatio(DEFAULT_EXPLORATION, 0.05, TOTAL_TRAINNING_ITERATIONS)
DEFAULT_RETENTION: float = 0.9
ZERO_EXPLORATION: float = 0.0
print(f'EXPLORATION_REDUCING_RATIO: {EXPLORATION_REDUCING_RATIO}')

SHOW_BOARD_STATES_ON_BATCH_TRAINNING: bool = False
SHOW_BOARD_STATES_ON_BATCH_MEASURING: bool = False

VERIFY_EACH_ITERATION_ON_BATCH_TRAINNING: bool = False
VERIFY_EACH_ITERATION_ON_BATCH_MEASURING: bool = False

RUN_LAST_GAME: bool = True
SHOW_BOARD_STATES_ON_LAST_GAME: bool = True


# agents = {
#     PLAYER_X_VALUE: RandomAgent(PLAYER_X_VALUE),
#     PLAYER_O_VALUE: MonteCarloEpisodeAgent(
#         PLAYER_O_VALUE,
#         exploration=DEFAULT_EXPLORATION,
#         retention=DEFAULT_RETENTION
#     )
# }
agents = {
    PLAYER_X_VALUE: MonteCarloEpisodeAgent(
        PLAYER_X_VALUE,
        exploration=DEFAULT_EXPLORATION,
        retention=DEFAULT_RETENTION
    ),
    PLAYER_O_VALUE: MonteCarloEpisodeAgent(
        PLAYER_O_VALUE,
        exploration=DEFAULT_EXPLORATION,
        retention=DEFAULT_RETENTION
    )
}

environment: Environment = TicTacToeEnvironmentImpl(
    agents[PLAYER_X_VALUE],
    agents[PLAYER_O_VALUE],
    WIN_REWARD,
    DRAW_REWARD,
    DEFAULT_REWARD,
    boardSize=BOARD_SIZE,
    initialState=None
)

results: dict = runTest(
    environment,
    PLAYER_X_VALUE,
    PLAYER_O_VALUE,
    agents,
    EXPLORATION_REDUCING_RATIO,
    EXPLORATION_REDUCING_RATIO,
    TOTAL_TRAINNING_ITERATIONS,
    TRAINNING_BATCH,
    VERIFY_EACH_ITERATION_ON_BATCH_TRAINNING,
    SHOW_BOARD_STATES_ON_BATCH_TRAINNING,
    MEASURING_BATCH,
    VERIFY_EACH_ITERATION_ON_BATCH_MEASURING,
    SHOW_BOARD_STATES_ON_BATCH_MEASURING,
    RUN_LAST_GAME,
    SHOW_BOARD_STATES_ON_LAST_GAME
    , playerXAgentSufix = 'XPlayer'
    , playerOAgentSufix = 'OPlayer'
)

# for agent in agents.values():
#     agent.printActionTable()
# log.prettyPython(log.debug, 'results', results, logLevel=log.DEBUG)

printGraph('tic tac toe', results)
