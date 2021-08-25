from python_helper import Constant as c
from python_helper import ObjectHelper, StringHelper, RandomHelper, log, SettingHelper
from reinforcement_learning import value as valueModule

from reinforcement_learning import MonteCarloEpisodeAgent, RandomAgent, Agent, Action, Environment, Episode, History, State, Reward, List, Tuple, Set, Dictionary, Id
from CartPole import CartPoleV1EnvironmentImpl
from CartPoleTrainning import runNewEpisode, runTest

from plot import printGraph


SettingHelper.updateActiveEnvironment(SettingHelper.LOCAL_ENVIRONMENT)
log.loadSettings()

MAX_REWARD: float = 1.0
MIN_REWARD: float = 0.0
WIN_REWARD: float = MAX_REWARD
DRAW_REWARD: float = MIN_REWARD
DEFAULT_REWARD: float = MIN_REWARD

PLAYER_X_VALUE: str = 'X'
REWARD_SIMBOL: str = 'R'

BOARD_SIZE: list = List([12, 12])
MAX_EPISODE_HISTORY_LENGHT: int = 4000

TOTAL_TRAINNING_ITERATIONS: int = 1000
TRAINNING_BATCH: int = 100
MEASURING_BATCH: int = 10

DEFAULT_EXPLORATION: float = 1 #0.09
EXPLORATION_REDUCING_RATIO: float = valueModule.getExplorationReducingRatio(DEFAULT_EXPLORATION, 0.05, TOTAL_TRAINNING_ITERATIONS)
DEFAULT_RETENTION: float = 0.9
ZERO_EXPLORATION_FOR_MEASUREMENT: float = 0.0
print(f'EXPLORATION_REDUCING_RATIO: {EXPLORATION_REDUCING_RATIO}')

SHOW_BOARD_STATES_ON_BATCH_TRAINNING: bool = False
SHOW_BOARD_STATES_ON_BATCH_MEASURING: bool = False

RUN_LAST_GAME: bool = True
SHOW_BOARD_STATES_ON_LAST_GAME: bool = True


AGENT_KEY = 'X'

agents = Dictionary({
    AGENT_KEY: MonteCarloEpisodeAgent(
        AGENT_KEY,
        exploration=DEFAULT_EXPLORATION,
        retention=DEFAULT_RETENTION
    )
})

environment = CartPoleV1EnvironmentImpl()

# runNewEpisode(
#     environment,
#     4000,
#     AGENT_KEY,
#     agents,
#     showStates = False,
#     verifyEachIteration = True
# )

results: dict = runTest(
    environment,
    MAX_EPISODE_HISTORY_LENGHT,
    PLAYER_X_VALUE,
    agents,
    EXPLORATION_REDUCING_RATIO,
    TOTAL_TRAINNING_ITERATIONS,
    TRAINNING_BATCH,
    SHOW_BOARD_STATES_ON_BATCH_TRAINNING,
    MEASURING_BATCH,
    SHOW_BOARD_STATES_ON_BATCH_MEASURING,
    RUN_LAST_GAME,
    SHOW_BOARD_STATES_ON_LAST_GAME
)

printGraph('Grid world', results)
