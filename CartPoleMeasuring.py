from python_helper import Constant as c
from python_helper import ObjectHelper, StringHelper, RandomHelper, log, SettingHelper
from reinforcement_learning import value as valueModule

from reinforcement_learning import MonteCarloEpisodeAgent, RandomAgent, Agent, Action, Environment, Episode, History, State, Reward, List, Tuple, Set, Dictionary, Id
from CartPole import CartPoleV1EnvironmentImpl
from CartPoleTrainning import runNewEpisode, runTest

from plot import printGraph


SettingHelper.updateActiveEnvironment(SettingHelper.LOCAL_ENVIRONMENT)
log.loadSettings()

ENVIRONMENT_KEY = 'CartPoleV1'
AGENT_KEY: str = 'X'
AGENT_SUFIX = f'{AGENT_KEY}'

# MAX_EPISODE_HISTORY_LENGHT: int = 30
#
# TOTAL_TRAINNING_ITERATIONS: int = 1000
# TRAINNING_BATCH: int = 100
# MEASURING_BATCH: int = 10

MAX_EPISODE_HISTORY_LENGHT: int = 20

TOTAL_TRAINNING_ITERATIONS: int = 800
TRAINNING_BATCH: int = 10
MEASURING_BATCH: int = 10

DEFAULT_EXPLORATION: float = 1 #0.09
EXPLORATION_REDUCING_RATIO: float = valueModule.getExplorationReducingRatio(DEFAULT_EXPLORATION, 0.05, TOTAL_TRAINNING_ITERATIONS)
DEFAULT_RETENTION: float = 0.95
ZERO_EXPLORATION_FOR_MEASUREMENT: float = 0.0
print(f'EXPLORATION_REDUCING_RATIO: {EXPLORATION_REDUCING_RATIO}')

SHOW_BOARD_STATES_ON_BATCH_TRAINNING: bool = False
SHOW_BOARD_STATES_ON_BATCH_MEASURING: bool = False

RUN_LAST_GAME: bool = False
SHOW_BOARD_STATES_ON_LAST_GAME: bool = False


agents = Dictionary({
    AGENT_KEY: MonteCarloEpisodeAgent(
        AGENT_KEY,
        exploration=DEFAULT_EXPLORATION,
        retention=DEFAULT_RETENTION
    )
})

environment = CartPoleV1EnvironmentImpl(ENVIRONMENT_KEY)

def runSample(measuringBatch):
    return runTest(
        environment,
        MAX_EPISODE_HISTORY_LENGHT,
        AGENT_KEY,
        agents,
        EXPLORATION_REDUCING_RATIO,
        10,
        0,
        False,
        measuringBatch,
        True,
        RUN_LAST_GAME,
        SHOW_BOARD_STATES_ON_LAST_GAME
    )

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
    AGENT_KEY,
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

# results: dict = runSample(1)

printGraph('Cart Pole V1', results, agents[AGENT_KEY])

# import gym
#
# env = gym.make('CartPole-v1')
# notDone = True
# while notDone:
#     obs = env.reset()
#     for step in range(100):
#         action = env.action_space.sample()
#         print(env.action_space, action)
#         nobs, reward, done, info = env.step(action)
#         print(step, nobs, reward, done, info)
#         env.render()
#     if reward > 0:
#         input('done')
#         notDone = False
