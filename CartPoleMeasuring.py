from python_helper import Constant as c
from python_helper import ObjectHelper, StringHelper, RandomHelper, log, SettingHelper

from reinforcement_learning import value as valueModule
from reinforcement_learning import MonteCarloEpisodeAgent, RandomAgent, Agent, Action, Environment, Episode, History, State, Reward, List, Tuple, Set, Dictionary, Id
from reinforcement_learning import trainningModule

from CartPole import CartPoleV1EnvironmentImpl

from plot import printGraph


SettingHelper.updateActiveEnvironment(SettingHelper.LOCAL_ENVIRONMENT)
log.loadSettings()

ENVIRONMENT_KEY = 'CartPoleV1'
AGENT_KEY: str = 'Y'
AGENT_SUFIX = f'{AGENT_KEY}'

# MAX_EPISODE_HISTORY_LENGHT: int = 30
# TOTAL_TRAINNING_ITERATIONS: int = 1000
# TRAINNING_BATCH_SIZE: int = 100
# MEASURING_BATCH_SIZE: int = 10

MAX_EPISODE_HISTORY_LENGHT: int = 12
TOTAL_TRAINNING_ITERATIONS: int = 10000
TRAINNING_BATCH_SIZE: int = 10
MEASURING_BATCH_SIZE: int = 10

DEFAULT_EXPLORATION: float = 1 #0.09
DEFAULT_RETENTION: float = 0.95


agents = Dictionary({
    AGENT_KEY: MonteCarloEpisodeAgent(
        AGENT_KEY,
        exploration=DEFAULT_EXPLORATION,
        retention=DEFAULT_RETENTION
    )
})

environment = CartPoleV1EnvironmentImpl(ENVIRONMENT_KEY, agentKey=AGENT_KEY)


def newMeasurementData() -> dict:
    return {
        'winCount': 0,
        'loseCount': 0,
        'measurementEpisodeLenList': []
    }


def updateMeasurementData(measurementData: dict, measurementEpisode: Episode):
    winner: str = AGENT_KEY if measurementEpisode.isMaxHistoryLenght() else None
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


def runOriginalCartPole():
    import gym
    env = gym.make('CartPole-v1')
    notDone = True
    while notDone:
        obs = env.reset()
        for step in range(100):
            action = env.action_space.sample()
            print(env.action_space, action)
            nobs, reward, done, info = env.step(action)
            print(step, nobs, reward, done, info)
            env.render()
        if reward > 0:
            input('done')
            notDone = False


def runSample(measuringBatch):
    return trainningModule.runTrainning(
        environment,
        AGENT_KEY,
        agents,
        10,
        0,
        measuringBatch,
        verifyEachIterationOnTrainningBatch=False,
        showBoardStatesOnTrainningBatch=True,
        verifyEachIterationOnMeasuringBatch=False,
        showBoardStatesOnMeasuringBatch=False,
        runLastGame=True,
        showBoardStatesOnLastGame=True,
        maxEpisodeHistoryLenght=MAX_EPISODE_HISTORY_LENGHT,
        newMeasurementData=newMeasurementData,
        updateMeasurementData=updateMeasurementData,
        getTrainningBatchResult=getTrainningBatchResult
    )


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

# results: dict = runSample(10)

printGraph('Cart Pole V1', results, agents[AGENT_KEY])

# runOriginalCartPole()
