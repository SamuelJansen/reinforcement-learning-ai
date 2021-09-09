from operator import itemgetter
from python_helper import Constant as c
from python_helper import ObjectHelper, StringHelper, RandomHelper, log, SettingHelper, ReflectionHelper
from reinforcement_learning.framework import value as valueModule
from reinforcement_learning.framework.object import Id
from reinforcement_learning.framework.value import List, Tuple, Set, Dictionary
from reinforcement_learning.framework.exception import FunctionNotImplementedException
from reinforcement_learning.framework import persistance as persistanceModule
from reinforcement_learning.framework.persistance import MongoDB

from reinforcement_learning.ai import episode as episodeModule

from reinforcement_learning.ai.agent import Agent, AgentConstants
from reinforcement_learning.ai.action import Action
from reinforcement_learning.ai.environment import Environment
from reinforcement_learning.ai.episode import Episode
from reinforcement_learning.ai.history import History
from reinforcement_learning.ai.state import State
from reinforcement_learning.ai.reward import Reward


class DataCollector:

    def __init__(self, data: Dictionary = None):
        self.data: Dictionary = Dictionary(data) if ObjectHelper.isNotNone(data) else Dictionary()

    def newMeasurementData(self):
        raise FunctionNotImplementedException()


    def updateMeasurementData(self, measurementEpisode: Episode):
        raise FunctionNotImplementedException()


    def updateTrainningBatchResult(self, trainningIteration: int):
        raise FunctionNotImplementedException()

    def getTrainningBatchResult(self) -> Dictionary:
        return Dictionary(self.data)


def prepareAgentsForNewTrainning(agents: dict, totalTrainningIterations: int, trainningBatchSize: int):
    for agentKey, agent in agents.items():
        agent.newTrainning(totalTrainningIterations, trainningBatchSize)


def finishAgentsTrainning(agents: dict):
    for agentKey, agent in agents.items():
        agent.finishTrainning()

def prepareAgentsForMeasurement(agents: dict):
    for agentKey, agent in agents.items():
        agent.freezeInternalState()


def updateAgentsAfterMeasurements(agents: dict):
    for agentKey, agent in agents.items():
        agent.updateInternalState()


def runNewEpisode(
    environment: Environment,
    agentKey: str,
    agents: Dictionary,
    maxEpisodeHistoryLenght: int,
    showStates: bool = False,
    verifyEachIteration: bool = False
):
    environment.reset()
    episode: Episode = Episode(
        environment,
        agents,
        maxEpisodeHistoryLenght
        history=History(),
        showStates=showStates
    )
    episode.run(agentPerspectiveKey=agentKey, verifyEachIteration=verifyEachIteration)
    return episode


def runTrainning(
    environment: Environment,
    agentKey: str,
    agents: dict,
    totalTrainningIterations: int,
    trainningBatchSize: int,
    measuringBatchSize: int,
    dataCollector: DataCollector,
    maxEpisodeHistoryLenght: int = episodeModule.INFINITE,
    verifyEachIterationOnTrainningBatch: bool = False,
    showBoardStatesOnTrainningBatch: bool = False,
    verifyEachIterationOnMeasuringBatch: bool = False,
    showBoardStatesOnMeasuringBatch: bool = False,
    runLastEpisode: bool = False,
    showBoardStatesOnLastEpisode: bool = False
):
    mongoDbAgents: dict = persistanceModule.loadMongoDbAgents(environment, agents)
    results = {}
    prepareAgentsForNewTrainning(agents, totalTrainningIterations, trainningBatchSize)
    for trainningIteration in range(totalTrainningIterations):
        for trainningEpisodeIndex in range(trainningBatchSize):
            episode: Episode = runNewEpisode(
                environment,
                agentKey,
                agents,
                maxHistoryLenght=maxEpisodeHistoryLenght,
                showStates=showBoardStatesOnTrainningBatch,
                verifyEachIteration=verifyEachIterationOnTrainningBatch
            )
            persistanceModule.saveEpisode(mongoDbAgents[agentKey], episode, muteLogs=True)
            if 0 == trainningEpisodeIndex % 10:
                log.debug(runTrainning, f'End of the {trainningIteration*trainningBatchSize+trainningEpisodeIndex} {environment.getKey()} {episode}. len(episode.hisotry): {len(episode.hisotry)}. {agents[agentKey].getInternalStateDescription()}')
        dataCollector.newMeasurementData()
        prepareAgentsForMeasurement(agents)
        for trainningEpisodeIndex in range(measuringBatchSize):
            measurementEpisode: Episode = runNewEpisode(
                environment,
                agentKey,
                agents,
                maxEpisodeHistoryLenght,
                showStates=showBoardStatesOnMeasuringBatch,
                verifyEachIteration=verifyEachIterationOnMeasuringBatch
            )
            dataCollector.updateMeasurementData(measurementEpisode)
        updateAgentsAfterMeasurements(agents)
        dataCollector.updateTrainningBatchResult(trainningIteration) ###- results[trainningIteration] =
    finishAgentsTrainning(agents)
    if runLastEpisode:
        lastEpisode: Episode = runNewEpisode(
            environment,
            agentKey,
            agents,
            maxEpisodeHistoryLenght,
            showStates=showBoardStatesOnLastEpisode,
            verifyEachIteration=False
        )
        log.debug(runTrainning, f'{agents[agentKey].getInternalStateDescription()}')
        log.debug(runTrainning, f'len(lastEpisode.history): {len(lastEpisode.history)}')
    persistanceModule.updateMongoDbAgents(mongoDbAgents)
    return dataCollector.getTrainningBatchResult()
