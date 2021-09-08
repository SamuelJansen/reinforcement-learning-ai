from operator import itemgetter
from python_helper import Constant as c
from python_helper import ObjectHelper, StringHelper, RandomHelper, log, SettingHelper, ReflectionHelper
from reinforcement_learning.framework import value as valueModule
from reinforcement_learning.framework.object import Id
from reinforcement_learning.framework.value import List, Tuple, Set, Dictionary
from reinforcement_learning.framework.exception import FunctionNotImplementedException
from reinforcement_learning.framework import persistance as persistanceModule
from reinforcement_learning.framework.persistance import MongoDB

from reinforcement_learning.ai.agent import Agent, AgentConstants
from reinforcement_learning.ai.action import Action
from reinforcement_learning.ai.environment import Environment
from reinforcement_learning.ai.episode import Episode
from reinforcement_learning.ai.history import History
from reinforcement_learning.ai.state import State
from reinforcement_learning.ai.reward import Reward



def notImplementedNewMeasurementData():
    raise FunctionNotImplementedException()


def notImplementedUpdateMeasurementData(measurementData, measurementEpisode):
    raise FunctionNotImplementedException()


def notImplepentedGetTrainningBatchResult(measurementData):
    raise FunctionNotImplementedException()


def prepareAgentsForNewTrainning(
    agents: dict,
    totalTrainningIterations: int,
    trainningBatchSize: int,
    maxEpisodeHistoryLenght: int
):
    for agentKey, agent in agents.items():
        agent.newTrainning(totalTrainningIterations, trainningBatchSize, maxEpisodeHistoryLenght)


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
        history=History(),
        maxHistoryLenght=maxEpisodeHistoryLenght,
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
    verifyEachIterationOnTrainningBatch: bool = False,
    showBoardStatesOnTrainningBatch: bool = False,
    verifyEachIterationOnMeasuringBatch: bool = False,
    showBoardStatesOnMeasuringBatch: bool = False,
    runLastGame: bool = False,
    showBoardStatesOnLastGame: bool = False,
    maxEpisodeHistoryLenght: int = None,
    newMeasurementData = notImplementedNewMeasurementData,
    updateMeasurementData = notImplementedUpdateMeasurementData,
    getTrainningBatchResult = notImplepentedGetTrainningBatchResult
):
    mongoDbAgents: dict = persistanceModule.loadMongoDbAgents(environment, agents)
    results = {}
    prepareAgentsForNewTrainning(agents, totalTrainningIterations, trainningBatchSize, maxEpisodeHistoryLenght)
    for trainningIteration in range(totalTrainningIterations):
        for index in range(trainningBatchSize):
            episode: Episode = runNewEpisode(
                environment,
                agentKey,
                agents,
                maxEpisodeHistoryLenght,
                showStates=showBoardStatesOnTrainningBatch,
                verifyEachIteration=verifyEachIterationOnTrainningBatch
            )
            persistanceModule.saveEpisode(mongoDbAgents[agentKey], episode, muteLogs=True)
            if 0 == index % 10:
                log.debug(runTrainning, f'End of the {trainningIteration*trainningBatchSize+index} {environment.getKey()} episode. {episode}. {agents[agentKey].getInternalStateDescription()}')
        measurementData = newMeasurementData()
        prepareAgentsForMeasurement(agents)
        for index in range(measuringBatchSize):
            measurementEpisode: Episode = runNewEpisode(
                environment,
                agentKey,
                agents,
                maxEpisodeHistoryLenght,
                showStates=showBoardStatesOnMeasuringBatch,
                verifyEachIteration=verifyEachIterationOnMeasuringBatch
            )
            updateMeasurementData(measurementData, measurementEpisode)
        updateAgentsAfterMeasurements(agents)
        results[trainningIteration] = getTrainningBatchResult(measurementData)
    finishAgentsTrainning(agents)
    if runLastGame:
        lastEpisode: Episode = runNewEpisode(
            environment,
            agentKey,
            agents,
            maxEpisodeHistoryLenght,
            showStates=showBoardStatesOnLastGame,
            verifyEachIteration=False
        )
        log.debug(runTrainning, f'{agents[agentKey].getInternalStateDescription()}')
        log.debug(runTrainning, f'len(lastEpisode.history): {len(lastEpisode.history)}')
    persistanceModule.updateMongoDbAgents(mongoDbAgents)
    return results
