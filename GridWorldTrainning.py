from python_helper import Constant as c
from python_helper import ObjectHelper, StringHelper, RandomHelper, log, SettingHelper, ReflectionHelper
from reinforcement_learning import value as valueModule
from reinforcement_learning import MongoDB

from reinforcement_learning import MonteCarloEpisodeAgent, RandomAgent, Agent, Action, Environment, Episode, History, State, Reward, List, Tuple, Set, Dictionary, Id


ZERO_EXPLORATION_FOR_MEASUREMENT: float = 0.0
AGENT_SUFIX = 'GridWorld'


def getWinner(environment: Environment) -> str:
    return environment._getWinner(environment.getState())


def prepareAgentForMeasurement(agent: Agent) -> float:
    originalExploration: float = agent.exploration
    agent.exploration = ZERO_EXPLORATION_FOR_MEASUREMENT
    agent.byPassAgentUpdate()
    return originalExploration


def updateAgentExploration(agent: Agent, originalExploration: float, playerXExplorationReducingRatio: float):
    agent.exploration = originalExploration * playerXExplorationReducingRatio
    agent.activateAgentUpdate()


def runNewEpisode(
    environment: Environment,
    episodeMaxHistoryLenght: int,
    playerXKey: str,
    agents: Dictionary,
    showStates: bool,
    verifyEachIteration: bool = False
):
    # print(f'Start of episode {index}: {episode}')
    environment.reset()
    episode: Episode = Episode(
        environment,
        agents,
        history=List(),
        maxHistoryLenght=episodeMaxHistoryLenght,
        showStates=showStates
    )
    while not environment.isFinalState(isEpisodeMaxHistoryLenght=episode.isMaxHistoryLenght()):
        # print(f'    exploration: {agents[environment.playerTurnKey].exploration}, retention: {agents[environment.playerTurnKey].retention}')
        # print('before next step')
        episode.nextSetp(agents[environment.playerTurnKey], environment, data=f'Exploration: {agents[environment.playerTurnKey].exploration}, retention: {agents[environment.playerTurnKey].retention}')
        # print('after next step')
    if verifyEachIteration:
        agents[playerXKey].printActionTable()
        input("hit enter to continue")
    return episode


def runTest(
    environment: Environment,
    episodeMaxHistoryLenght: int,
    playerXKey: str,
    agents: dict,
    playerXExplorationReducingRatio: float,
    totalTrainningIterations: int,
    trainningBatch: int,
    showBoardStatesOnTrainningBatch: bool,
    measuringBatch: int,
    showBoardStatesOnMeasuringBatch: bool,
    runLastGame: bool,
    showBoardStatesOnLastGame: bool
):
    mongoDBXPlayer = MongoDB(ReflectionHelper.getClassName(agents[playerXKey]), 'XPlayer')
    mongoDBXPlayer.loadActionTable(agents[playerXKey])
    results = {}
    for iteration in range(totalTrainningIterations):
        for index in range(trainningBatch):
            episode: Episode = runNewEpisode(
                environment,
                episodeMaxHistoryLenght,
                playerXKey,
                agents,
                showBoardStatesOnTrainningBatch
            )
            if 0 == index % 10:
                log.debug(runTest, f'End of episode {iteration*trainningBatch+index}: {episode}. Exploration: {agents[playerXKey].exploration}, retention: {agents[playerXKey].retention}')
        winCount: int = 0
        loseCount: int = 0
        originalExploration = prepareAgentForMeasurement(agents[playerXKey])
        measurementEpisodeLenList: list = []
        for index in range(measuringBatch):
            # print(f'exploration: {agents[playerXKey].exploration}, retention: {agents[playerXKey].retention}')
            measurementEpisode: Episode = runNewEpisode(
                environment,
                episodeMaxHistoryLenght,
                playerXKey,
                agents,
                showBoardStatesOnMeasuringBatch, ###- if agents[playerXKey].exploration<0.1 else False,
                verifyEachIteration=False ###- True if agents[playerXKey].exploration<0.1 else False
            )
            winner: str = getWinner(environment)
            if playerXKey == winner:
                winCount += 1
            else:
                loseCount += 1
            measurementEpisodeLenList.append(len(measurementEpisode.history))
        updateAgentExploration(agents[playerXKey], originalExploration, playerXExplorationReducingRatio)
        results[iteration] = {
            'winCount': winCount
            , 'loseCount': loseCount
            , 'drawCount': 0
            , 'measurementEpisodeLenList': measurementEpisodeLenList
        }
    if runLastGame:
        agents[playerXKey].exploration = ZERO_EXPLORATION_FOR_MEASUREMENT
        lastEpisode: Episode = runNewEpisode(
            environment,
            episodeMaxHistoryLenght,
            playerXKey,
            agents,
            showBoardStatesOnLastGame
        )
        log.debug(runTest, f'Exploration: {agents[playerXKey].exploration}, retention: {agents[playerXKey].retention}')
        log.debug(runTest, f'len(lastEpisode.history): {len(lastEpisode.history)}')
    mongoDBXPlayer.persistActionTable(agents[playerXKey])
    return results
