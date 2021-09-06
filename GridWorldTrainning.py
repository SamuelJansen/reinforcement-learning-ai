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
    playerKey: str,
    agents: Dictionary,
    showStates: bool,
    verifyEachIteration: bool = False
):
    # print(f'Start of episode {index}: {episode}')
    environment.reset()
    episode: Episode = Episode(
        environment,
        agents,
        history=History(),
        maxHistoryLenght=episodeMaxHistoryLenght,
        showStates=showStates
    )
    while not environment.isFinalState(isEpisodeMaxHistoryLenght=episode.isMaxHistoryLenght()):
        # print(f'    exploration: {agents[environment.playerTurnKey].exploration}, retention: {agents[environment.playerTurnKey].retention}')
        # print('before next step')
        episode.nextSetp(agents[environment.playerTurnKey], data=f'Exploration: {agents[environment.playerTurnKey].exploration}, retention: {agents[environment.playerTurnKey].retention}')
        # print('after next step')
    if verifyEachIteration:
        agents[playerKey].printActionTable()
        input("hit enter to continue")
    return episode


def runTest(
    environment: Environment,
    episodeMaxHistoryLenght: int,
    playerKey: str,
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
    mongoDBAgent = MongoDB(
        agents[playerKey],
        environment
    )
    mongoDBAgent.loadActionTable(agents[playerKey])
    results = {}
    for iteration in range(totalTrainningIterations):
        for index in range(trainningBatch):
            episode: Episode = runNewEpisode(
                environment,
                episodeMaxHistoryLenght,
                playerKey,
                agents,
                showBoardStatesOnTrainningBatch
            )
            mongoDBAgent.persistEpisode(episode, muteLogs=True)
            if 0 == index % 10:
                log.debug(runTest, f'End of episode {iteration*trainningBatch+index}: {episode}. Exploration: {agents[playerKey].exploration}, retention: {agents[playerKey].retention}')
        winCount: int = 0
        loseCount: int = 0
        originalExploration = prepareAgentForMeasurement(agents[playerKey])
        measurementEpisodeLenList: list = []
        for index in range(measuringBatch):
            # print(f'exploration: {agents[playerKey].exploration}, retention: {agents[playerKey].retention}')
            measurementEpisode: Episode = runNewEpisode(
                environment,
                episodeMaxHistoryLenght,
                playerKey,
                agents,
                showBoardStatesOnMeasuringBatch, ###- if agents[playerKey].exploration<0.1 else False,
                verifyEachIteration=False ###- True if agents[playerKey].exploration<0.1 else False
            )
            winner: str = getWinner(environment)
            if playerKey == winner:
                winCount += 1
            else:
                loseCount += 1
            measurementEpisodeLenList.append(len(measurementEpisode.history))
        updateAgentExploration(agents[playerKey], originalExploration, playerXExplorationReducingRatio)
        results[iteration] = {
            'winCount': winCount
            , 'loseCount': loseCount
            , 'drawCount': 0
            , 'measurementEpisodeLenList': measurementEpisodeLenList
        }
    if runLastGame:
        originalExploration = prepareAgentForMeasurement(agents[playerKey])
        lastEpisode: Episode = runNewEpisode(
            environment,
            episodeMaxHistoryLenght,
            playerKey,
            agents,
            showBoardStatesOnLastGame
        )
        log.debug(runTest, f'Exploration: {agents[playerKey].exploration}, retention: {agents[playerKey].retention}')
        log.debug(runTest, f'len(lastEpisode.history): {len(lastEpisode.history)}')
        updateAgentExploration(agents[playerKey], originalExploration, playerXExplorationReducingRatio)
    mongoDBAgent.persistActionTable(agents[playerKey])
    return results
