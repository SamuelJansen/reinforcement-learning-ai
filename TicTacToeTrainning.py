from python_helper import Constant as c
from python_helper import ObjectHelper, StringHelper, RandomHelper, log, SettingHelper, ReflectionHelper
from reinforcement_learning import value as valueModule
from reinforcement_learning import MongoDB

from reinforcement_learning import MonteCarloEpisodeAgent, RandomAgent, Action, Agent, Environment, Episode, History, State, Reward, List, Tuple, Set, Dictionary, Id


ZERO_EXPLORATION_FOR_MEASUREMENT: float = 0.0


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
    playerXKey: str,
    playerOKey: str,
    agents: Dictionary,
    showStates: bool,
    verifyEachIteration: bool = False
):
    environment.reset()
    episode: Episode = Episode(
        environment,
        agents,
        history=List(),
        showStates=showStates
    )
    # print(f'Start of episode {index}: {episode}')
    while not environment.isFinalState(isEpisodeMaxHistoryLenght=episode.isMaxHistoryLenght()):
        # print(f'    exploration: {agents[environment.playerTurnKey].exploration}, retention: {agents[environment.playerTurnKey].retention}')
        # print('before next step')
        episode.nextSetp(agents[environment.playerTurnKey], environment, data=f'Exploration: {agents[environment.playerTurnKey].exploration}, retention: {agents[environment.playerTurnKey].retention}')
        # print('after next step')
    if verifyEachIteration:
        agents[playerOKey].printActionTable()
        input("hit enter to continue")
    # print(f'---> winner: {getWinner(environment)}, state: {environment.state}')
    return episode

def runTest(
    environment: Environment,
    playerXKey: str,
    playerOKey: str,
    agents: dict,
    playerXExplorationReducingRatio: float,
    playerOExplorationReducingRatio: float,
    totalTrainningIterations: int,
    trainningBatch: int,
    verifyEachIterationOnTrainningBatch: bool,
    showBoardStatesOnTrainningBatch: bool,
    measuringBatch: int,
    verifyEachIterationOnMeasuringBatch: bool,
    showBoardStatesOnMeasuringBatch: bool,
    runLastGame: bool,
    showBoardStatesOnLastGame: bool,
    playerXAgentSufix: str = None,
    playerOAgentSufix: str = None
):
    mongoDBXPlayer = MongoDB(
        ReflectionHelper.getClassName(agents[playerXKey]),
        playerXAgentSufix if ObjectHelper.isNotNone(playerXAgentSufix) and StringHelper.isNotBlank(playerXAgentSufix) else ReflectionHelper.getClassName(environment)
    )
    mongoDBXPlayer.loadActionTable(agents[playerXKey])
    mongoDBOPlayer = MongoDB(
        ReflectionHelper.getClassName(agents[playerOKey]),
        playerOAgentSufix if ObjectHelper.isNotNone(playerOAgentSufix) and StringHelper.isNotBlank(playerOAgentSufix) else ReflectionHelper.getClassName(environment)
    )
    mongoDBOPlayer.loadActionTable(agents[playerOKey])
    results = {}
    for iteration in range(totalTrainningIterations):
        for index in range(trainningBatch):
            episode: Episode = runNewEpisode(
                environment,
                playerXKey,
                playerOKey,
                agents,
                showBoardStatesOnTrainningBatch,
                verifyEachIteration=verifyEachIterationOnTrainningBatch
            )
            if 0 == index % 10:
                log.debug(runTest, f'End of episode {iteration*trainningBatch+index}: {episode}. Exploration: {agents[playerOKey].exploration}, retention: {agents[playerOKey].retention}')
        winCount: int = 0
        loseCount: int = 0
        originalExplorationPlayerO = prepareAgentForMeasurement(agents[playerOKey])
        originalExplorationPlayerX = prepareAgentForMeasurement(agents[playerXKey])
        measurementEpisodeLenList: list = []
        for index in range(measuringBatch):
            measurementEpisode: Episode = runNewEpisode(
                environment,
                playerXKey,
                playerOKey,
                agents,
                showBoardStatesOnMeasuringBatch,
                verifyEachIteration=verifyEachIterationOnMeasuringBatch
            )
            winner: str = getWinner(environment)
            if playerOKey == winner:
                winCount += 1
            elif playerXKey == winner:
                loseCount += 1
            # print(f'    ---> episode winner: {winner}, state: {environment.state}')
            measurementEpisodeLenList.append(len(measurementEpisode.history))
        updateAgentExploration(agents[playerOKey], originalExplorationPlayerO, playerXExplorationReducingRatio)
        updateAgentExploration(agents[playerXKey], originalExplorationPlayerX, playerOExplorationReducingRatio)
        results[iteration] = {
            'winCount': winCount
            , 'loseCount': loseCount
            , 'drawCount': measuringBatch - winCount - loseCount
            , 'measurementEpisodeLenList': measurementEpisodeLenList
        }
    if runLastGame:
        agents[playerOKey].exploration = ZERO_EXPLORATION_FOR_MEASUREMENT
        agents[playerXKey].exploration = ZERO_EXPLORATION_FOR_MEASUREMENT
        lastEpisode: Episode = runNewEpisode(
            environment,
            playerXKey,
            playerOKey,
            agents,
            showBoardStatesOnLastGame,
            verifyEachIteration=False
        )
        log.debug(runTest, f'X player: Exploration: {agents[playerXKey].exploration}, retention: {agents[playerXKey].retention}')
        log.debug(runTest, f'O player: Exploration: {agents[playerOKey].exploration}, retention: {agents[playerOKey].retention}')
        log.debug(runTest, f'len(lastEpisode.history): {len(lastEpisode.history)}')
    mongoDBXPlayer.persistActionTable(agents[playerXKey])
    mongoDBOPlayer.persistActionTable(agents[playerOKey])
    return results
