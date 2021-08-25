from python_helper import Constant as c
from python_helper import ObjectHelper, StringHelper, RandomHelper, log, SettingHelper
from reinforcement_learning import value as valueModule
from reinforcement_learning import MonteCarloEpisodeAgent, RandomAgent, Agent, Action, Environment, Episode, History, State, Reward, List, Tuple, Set, Dictionary, Id


SettingHelper.updateActiveEnvironment(SettingHelper.LOCAL_ENVIRONMENT)
log.loadSettings()


EMPTY_STATE_VALUE: str = c.SPACE

PLAYER_X_VALUE: str = 'X'
REWARD_SIMBOL = 'R'

DEFAULT_BOARD_SIZE: list = [4, 4]


class HumanAgentImpl(Agent):

    def getAction(self, state: State, possibleActions: List):
        while True:
            choice = input(f'asdw ({possibleActions}): ')
            if 'a' == choice:
                return List([(-1, 0, self.key)])
            if 's' == choice:
                return List([(0, -1, self.key)])
            if 'd' == choice:
                return List([(1, 0, self.key)])
            if 'w' == choice:
                return List([(0, 1, self.key)])

    def _update(self, history: History, isFinalState: bool = False):
        ...


class SquareGridEnvironmentImpl(Environment):

    VERTICAL_BOARD_SEPARATOR = f'|'
    HORIZONTAL_BOARD_SEPARATOR = c.DASH
    EMPTY_STATE_VALUE: str = EMPTY_STATE_VALUE
    REWARD_SIMBOL = REWARD_SIMBOL

    def __init__(
        self,
        winReward: float,
        drawReward: float,
        defaultReward: float,
        *args,
        playerTurnKey: str = PLAYER_X_VALUE,
        boardSize: list = DEFAULT_BOARD_SIZE,
        targetPositions: List = List(),
        valueSpacement: int = 3,
        margin: int = 3,
        initialState: State = None,
        **kwargs
    ):
        self.__originalArgs__ = [
            winReward,
            drawReward,
            defaultReward
        ]
        self.__originalKwargs__ = {
            'playerTurnKey': playerTurnKey,
            'boardSize': boardSize,
            'valueSpacement': valueSpacement,
            'margin': margin,
            'targetPositions': targetPositions,
            'initialState': initialState
        }
        self.winReward: float = winReward
        self.drawReward: float = drawReward
        self.defaultReward: float = defaultReward
        self.boardSize = boardSize
        self.targetPositions = targetPositions
        self.valueSpacement = valueSpacement
        self.margin = margin
        self.playerTurnKey = playerTurnKey
        self.nextPlayerTurn = {
            self.playerTurnKey: self.playerTurnKey
        }
        self.originalPlayerTurnKey = str(self.playerTurnKey)
        initialState: State = self.getInitialState(initialState)
        Environment.__init__(self, initialState, *args, **kwargs)

    def getInitialState(self, initialState):
        if ObjectHelper.isNotNone(initialState):
            return initialState
        initialState: State = State(
            List([
                List([
                    self.EMPTY_STATE_VALUE for _ in range(self.boardSize[0])
                ]) for _ in range(self.boardSize[1])
            ])
        )
        initialState[0][0] = PLAYER_X_VALUE
        for r in self.targetPositions:
            # print(r)
            # print(r[0])
            # print(r[1])
            initialState[r[1]][r[0]] = self.REWARD_SIMBOL
        return initialState

    def getPossibleActions(self):
        # print('getting possible actions')
        currentPosition = self._getCurrentPosition(self.state)
        # print(currentPosition, self.state)
        possibleActions = List()
        for h in [-1, 0, 1]:
            for v in [-1, 0, 1]:
                if 1 == [h, v].count(0): ###- dimentions - 1
                    if 0 <= currentPosition[0] + h < len(self.state) and 0 <= currentPosition[1] + v < len(self.state[0]) :
                        if self.state[currentPosition[0] + h][currentPosition[1] + v] in [self.EMPTY_STATE_VALUE, self.REWARD_SIMBOL]:
                            possibleActions.append(Action([(h, v, self.playerTurnKey)]))
        # print(f'possible {possibleActions}')
        return possibleActions

    def _getCurrentPosition(self, state: State):
        for v in range(len(self.state)):
            for h in range(len(self.state[0])):
                if not state[v][h] == self.EMPTY_STATE_VALUE:
                    return [v, h]

    def updateState(self, action: Action, agents: List, willBeEpisodeMaxHistoryLenght: bool) -> tuple:
        # print('updating state')
        fromState = self.getState()
        currentPosition = self._getCurrentPosition(fromState)
        # print(fromState, currentPosition, action])
        self._validateGameNotFinished(fromState)
        if ObjectHelper.isNotNone(action):
            toState: State = State(fromState).getCopy()
            for actionValue in action:
                if toState[currentPosition[0] + actionValue[0]][currentPosition[1] + actionValue[1]] not in [self.REWARD_SIMBOL, self.EMPTY_STATE_VALUE]:
                    raise Exception(f'Invalid action: {actionValue} from state {fromState} to state {toState}')
                toState[currentPosition[0]][currentPosition[1]] = self.EMPTY_STATE_VALUE
                toState[currentPosition[0] + actionValue[0]][currentPosition[1] + actionValue[1]] = actionValue[2]
        toState.updateHash()
        # print(fromState, action, toState)
        self.setState(toState)
        isFinalState: bool = self.isFinalState(state=toState, isEpisodeMaxHistoryLenght=willBeEpisodeMaxHistoryLenght)
        reward: Reward = self.getReward(fromState, toState, agents, isFinalState)
        # print(f'isFinalState: {isActualyFinalState}')
        # print('updating state finished')
        return toState, reward, isFinalState

    def nextState(self):
        self.playerTurnKey = self.nextPlayerTurn.get(self.playerTurnKey)

    def getReward(self, fromState: State, toState: State, agents: List, isFinalState: bool) -> Reward:
        # print('getting reward')
        # print(f"Agents: {agents}")
        self._validateGameNotFinished(fromState)
        reward = Reward({key: self.winReward if isFinalState and key is self.playerTurnKey else self.defaultReward for key, agent in agents.items()})
        # print('getting reward finished')
        return reward

    def isFinalState(self, state: State = None, isEpisodeMaxHistoryLenght: bool = None) -> bool:
        # print('is final state')
        return (
            ObjectHelper.isNotNone(isEpisodeMaxHistoryLenght) and isEpisodeMaxHistoryLenght
        ) or (
            ObjectHelper.isNotNone(self._getWinner(self.state if ObjectHelper.isNone(state) else state))
        )

    def printState(self, lastAction: Action, data: str = c.BLANK):
        state: State = self.getState()
        # print(state)
        horizontalSeparator = StringHelper.join([self.HORIZONTAL_BOARD_SEPARATOR * self.valueSpacement for _ in range(len(self.state[0]))], character=f'{self.HORIZONTAL_BOARD_SEPARATOR * len(self.VERTICAL_BOARD_SEPARATOR)}')
        print(f'{c.NEW_LINE}State: {state.getId()}')
        print(f'- Player turn: {self.playerTurnKey}{f", {data}" if not c.BLANK == data else c.BLANK}')
        print(f'- Action: {lastAction}{c.NEW_LINE}')
        print(StringHelper.join(
            [
                c.SPACE * (self.valueSpacement + self.margin),
                StringHelper.join([str(index).center(self.valueSpacement) for index in range(len(self.state[0]))], character=c.SPACE*len(self.VERTICAL_BOARD_SEPARATOR)),
                c.NEW_LINE
            ]
        ))
        print(StringHelper.join(
            [
                StringHelper.join([
                    f'{str(valueModule.indexOf(row, state)).center(self.valueSpacement + self.margin)}',
                    StringHelper.join([f'{str(value).center(self.valueSpacement)}' for value in row], character=self.VERTICAL_BOARD_SEPARATOR),
                    c.NEW_LINE
                ]) for row in state
            ],
            character=f'{c.BLANK.center(self.valueSpacement + self.margin)}{horizontalSeparator}{c.NEW_LINE}'
        ))
        # print('end of print state')

    def _getWinner(self, state: State):
        currentPosition = self._getCurrentPosition(state)
        if currentPosition in self.targetPositions:
            if state[currentPosition[0]][currentPosition[1]] not in [self.REWARD_SIMBOL, self.playerTurnKey]:
                raise Exception('Error while evaluating winner')
            return state[currentPosition[0]][currentPosition[1]]

    def _validateGameNotFinished(self, fromState: State):
        if self.isFinalState(state=fromState):
            raise Exception(f'Episode should be finished: {fromState}')

    def _reset(self):
        self.playerTurnKey = str(self.originalPlayerTurnKey)
