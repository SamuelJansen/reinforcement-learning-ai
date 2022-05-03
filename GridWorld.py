import msvcrt

from python_helper import Constant as c
from python_helper import ObjectHelper, StringHelper, RandomHelper, log, SettingHelper, EnvironmentHelper

from reinforcement_learning import value as valueModule
from reinforcement_learning import MonteCarloEpisodeAgent, RandomAgent, Agent, Action, Environment, Episode, History, State, Reward, List, Tuple, Set, Dictionary, Id
from reinforcement_learning import environmentModule

SettingHelper.updateActiveEnvironment(SettingHelper.LOCAL_ENVIRONMENT)
log.loadSettings()


EMPTY_STATE_VALUE: str = c.SPACE

DEFAULT_PLAYER_KEY: str = 'X'
REWARD_SIMBOL = 'R'

DEFAULT_BOARD_SIZE: list = [4, 4]


class HumanAgentImpl(Agent):

    INPUTTER = None

    def getAction(self, state: State, possibleActions: List):
        self.loadInputterIfNeeded()
        print(f'asdw ({possibleActions}): ', end=c.BLANK)
        while True:
            choice = self.inputter.getch()
            # choice = inputter.getche(f'asdw ({possibleActions}): ')
            if b'a' == choice:
                action = Action([(0, -1, self.key)])
            elif b'd' == choice:
                action = Action([(0, 1, self.key)])
            elif b's' == choice:
                action = Action([(1, 0, self.key)])
            elif b'w' == choice:
                action = Action([(-1, 0, self.key)])
            if action.getHash() in [a.getHash() for a in possibleActions]:
                print()
                return List([]), action
            action = List([])

    def _update(self, history: History, episode: environmentModule.ShouldBeEpisode):
        ...

    def loadInputterIfNeeded(self):
        if ObjectHelper.isNone(self.INPUTTER):
            if EnvironmentHelper.isLinux():
                import msvcrt as inputter
            else :
                import getch as inputter
            self.inputter = inputter


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
        playerTurnKey: str = DEFAULT_PLAYER_KEY,
        boardSize: list = DEFAULT_BOARD_SIZE,
        targetPositions: List = List(),
        valueSpacement: int = 3,
        margin: int = 3,
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
            'targetPositions': targetPositions
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
            self.getCurrentAgentKey(): self.getCurrentAgentKey()
        }
        self.originalPlayerTurnKey = self.getCurrentAgentKey()
        Environment.__init__(self, *args, **kwargs)

    def getCurrentAgentKey(self) -> str:
        return str(self.playerTurnKey)

    def setState(self, state: State):
        self.state = state.getCopy()

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
        initialState[0][0] = DEFAULT_PLAYER_KEY
        for r in self.targetPositions:
            # print(r)
            # print(r[0])
            # print(r[1])
            initialState[r[0]][r[1]] = self.REWARD_SIMBOL
        return initialState

    def getPossibleActions(self):
        # print('getting possible actions')
        currentPosition = self._getCurrentPosition(self.state, self.getCurrentAgentKey())
        # print(currentPosition, self.state)
        possibleActions = List()
        for h in [-1, 0, 1]:
            for v in [-1, 0, 1]:
                if 1 == [h, v].count(0): ###- dimentions - 1
                    if 0 <= currentPosition[0] + h < len(self.state) and 0 <= currentPosition[1] + v < len(self.state[0]) :
                        if self.state[currentPosition[0] + h][currentPosition[1] + v] in [self.EMPTY_STATE_VALUE, self.REWARD_SIMBOL]:
                            possibleActions.append(Action([(h, v, self.getCurrentAgentKey())]))
        # print(f'fromState: {self.state}')
        # print(f'currentPosition: {currentPosition}')
        # print(f'possibleActions [{StringHelper.join([a.getId() + " - " + a.__ai_hash__ + " - " + str(a) for a in possibleActions], character=c.COMA + c.SPACE)}]')
        return possibleActions

    def _getCurrentPosition(self, state: State, agentKey: str):
        for v in range(len(state)):
            for h in range(len(state[0])):
                if state[v][h] == agentKey and not state[v][h] == self.EMPTY_STATE_VALUE:
                    return [v, h]

    def updateState(self, action: Action, episode: environmentModule.ShouldBeEpisode) -> tuple:
        fromState = self.getState()
        currentPosition = self._getCurrentPosition(fromState, self.getCurrentAgentKey())
        self._validateGameNotFinished(fromState)

        if ObjectHelper.isNotNone(action):
            toState: State = State(fromState).getCopy()
            for actionValue in action:
                if toState[currentPosition[0] + actionValue[0]][currentPosition[1] + actionValue[1]] not in [self.REWARD_SIMBOL, self.EMPTY_STATE_VALUE]:
                    raise Exception(f'Invalid action: {actionValue} from state {fromState} to state {toState}')
                toState[currentPosition[0]][currentPosition[1]] = self.EMPTY_STATE_VALUE
                toState[currentPosition[0] + actionValue[0]][currentPosition[1] + actionValue[1]] = actionValue[2]

        toState.updateHash()
        self.setState(toState)

        isFinalState: bool = self.isFinalState(state=toState, episode=episode)
        reward: Reward = self.getReward(fromState, toState, episode, isFinalState)
        return toState, reward, isFinalState

    def prepareNextState(self):
        self.playerTurnKey = self.nextPlayerTurn.get(self.getCurrentAgentKey())

    def getReward(self, fromState: State, toState: State, episode: environmentModule.ShouldBeEpisode, isFinalState: bool) -> Reward:
        self._validateGameNotFinished(fromState)
        reward = Reward({key: self.winReward if isFinalState and key is self.getCurrentAgentKey() else self.defaultReward for key, agent in episode.agents.items()})
        return reward

    def _isInFinalStateCondition(self, state: State = None) -> bool:
        return ObjectHelper.isNotNone(self._getWinner(state, self.getCurrentAgentKey()))

    def printState(self, data: str = c.BLANK):
        horizontalSeparator = StringHelper.join([self.HORIZONTAL_BOARD_SEPARATOR * self.valueSpacement for _ in range(len(self.state[0]))], character=f'{self.HORIZONTAL_BOARD_SEPARATOR * len(self.VERTICAL_BOARD_SEPARATOR)}')
        if StringHelper.isNotBlank(data) :
            print(f'{data}')
        print(f'- Player turn: {self.getCurrentAgentKey()}')
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
                    f'{str(valueModule.indexOf(row, self.state)).center(self.valueSpacement + self.margin)}',
                    StringHelper.join([f'{str(value).center(self.valueSpacement)}' for value in row], character=self.VERTICAL_BOARD_SEPARATOR),
                    c.NEW_LINE
                ]) for row in self.state
            ],
            character=f'{c.BLANK.center(self.valueSpacement + self.margin)}{horizontalSeparator}{c.NEW_LINE}'
        ))
        # print('end of print state')

    def _getWinner(self, state: State, agentKey: str):
        currentPosition = self._getCurrentPosition(state, agentKey)
        if currentPosition in self.targetPositions:
            if state[currentPosition[0]][currentPosition[1]] not in [self.REWARD_SIMBOL, self.getCurrentAgentKey()]:
                raise Exception('Error while evaluating winner')
            return state[currentPosition[0]][currentPosition[1]]

    def _validateGameNotFinished(self, fromState: State):
        if self.isFinalState(state=fromState):
            raise Exception(f'Episode should be finished: {fromState}')

    def _reset(self):
        self.playerTurnKey = str(self.originalPlayerTurnKey)
