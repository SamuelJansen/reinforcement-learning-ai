from python_helper import Constant as c
from python_helper import ObjectHelper, StringHelper
from reinforcement_learning import value as valueModule
from reinforcement_learning import MonteCarloEpisodeAgent, RandomAgent, Action, Agent, Environment, Episode, History, State, Reward, List, Tuple, Set, Dictionary, Id
from reinforcement_learning import environmentModule

DEFAULT_BOARD_SIZE: int = 3
EMPTY_STATE_VALUE: str = c.SPACE


class TicTacToeEnvironmentImpl(Environment):

    VERTICAL_BOARD_SEPARATOR = f'{c.SPACE}|{c.SPACE}'
    HORIZONTAL_BOARD_SEPARATOR = c.DASH
    EMPTY_STATE_VALUE: str = EMPTY_STATE_VALUE

    def __init__(
        self,
        playerX: Agent,
        playerY: Agent,
        winReward: float,
        drawReward: float,
        defaultReward: float,
        *args,
        boardSize: int = DEFAULT_BOARD_SIZE,
        valueSpacement: int = 3,
        margin: int = 3,
        initialState: List = None,
        **kwargs
    ):
        self.__originalArgs__ = [
            playerX,
            playerY,
            winReward,
            drawReward,
            defaultReward
        ]
        self.__originalKwargs__ = {
            'boardSize': boardSize,
            'valueSpacement': valueSpacement,
            'margin': margin,
            'initialState': initialState
        }
        self.winReward: float = winReward
        self.drawReward: float = drawReward
        self.defaultReward: float = defaultReward
        self.boardSize: int = boardSize
        self.valueSpacement: int = valueSpacement
        self.margin: int = margin

        self.playerX: Agent = playerX
        self.playerY: Agent = playerY
        self.nextPlayerTurn: dict = {
            self.playerX.key : self.playerY.key,
            self.playerY.key : self.playerX.key
        }
        initialState: State = self.getInitialState(initialState)
        Environment.__init__(self, initialState, *args, **kwargs)

    def setState(self, state: State):
        self.state = state.getCopy()

    def getCurrentAgentKey(self) -> str:
        return str(self.playerTurnKey)

    def getInitialState(self, initialState: List = None):
        self.playerTurnKey: str = self.playerX.key
        self.originalPlayerTurnKey: str = self.getCurrentAgentKey()
        return initialState if ObjectHelper.isNotNone(initialState) else State(
            List([
                List([
                    self.EMPTY_STATE_VALUE for _ in range(self.boardSize)
                ]) for _ in range(self.boardSize)
            ])
        )

    def getPossibleActions(self):
        possibleActions = List()
        for h in range(len(self.state)):
            for v in range(len(self.state)):
                if self.state[h][v] == self.EMPTY_STATE_VALUE:
                    possibleActions.append(Action([(h, v, self.getCurrentAgentKey())]))
        return possibleActions

    def updateState(self, action: Action, episode: environmentModule.ShouldBeEpisode) -> tuple:
        fromState = self.getState()
        self._validateGameNotFinished(fromState)
        
        if ObjectHelper.isNotNone(action):
            toState: State = State(fromState).getCopy()
            for actionValue in action:
                if not self.EMPTY_STATE_VALUE == toState[actionValue[0]][actionValue[1]]:
                    raise Exception(f'Invalid action: {actionValue} from state {fromState} to state {toState}')
                toState[actionValue[0]][actionValue[1]] = actionValue[2]

        toState.updateHash()
        self.setState(toState)

        temp_reward, temp_isFinalState = self.getRewardWhileUpdating(fromState, toState, episode)
        reward: Reward = temp_reward
        isFinalState: bool = temp_isFinalState

        return toState, reward, isFinalState

    def prepareNextState(self):
        self.playerTurnKey = self.nextPlayerTurn.get(self.getCurrentAgentKey())

    def getReward(self,
        fromState: State,
        toState: State,
        episode: environmentModule.ShouldBeEpisode,
        isFinalState: bool,
        willBeEpisodeMaxHistoryLenghtWhileUpdating: bool = False
    ) -> Reward:
        winner = self._getWinner(toState)
        self._validateGameNotFinished(fromState)
        if isFinalState:
            if winner is self.EMPTY_STATE_VALUE:
                return Reward({key: self.drawReward for key in episode.agents})
            return Reward({key: self.winReward if key is winner else self.defaultReward for key in episode.agents})
        # print(f'self.getReward: {self.EMPTY_STATE_VALUE} == {winner}: {self.EMPTY_STATE_VALUE == winner}')
        return Reward({key: self.defaultReward for key in episode.agents})

    def isFinalState(self, state: State = None, episode: environmentModule.ShouldBeEpisode = None) -> bool:
        return (
            False if ObjectHelper.isNone(episode) else episode.isMaxHistoryLenght()
        ) or (
            ObjectHelper.isNotNone(self._getWinner(state if ObjectHelper.isNotNone(state) else self.state))
        )

    def printState(self, data=c.BLANK):
        horizontalSeparator = StringHelper.join([self.HORIZONTAL_BOARD_SEPARATOR * self.valueSpacement for _ in range(len(self.state))], character=f'{self.HORIZONTAL_BOARD_SEPARATOR * len(self.VERTICAL_BOARD_SEPARATOR)}')
        if StringHelper.isNotBlank(data) :
            print(f'{data}')
        print(f'- Player turn: {self.getCurrentAgentKey()}')
        print(StringHelper.join(
            [
                c.SPACE * (self.valueSpacement + self.margin),
                StringHelper.join([str(index).center(self.valueSpacement) for index in range(len(self.state))], character=c.SPACE*len(self.VERTICAL_BOARD_SEPARATOR)),
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

    def _getWinner(self, state: State):
        winner = None
        for possiblePlay in self.nextPlayerTurn:

            for row in state:
                verifyRowSet = set(possiblePlay == value for value in row)
                if 1 == len(verifyRowSet) and True in verifyRowSet:
                    winner = possiblePlay

            for columnIndex in range(len(state)):
                isFinished: bool = True
                for row in state:
                    isFinished = isFinished and possiblePlay == row[columnIndex]
                    if not isFinished:
                        break
                if isFinished:
                    winner = possiblePlay

            isPositiveDiagonalyFinished: bool = True
            for index in range(len(state)):
                isPositiveDiagonalyFinished = possiblePlay == state[index][index]
                # print(f'{possiblePlay} == {state[index][index]}: {possiblePlay == state[index][index]}')
                if not isPositiveDiagonalyFinished:
                    # print('not isPositiveDiagonalyFinished')
                    break
            if isPositiveDiagonalyFinished:
                winner = possiblePlay

            isNegativeDiagonalyFinished: bool = True
            for index in range(len(state)):
                isNegativeDiagonalyFinished = possiblePlay == state[index][len(state) - index - 1]
                # print(f'{possiblePlay} == {state[index][len(self.state) - index - 1]}: {possiblePlay == state[index][len(self.state) - index - 1]}')
                if not isNegativeDiagonalyFinished:
                    # print('not isNegativeDiagonalyFinished')
                    break
            if isNegativeDiagonalyFinished:
                winner = possiblePlay

        if ObjectHelper.isNone(winner):
            actionsTaken: int = 0
            for row in state:
                for value in row:
                    if not self.EMPTY_STATE_VALUE == value:
                        actionsTaken += 1
            if len(state)**2 == actionsTaken:
                winner = self.EMPTY_STATE_VALUE
        # print(f'winner: {winner}, state: {state}')
        return winner

    def _validateGameNotFinished(self, fromState: State):
        if self.isFinalState(state=fromState):
            raise Exception(f'Episode should be finished: {fromState}')

    def _reset(self):
        self.playerTurnKey = str(self.originalPlayerTurnKey)
