from python_helper import Constant as c
from python_helper import ObjectHelper, StringHelper, RandomHelper, log, SettingHelper
from reinforcement_learning import value as valueModule
from reinforcement_learning import MonteCarloEpisodeAgent, RandomAgent, Agent, Action, Environment, Episode, History, State, Reward, List, Tuple, Set, Dictionary, Id

BLACK_PLAYER_KEY: str = "B"
WHITE_PLAYER_KEY: str = "W"


class StateImpl(State):
    ...


class RandomAgentImpl(RandomAgent):
    ...


class HumanAgentImpl(Agent):

    def getAction(self, state: State, possibleActions: List):
        ...

    def _update(self, history: History, isFinalState: bool = False):
        ...


class ActionImpl(Action):
    ...


class Piece:
    def __init__(self, key: str, originalPosition: tuple):
        self.key: str = key
        self.originalPosition: tuple = tuple([*originalPosition])
        self.removeFromBoard()

    def reset(self):
        self._setPieceInGame()
        self.updateCurrentPosition(self.originalPosition)

    def updateCurrentPosition(self, newCurrentPosition: tuple):
        self.currentPosition = tuple([*newCurrentPosition])
        self._setPieceInGame()

    def removeFromBoard(self):
        self.currentPosition = None
        self.inGame = False

    def isInGame(self):
        return self.inGame

    def _setPieceInGame(self):
        self._setInGame(True)

    def _setInGame(self, inGame: bool):
        self.inGame = inGame


class Player:
    def __init__(self, pieces: list, *args, **kwargs):
        self.pieces = pieces

    def getPieces(self):
        return self.pieces

    def getPiecesInGame(self):
        return [piece for piece in self.getPieces() if piece.isInGame()]


class MonteCarloEpisodeAgentChessPlayer(MonteCarloEpisodeAgent, Player):
    def __init__(self, pieces: list, *args, **kwargs):
        MonteCarloEpisodeAgent.__init__(self, *args, **kwargs)
        Player.__init__(self, pieces)


class ChessEnvironmentImpl(Environment):

    VERTICAL_BOARD_SEPARATOR: str = '|'
    HORIZONTAL_BOARD_SEPARATOR = c.DASH
    EMPTY_SQUARE_VALUE: str = c.SPACE

    def __init__(
        self,
        whitePlayer: Player,
        blackPlayer: Player,
        *args,
        **kwargs
    ):
        Environment.__init__(self, *args, **kwargs)
        self.whitePlayer = whitePlayer
        self.blackPlayer = blackPlayer
        self._reset()

    def getPossibleActions(self):
        ...

    def updateState(self, action: Action, agents: List, willBeFinalState: bool) -> tuple:
        return toState, reward, isFinalState

    def nextState(self):
        self._updatePlayerTurn()

    def getReward(self, fromState: State, toState: State, agents: List, isFinalState: bool) -> Reward:
        return Reward(0)

    def isFinalState(self, state: State = None, isFinalState: bool = None) -> bool:
        # print('is final state')
        return (ObjectHelper.isNotNone(isFinalState) and isFinalState) or (
            ObjectHelper.isNotNone(self._getWinner(self.state if ObjectHelper.isNone(state) else state))
        )

    def printState(self, lastAction: Action, data: str = c.BLANK):
        ...

    def _reset(self):
        self.nextPlayerTurn: dict = {
            self.whitePlayer.key: self.blackPlayer,
            self.blackPlayer.key: self.whitePlayer
        }
        for piece in [*self.whitePlayer.getPieces(), *self.blackPlayer.getPieces()]:
            piece.reset()
            self._setPiece(piece, piece.originalPosition)
        self._setFirstPlayerTurn()

    def _getWinner(self, state: State) :
        return None

    def _setPiece(self, piece: Piece, position: tuple):
        self.state[piece.currentPosition[0]][piece.currentPosition[1]] = None
        self.state[position[0]][position[1]] = piece

    def _setFirstPlayerTurn(self):
        self.playerTurn = self.nextPlayerTurn[self.blackPlayer.key]

    def _updatePlayerTurn(self):
        self.playerTurn = self.nextPlayerTurn[self.playerTurn.key]


whitePlayer = MonteCarloEpisodeAgentChessPlayer([], WHITE_PLAYER_KEY)
blackPlayer = MonteCarloEpisodeAgentChessPlayer([], BLACK_PLAYER_KEY)
environment = ChessEnvironmentImpl(whitePlayer, blackPlayer, List([]))

print(whitePlayer)
print(blackPlayer)
print(environment)
