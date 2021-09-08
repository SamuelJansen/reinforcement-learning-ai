from python_helper import Constant as c
from reinforcement_learning.framework.object import Object, Id
from reinforcement_learning.framework.value import (
    List
)
from reinforcement_learning.framework.exception import (
    MethodNotImplementedException
)

from reinforcement_learning.ai.state import State
from reinforcement_learning.ai.action import Action
from reinforcement_learning.ai.reward import Reward


class ShouldBeEpisode:
    ...


class Environment(Object):

    def __init__(self, state: State, key: str, id: Id = None):
        self.key: str = key
        self.__originalInitialState__: State = state.getCopy()
        self.setState(state)
        Object.__init__(self, id=id)

    def getCopy(self):
        raise Exception("please review this implementation")
        args = [*self.__originalArgs__]
        kwargs = {
            **self.__originalKwargs__,
            **{
                'id': self.getId(),
                'initialState': self.getState()
            }
        }
        return self.__class__(
            *args,
            **kwargs
        )

    def getKey(self) -> str:
        return str(self.key)

    def reset(self):
        # print('environment reseting')
        self.setState(self.__originalInitialState__.getCopy())
        self._reset()
        # print('environment reseted')

    def getCurrentAgentKey(self) -> str:
        raise MethodNotImplementedException()

    def getState(self):
        return self.state.getCopy()

    def setState(self, state: State):
        raise MethodNotImplementedException()

    def getStateId(self):
        return self.state.getId()

    def getPossibleActions(self):
        raise MethodNotImplementedException()

    def updateState(self, action: Action, agents: List, episode: ShouldBeEpisode):
        raise MethodNotImplementedException()

    def prepareNextState(self):
        raise MethodNotImplementedException()

    def isFinalState(self, state: State = None, episode: ShouldBeEpisode = None):
        raise MethodNotImplementedException()

    def getRewardWhileUpdating(self, fromState: State, toState: State, episode: ShouldBeEpisode) -> tuple:
        willBeEpisodeMaxHistoryLenghtWhileUpdating: bool = episode.willBeMaxHistoryLenght()
        isFinalState: bool = self.isFinalState(state=toState, episode=episode) or willBeEpisodeMaxHistoryLenghtWhileUpdating
        return self.getReward(
            fromState,
            toState,
            episode,
            isFinalState,
            willBeEpisodeMaxHistoryLenghtWhileUpdating=willBeEpisodeMaxHistoryLenghtWhileUpdating
        ), isFinalState

    def getReward(self,
        fromState: State,
        toState: State,
        episode: ShouldBeEpisode,
        isFinalState: bool,
        willBeEpisodeMaxHistoryLenghtWhileUpdating: bool = False
    ):
        raise MethodNotImplementedException()

    def printState(self, data: str = c.BLANK):
        raise MethodNotImplementedException()

    def _reset(self):
        raise MethodNotImplementedException()
