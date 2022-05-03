from python_helper import Constant as c
from python_helper import  ObjectHelper

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

    def __init__(self, key: str, state: State = None, id: Id = None, __originalInitialState__: State = None):
        self.key: str = key
        self.setState(self.getInitialState(state))
        self.__originalInitialState__: State = __originalInitialState__ if ObjectHelper.isNotNone(__originalInitialState__) else self.state if ObjectHelper.isNone(state) else state.getCopy()
        Object.__init__(self, id=id)

    def getCopy(self):
        raise Exception("please review this implementation")
        args = [
            *self.__originalArgs__,
            self.getState(),
            self.getKey()
        ]
        kwargs = {
            **self.__originalKwargs__,
            **{
                'id': self.getId()
            }
        }
        return self.__class__(
            self.getState(),
            self.getKey(),
            id=self.getId(),
            __originalInitialState__ = self.__originalInitialState__.getCopy()
        )

    def getKey(self) -> str:
        return str(self.key)

    def getInitialState(self) -> State:
        raise MethodNotImplementedException()

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

    def isFinalState(self, state: State = None, episode: ShouldBeEpisode = None) -> bool:
        return (
            False if ObjectHelper.isNone(episode) else episode.isMaxHistoryLenght()
        ) or (
            self._isInFinalStateCondition(state if ObjectHelper.isNotNone(state) else self.state)
        )

    def _isInFinalStateCondition(self, state: State = None, episode: ShouldBeEpisode = None) -> bool:
        raise MethodNotImplementedException()

    def getReward(self,
        fromState: State,
        toState: State,
        episode: ShouldBeEpisode,
        isFinalState: bool
    ):
        raise MethodNotImplementedException()

    def printState(self, data: str = c.BLANK):
        raise MethodNotImplementedException()

    def _reset(self):
        raise MethodNotImplementedException()
