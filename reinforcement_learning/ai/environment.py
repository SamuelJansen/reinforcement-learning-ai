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


class Environment(Object):

    def __init__(self, state: State, id: Id = None):
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

    def reset(self):
        # print('environment reseting')
        self.setState(self.__originalInitialState__.getCopy())
        self._reset()
        # print('environment reseted')

    def getState(self):
        return self.state.getCopy()

    def setState(self, state: State):
        self.state = state.getCopy()

    def getStateId(self):
        return self.state.getId()

    def getPossibleActions(self):
        raise MethodNotImplementedException()

    def updateState(self, action: Action, agents: List, isEpisodeMaxHistoryLenght: bool = False):
        raise MethodNotImplementedException()

    def nextState(self):
        raise MethodNotImplementedException()

    def isFinalState(self, state: State = None, isEpisodeMaxHistoryLenght: bool = False):
        raise MethodNotImplementedException()

    def getReward(fromState: State, toState: State, agents: List, isFinalState: bool):
        raise MethodNotImplementedException()

    def printState(self, lastAction: Action, data: str = c.BLANK):
        raise MethodNotImplementedException()

    def _reset(self):
        raise MethodNotImplementedException()
