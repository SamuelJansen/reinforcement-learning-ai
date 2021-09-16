from operator import itemgetter
from python_helper import Constant as c
from python_helper import ObjectHelper, RandomHelper, ReflectionHelper, log
from reinforcement_learning.framework.object import Object, Id, getId
from reinforcement_learning.framework import value as valueModule
from reinforcement_learning.framework.value import List, Dictionary
from reinforcement_learning.ai.history import History
from reinforcement_learning.ai.state import State
from reinforcement_learning.ai.action import Action
from reinforcement_learning.ai.reward import Reward
from reinforcement_learning.framework.exception import (
    MethodNotImplementedException
)


def getExplorationReducingRatio(exploration: float, target: float, totalTrainningIterations: float):
    return (target * exploration) ** (exploration / float(totalTrainningIterations))


class AgentConstants:

    ID = 'id'
    ACTIONS = 'actions'
    ACTION = 'action'
    ACTION_VALUE = 'value'
    ACTION_VISITS = 'visits'

    ID_DB_KEY = f'{c.UNDERSCORE}{ID}'
    STATE_HASH_DB_KEY = 'stateHash'
    ACTIONS_DB_KEY = ACTIONS
    ACTION_DB_KEY = ACTION
    ACTION_VALUE_DB_KEY = ACTION_VALUE
    ACTION_VISITS_DB_KEY = ACTION_VISITS


class Agent(Object):

    def __init__(
        self,
        key: str,
        actionTable: dict = None,
        id: Id = None
    ):
        """
        'actionTable': {
            'stateHash': {
                '_id': MongoDbId('something'),
                'actions': [
                    {
                        'action': 'some action',
                        'value': float(),
                        'visits': int()
                    }
                ]
            }
        }
        """
        Object.__init__(self, id=id)
        self.key: str = key
        self.setNotTrainning()
        self.setActionTable({} if ObjectHelper.isEmpty(actionTable) else actionTable)
        self.activateUpdate()

    def getCopy(self):
        raise Exception('Agents should not be copied')

    def getKey(self) -> str:
        return str(self.key)

    def setActionTable(self, actionTable: dict):
        self.actionTable = actionTable

    def getActionTable(self):
        return self.actionTable

    def getDefaultStateActions(self):
        raise MethodNotImplementedException()

    def getAction(self, state: State, possibleActions: List) -> tuple:
        raise MethodNotImplementedException()

    def update(self, history: History, isFinalState: bool):
        if self.doUpdate:
            self._update(history, isFinalState)

    def _update(self, history: History, isFinalState: bool):
        raise MethodNotImplementedException()

    def access(self, givenStateHash: State) -> dict:
        for stateHash, visit in self.actionTable.items():
            if givenStateHash == stateHash:
                return visit
        return Dictionary()

    def accessByAction(self, stateHash: str, action: Action) -> dict:
        for v in self.access(stateHash).get(AgentConstants.ACTIONS, List()):
            if action.getHash() == v[AgentConstants.ACTION].getHash():
                return v
        # for stateHash, visit in self.actionTable.items():
        #     if givenStateHash == stateHash:
        #         for v in visit[AgentConstants.ACTIONS]:
        #             # print('===========================================================================================')
        #             # print(action)
        #             # print(action.getHash())
        #             # print(v[AgentConstants.ACTION])
        #             # print(v[AgentConstants.ACTION].getHash())
        #             if action.getHash() == v[AgentConstants.ACTION].getHash():
        #                 return v
        return Dictionary()

    def byPassUpdate(self):
        self.doUpdate: bool = False

    def activateUpdate(self):
        self.doUpdate: bool = True

    def freezeInternalState(self):
        self.byPassUpdate()

    def updateInternalState(self):
        self.activateUpdate()

    def newTrainning(self, totalTrainningIterations: int, trainningBatchSize: int):
        self.setIsTrainning()
        self._newTrainning(totalTrainningIterations, trainningBatchSize)

    def finishTrainning(self):
        self._finishTrainning()
        self.setNotTrainning()

    def _newTrainning(self, totalTrainningIterations: int, trainningBatchSize: int):
        pass

    def _finishTrainning(self):
        pass

    def setNotTrainning(self):
        self.trainning = False

    def setIsTrainning(self):
        self.trainning = True

    def isTrainning(self) -> bool:
        return bool(self.trainning)

    def printActionTable(self):
        actionTable = self.getActionTable()
        for k, v in actionTable.items():
            actionTable[k] = valueModule.sortedBy(v[AgentConstants.ACTIONS], AgentConstants.ACTION_VALUE)
        log.debug(self.printActionTable, f'"{self.getKey()}" agent action table size: {len(self.actionTable)}')
        if input("Print action table (y/n)?") in ['y', 'Y', 'yes', 'Yes', 'YES']:
            log.prettyPython(self.printActionTable, f'"{self.getKey()}" agent action table - ex.: q(s,a)', actionTable, logLevel=log.DEBUG)

    def getInternalStateDescription(self):
        return f'{self.getKey()} agent. Internal state -> empty'

    def __str__(self):
        return f'{ReflectionHelper.getClassName(self)}(id: {self.getId()})'

    def __repr__(self):
        return self.__str__()
