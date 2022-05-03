from python_helper import Constant as c
from python_helper import ReflectionHelper, log, ObjectHelper, StringHelper

from reinforcement_learning.framework import value as valueModule

from reinforcement_learning.framework.object import Object, Id
from reinforcement_learning.framework.value import List, Dictionary

from reinforcement_learning.ai.action import Action
from reinforcement_learning.ai.agent import Agent, AgentConstants
from reinforcement_learning.ai.environment import Environment
from reinforcement_learning.ai.event import Event
from reinforcement_learning.ai.history import History
from reinforcement_learning.ai.reward import Reward
from reinforcement_learning.ai.state import State


AI_CLASSES = [
    valueModule.getSimpleName(State),
    valueModule.getSimpleName(Agent),
    valueModule.getSimpleName(Environment),
    valueModule.getSimpleName(Reward),
    valueModule.getSimpleName(Action)
]


def getKey(instanceOrAiClass):
    if isinstance(instanceOrAiClass, str):
        return instanceOrAiClass
    if instanceOrAiClass in AI_CLASSES:
        return ReflectionHelper.getName(instanceOrAiClass)
    for aiClass in AI_CLASSES:
        if isinstance(instanceOrAiClass, aiClass):
            return ReflectionHelper.getName(aiClass)


INFINITE: int = int(-1)


class Episode(Object):

    def __init__(
        self,
        environment: Environment,
        agents: Dictionary,
        maxHistoryLenght: int = INFINITE,
        history: History = None,
        id: Id = None,
        showStates: bool = True
    ):
        Object.__init__(self, id=id)
        self.setHistory(history)
        self.environment: Environment = environment
        self.agents: Dictionary = agents
        self.maxHistoryLenght: int = maxHistoryLenght if ObjectHelper.isNotNone(maxHistoryLenght) else INFINITE
        # print(f"Agents: {agents}")
        self.updating = False
        self.setUpdating()

        self.showStates: bool = bool(showStates)
        if self.showStates:
            self.environment.printState(None)

    def setUpdating(self):
        self.updating = True

    def setNotUpdating(self):
        self.updating = False

    def isUpdating(self) -> bool:
        return bool(self.updating)

    def run(self, verifyEachIteration: bool = False, agentPerspectiveKey: str = None):
        while not self.environment.isFinalState(episode=self):
            self.nextSetp(self.agents[self.environment.getCurrentAgentKey()])
        if verifyEachIteration and ObjectHelper.isNotNone(agentPerspectiveKey):
            self.agents[agentPerspectiveKey].printActionTable()

    def setHistory(self, history: History):
        self.history: History = History(history)

    def getHistory(self):
        return self.history

    def getCopy(self):
        return Episode(
            self.environment,
            self.agents,
            maxHistoryLenght=self.maxHistoryLenght,
            history=self.getHistory(),
            id=self.getId(),
            showStates=self.showStates
        )

    def _willBeMaxHistoryLenght(self):
        return False if INFINITE == self.maxHistoryLenght else self.maxHistoryLenght <= len(self.history) + 1 if self.isUpdating() else self.isMaxHistoryLenght()

    def isMaxHistoryLenght(self):
        return False if INFINITE == self.maxHistoryLenght else self.maxHistoryLenght <= len(self.history) if not self.isUpdating() else self._willBeMaxHistoryLenght()

    def nextSetp(self, agent: Agent, data: str = c.BLANK):
        self.setUpdating()
        if not INFINITE == self.maxHistoryLenght and len(self.history) > self.maxHistoryLenght:
            raise Exception('Last episode event missed')

        fromState: State = self.environment.getState()
        possibleActions: List = self.environment.getPossibleActions()

        agentKnowlege_temp, action_temp = agent.getAction(fromState, possibleActions)
        agentKnowlege: List = agentKnowlege_temp
        action: Action = action_temp

        toState, reward, isFinalState = self.environment.updateState(
            action,
            self
        )
        # print(f'toState: {toState}, reward: {reward}, isFinalState: {isFinalState}')

        event: Event = Event(
            agent,
            fromState,
            possibleActions,
            agentKnowlege,
            action,
            toState,
            reward
        )
        self.history.append(event)

        if self.showStates:
            if StringHelper.isBlank(data) :
                visitAccess = event.agent.access(event.fromState.getHash())
                visits = [] if ObjectHelper.isEmpty(visitAccess) else visitAccess[AgentConstants.ACTIONS]
                agentParametersData = f'{self.environment.getKey()} {event.agent.getInternalStateDescription()}'
                fromStateData = f'- From state: {event.fromState}, hash: {event.fromState.getHash()}'
                possibleActionData = f'''- Possible actions: {StringHelper.join(
                    [
                        str(action) + ", " + action.getHash() for action in possibleActions
                    ],
                    character=" -- "
                )}'''
                agentKnowlegeData = f'- Agent action knowlege: {agentKnowlege}'
                actionListData = f'''- Agent action list data: {StringHelper.join(
                    [
                        str(v[AgentConstants.ACTION]) + c.COMA_SPACE + v[AgentConstants.ACTION].getHash() for v in visits if v[AgentConstants.ACTION].getHash() in [
                            a.getHash() for a in possibleActions
                        ]
                    ],
                    character=" -- "
                )}'''
                actionData = f'- Action: {action}, hash: {action.getHash()}'
                toStateData = f'- To state: {event.toState}, hash: {event.toState.getHash()}'
                rewardData = f'- Reward: {reward}'
                historyData = f'- History lenght: {len(self.history)}, max history lenght: {self.maxHistoryLenght}, isFinalState: {isFinalState}'
                data = f'{c.NEW_LINE}{agentParametersData}{c.NEW_LINE}{fromStateData}{c.NEW_LINE}{possibleActionData}{c.NEW_LINE}{agentKnowlegeData}{c.NEW_LINE}{actionListData}{c.NEW_LINE}{actionData}{c.NEW_LINE}{toStateData}{c.NEW_LINE}{rewardData}{c.NEW_LINE}{historyData}'
            self.environment.printState(data=data)

        for key, agent in self.getAgents().items():
            agent.update(
                self.history,
                isFinalState
            )

        if not isFinalState:
            self.environment.prepareNextState()

        self.setNotUpdating()

    def printHistory(self):
        log.prettyPython(self.printHistory, 'History', self.history, logLevel=log.DEBUG)

    def getAgents(self) -> Dictionary:
        return self.agents

    def getHistoryByAgent(self, agent: Agent) -> History:
        # print('getting hitory by agent')
        return History([event for event in self.history if event.getAgentId() == agent.getId()])

    def __str__(self):
        return f'{ReflectionHelper.getClassName(self)}(id: {self.getId()})'

    def __repr__(self):
        return self.__str__()
