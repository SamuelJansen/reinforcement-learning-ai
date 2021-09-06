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


class Episode(Object):

    def __init__(
        self,
        environment: Environment,
        agents: Dictionary,
        history: History = None,
        maxHistoryLenght: int = None,
        id: Id = None,
        showStates: bool = True
    ):
        Object.__init__(self, id=id)
        self.environment: Environment = environment
        self.agents: Dictionary = agents
        self.maxHistoryLenght: int = maxHistoryLenght
        # print(f"Agents: {agents}")
        self.showStates: bool = bool(showStates)
        self.setHistory(history)
        if self.showStates:
            self.environment.printState(None)

    def setHistory(self, history: History):
        self.history: History = History(history)

    def getHistory(self):
        return self.history

    def getCopy(self):
        return Episode(
            self.environment,
            self.agents,
            history=self.getHistory(),
            maxHistoryLenght=self.maxHistoryLenght,
            id=self.getId(),
            showStates=self.showStates
        )

    def willBeMaxHistoryLenght(self):
        return False if ObjectHelper.isNone(self.maxHistoryLenght) else self.maxHistoryLenght <= len(self.history) + 1

    def isMaxHistoryLenght(self):
        return False if ObjectHelper.isNone(self.maxHistoryLenght) else self.maxHistoryLenght <= len(self.history)

    def nextSetp(self, agent: Agent, data: str = c.BLANK):
        # print('episode nextStep')
        # print(f'agent: {agent}')
        # print('-    nextStep: self.environment.getState()')
        fromState: State = self.environment.getState()
        # print(f'fromState: {fromState}')
        # print('-    nextStep: self.environment.getPossibleActions()')
        possibleActions: List = self.environment.getPossibleActions()
        # print(f'possibleActions: {possibleActions}')
        # print('-    nextStep: agent.getAction(fromState, possibleActions)')
        agentKnowlege_temp, action_temp = agent.getAction(fromState, possibleActions, showActionChoice=bool(self.showStates))
        agentKnowlege: List = agentKnowlege_temp
        action: Action = action_temp
        # print(f'action: {action}')
        # print('-    nextStep: self.environment.updateState(action)')
        toState, reward, isFinalState = self.environment.updateState(
            action,
            self.agents,
            self.willBeMaxHistoryLenght()
        )
        # print(f'toState: {toState}, reward: {reward}')
        # print('-    nextStep: Event()')
        event: Event = Event(
            agent,
            fromState,
            possibleActions,
            agentKnowlege,
            action,
            toState,
            reward
        )
        # print(f'   fromState: {fromState}, \n   possibleActions: {possibleActions}, \n   action: {action}, \n   toState: {toState}, \n   reward: {reward}')
        self.history.append(event)
        # print(self.history)
        if self.showStates:
            if StringHelper.isBlank(data) :
                visitAccess = event.agent.access(event.fromState.getHash())
                visits = [] if ObjectHelper.isEmpty(visitAccess) else visitAccess[AgentConstants.ACTIONS]
                agentParametersData = f'Exploration: {event.agent.exploration}, retention: {event.agent.retention}'
                fromStateData = f'{c.NEW_LINE}- From state: {event.fromState}, hash: {event.fromState.getHash()}'
                possibleActionData = f'''{c.NEW_LINE}- Possible actions: {StringHelper.join(
                    [
                        str(action) + ", " + action.getHash() for action in possibleActions
                    ],
                    character=" -- "
                )}'''
                agentKnowlegeData = f'{c.NEW_LINE}- Agent action knowlege: {agentKnowlege}'
                actionListData = f'''{c.NEW_LINE}- Agent action list data: {StringHelper.join(
                    [
                        str(v[AgentConstants.ACTION]) + c.COMA_SPACE + v[AgentConstants.ACTION].getHash() for v in visits if v[AgentConstants.ACTION].getHash() in [
                            a.getHash() for a in possibleActions
                        ]
                    ],
                    character=" -- "
                )}'''
                actionData = f'{c.NEW_LINE}- Action: {action}, hash: {action.getHash()}'
                toStateData = f'{c.NEW_LINE}- To state: {event.toState}, hash: {event.toState.getHash()}'
                rewardData = f'{c.NEW_LINE}- Reward: {reward}'
                data = f'{agentParametersData}{fromStateData}{possibleActionData}{agentKnowlegeData}{actionListData}{actionData}{toStateData}{rewardData}'
            self.environment.printState(data=data)
        # print('-    nextStep: self.getAgents()')
        for key, agent in self.getAgents().items():
            agent.update(
                self.history,
                isFinalState
            )
        # print(self.history)
        # print('-    nextStep: self.nextState()')
        if not isFinalState:
            self.environment.prepareNextState()
        # print(f'   fromState: {fromState}, \n   possibleActions: {possibleActions}, \n   action: {action}, \n   toState: {toState}, \n   reward: {reward}')
        # print('episode nextStep finished')

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
