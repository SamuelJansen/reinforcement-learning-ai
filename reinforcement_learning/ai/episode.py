from python_helper import Constant as c
from python_helper import ReflectionHelper, log, ObjectHelper

from reinforcement_learning.framework import value as valueModule

from reinforcement_learning.framework.object import Object, Id
from reinforcement_learning.framework.value import List, Dictionary

from reinforcement_learning.ai.action import Action
from reinforcement_learning.ai.agent import Agent
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
        self.showStates: bool = showStates
        self.setHistory(history)
        if self.showStates:
            environment.printState(None)

    def setHistory(
        self,
        history: History
    ):
        self.history: History = history

    def getHistory(self):
        return self.history

    def getCopy(self):
        return Episode(
            history=self.getHistory(),
            id=self.getId()
        )

    def willBeMaxHistoryLenght(self):
        return False if ObjectHelper.isNone(self.maxHistoryLenght) else self.maxHistoryLenght < len(self.history) + 1

    def isMaxHistoryLenght(self):
        return False if ObjectHelper.isNone(self.maxHistoryLenght) else self.maxHistoryLenght <= len(self.history)

    def nextSetp(self, agent: Agent, environment: Environment, data: str = c.BLANK):
        # print('episode nextStep')
        # print(f'agent: {agent}')
        # print('-    nextStep: environment.getState()')
        fromState: State = environment.getState()
        # print(f'fromState: {fromState}')
        # print('-    nextStep: environment.getPossibleActions()')
        possibleActions: List = environment.getPossibleActions()
        # print(f'possibleActions: {possibleActions}')
        # print('-    nextStep: agent.getAction(fromState, possibleActions)')
        action: Action = agent.getAction(fromState, possibleActions)
        # print(f'action: {action}')
        # print('-    nextStep: environment.updateState(action)')
        toState, reward, isFinalState = environment.updateState(
            action,
            self.agents,
            self.willBeMaxHistoryLenght()
        )
        # print(f'toState: {toState}, reward: {reward}')
        # print('-    nextStep: Event()')
        event: Event = Event(
            agent=agent,
            fromState=fromState,
            action=action,
            toState=toState,
            reward=reward
        )
        # print(f'   fromState: {fromState}, \n   possibleActions: {possibleActions}, \n   action: {action}, \n   toState: {toState}, \n   reward: {reward}')
        self.history.append(event)
        # print(self.history)
        if self.showStates:
            environment.printState(action, data=data)
        # print('-    nextStep: self.getAgents()')
        for key, agent in self.getAgents().items():
            agent.update(
                self.getHistoryByAgent(agent),
                reward,
                isFinalState
            )
        # print(self.history)
        # print('-    nextStep: self.nextState()')
        if not isFinalState:
            environment.nextState()
        # print(f'   fromState: {fromState}, \n   possibleActions: {possibleActions}, \n   action: {action}, \n   toState: {toState}, \n   reward: {reward}')
        # print('episode nextStep finished')

    def printHistory(self):
        log.prettyPython(self.printHistory, 'History', self.history, logLevel=log.DEBUG)

    def getAgents(self) -> Dictionary:
        return self.agents

    def getHistoryByAgent(self, agent: Agent) -> History:
        # print('getting hitory by agent')
        agentHistory = History()
        for event in self.history:
            if event.getAgentId() == agent.getId():
                agentHistory.append(event)
        # print('hitory by agent get')
        return agentHistory

    def __str__(self):
        return f'{ReflectionHelper.getClassName(self)}(id: {self.getId()})'

    def __repr__(self):
        return self.__str__()
