from python_helper import Constant as c
from python_helper import ReflectionHelper, ObjectHelper

from reinforcement_learning.framework.object import Object, Id
from reinforcement_learning.framework.value import List

from reinforcement_learning.ai.action import Action
from reinforcement_learning.ai.agent import Agent
from reinforcement_learning.ai.history import History
from reinforcement_learning.ai.reward import Reward
from reinforcement_learning.ai.state import State


class Event(Object):

    def __init__(
        self,
        agent: Agent,
        fromState: State,
        possibleActions: List,
        agentKnowlege: List,
        action: Action,
        toState: State,
        reward: Reward,
        id: Id = None
    ):
        self.setAgent(agent)
        self.setFromState(fromState)
        self.setPossibleActions(possibleActions)
        self.setAgentKnowlege(agentKnowlege)
        self.setAction(action)
        self.setToState(toState)
        self.setReward(reward)
        Object.__init__(self, id=id)

    def asJson(self):
        return {
            'agentKey': self.agent.getKey(),
            'fromState': self.fromState,
            'possibleActions': self.possibleActions,
            'agentKnowlege': self.agentKnowlege,
            'action': self.action,
            'toState': self.toState,
            'reward': self.reward
        }

    def setAgent(self, agent: Agent):
        self.agent = agent

    def getAgent(self) -> Agent:
        return self.agent

    def getAgentId(self) -> Id:
        return self.agent.getId()

    def setFromState(self, fromState: State):
        self.__originalFromState__ = fromState
        self.fromState = fromState.getCopy()
        # self.fromState = fromState

    def getFromState(self) -> State:
        # return self.fromState.getCopy()
        return self.fromState

    def getFromStateId(self) -> Id:
        return self.fromState.getId()

    def getOriginalFromState(self):
        return self.__originalFromState__

    def setPossibleActions(self, possibleActions):
        self.__originalPossibleActions__ = possibleActions
        self.possibleActions = possibleActions.getCopy()

    def getPossibleActions(self):
        return self.possibleActions

    def getPossibleActionsId(self):
        return self.possibleActions.getId()

    def setAgentKnowlege(self, agentKnowlege):
        self.__originalAgentKnowlege__ = agentKnowlege
        self.agentKnowlege = agentKnowlege.getCopy()

    def getAgentKnowlege(self):
        return self.agentKnowlege

    def getAgentKnowlegeId(self):
        return self.agentKnowlege.getId()

    def setAction(self, action: Action):
        self.__originalAction__ = action
        self.action = action.getCopy()
        # self.action = action

    def getAction(self) -> Action:
        # return self.action.getCopy()
        return self.action

    def getActionId(self) -> Id:
        return self.action.getId()

    def getOriginalAction(self):
        return self.__originalAction__

    def setToState(self, toState: State):
        self.__originalToState__ = toState
        self.toState = toState.getCopy()
        # self.toState = toState

    def getToState(self) -> State:
        # return self.toState.getCopy()
        return self.toState

    def getToStateId(self) -> Id:
        return self.toState.getId()

    def getOriginalToState(self):
        return self.__originalToState__

    def setReward(self, reward: Reward):
        self.__originalReward__ = reward
        self.reward = reward.getCopy()
        # self.reward = reward

    def getReward(self) -> Reward:
        # return self.reward.getCopy()
        return self.reward

    def getRewardId(self) -> Id:
        return self.reward.getId()

    def getOriginalReward(self):
        return self.__originalReward__

    def getCopy(self):
        return Event(
            self.getAgent(),
            self.getFromState(),
            self.getPossibleActions(),
            self.getAgentKnowlege(),
            self.getAction(),
            self.getToState(),
            self.getReward(),
            id=self.getId()
        )

    def __str__(self):
        return f'{ReflectionHelper.getClassName(self)}({c.NEW_LINE}{c.TAB}id: {self.getId()}{c.NEW_LINE}{c.TAB}- Agent:{self.agent}{c.NEW_LINE}{c.TAB}- From state:{self.fromState}{c.NEW_LINE}{c.TAB}- Possible actions:{self.possibleActions}{c.NEW_LINE}{c.TAB}- Agent knowlege:{self.agentKnowlege}{c.NEW_LINE}{c.TAB}- Action: {self.action}{c.NEW_LINE}{c.TAB}- To state: {self.toState}{c.NEW_LINE}{c.TAB}- Reward: {self.reward})'

    def __repr__(self):
        return self.__str__()
