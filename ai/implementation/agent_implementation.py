from operator import itemgetter
from python_helper import ObjectHelper, RandomHelper, ReflectionHelper, log
from reinforcement_learning.framework.object import Object, Id, getId
from reinforcement_learning.framework import value as valueModule
from reinforcement_learning.framework.value import List, Dictionary
from reinforcement_learning.ai.history import History
from reinforcement_learning.ai.state import State
from reinforcement_learning.ai.action import Action
from reinforcement_learning.ai.event import Event
from reinforcement_learning.ai.reward import Reward
from reinforcement_learning.ai.agent import Agent, AgentConstants
from reinforcement_learning.framework.exception import (
    MethodNotImplementedException
)


class RandomAgent(Agent):

    def __init__(self, *args, **kwargs):
        Agent.__init__(self, *args, **kwargs)

    def getAction(self, state: State, possibleActions: List):
        return RandomHelper.sample(possibleActions)

    def _update(self, history: History, reward: Reward, isFinalState: bool):
        ...


class MonteCarloEpisodeAgent(Agent):
    def __init__(self, *args, **kwargs):
        Agent.__init__(self, *args, **kwargs)

    def getAction(self, state: State, possibleActions: List):
        # print('agent getting action')
        if self.exploration > RandomHelper.integer(minimum=0, maximum=100) / 100:
            return RandomHelper.sample(possibleActions)
        # actionTable = self.getActionTable()
        stateHash: str = state.getHash()
        visits = self.actionTable.get(stateHash, self.getDefaultStateActions())[AgentConstants.ACTIONS]
        actionList = [
            visit[AgentConstants.ACTION] for visit in sorted(visits, key=itemgetter(AgentConstants.ACTION_VALUE)) if visit[AgentConstants.ACTION].getHash() in [
                action.getHash() for action in possibleActions
            ]
        ]
        action: Action = actionList[-1] if ObjectHelper.isNotEmpty(actionList) else RandomHelper.sample(
            [action for action in possibleActions]
        )
        # print('agent action get')
        return action.getCopy()

    def getDefaultStateActions(self):
        return {
            AgentConstants.ID: getId(),
            AgentConstants.ACTIONS: []
        }

    def _update(self, history: History, reward: Reward, isFinalState: bool):
        # print('agent updating state')
        if isFinalState:
            rewardValue = reward.getValue(self.key)
            for deepnes, event in enumerate(history[::-1]):
                rewardTamed: float = rewardValue * (self.retention**deepnes)
                # print(self.actionTable)
                stateHash: str = event.fromState.getHash()
                if stateHash not in self.actionTable:
                    self.actionTable[stateHash] = self.getDefaultStateActions()
                    self._appendFirstVisit(stateHash, event, rewardTamed)
                else:
                    visit = self.accessByAction(stateHash, event.action)
                    if ObjectHelper.isEmpty(visit):
                        self._appendFirstVisit(stateHash, event, rewardTamed)
                    else:
                        visit[AgentConstants.ACTION_VISITS] += 1
                        visit[AgentConstants.ACTION_VALUE] += (rewardTamed - visit[AgentConstants.ACTION_VALUE]) / visit[AgentConstants.ACTION_VISITS]
        # print('agent state updated')

    def _appendFirstVisit(self, stateHash: str, event: Event, rewardTamed: float):
        self.actionTable[stateHash][AgentConstants.ACTIONS].append(
            {
                AgentConstants.ACTION: event.action,
                AgentConstants.ACTION_VALUE: rewardTamed,
                AgentConstants.ACTION_VISITS: 1
            }
        )
