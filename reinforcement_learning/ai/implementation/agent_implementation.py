from operator import itemgetter
from python_helper import ObjectHelper, RandomHelper, ReflectionHelper, log

from reinforcement_learning.framework.object import Object, Id, getId
from reinforcement_learning.framework import value as valueModule
from reinforcement_learning.framework.value import List, Dictionary
from reinforcement_learning.framework.exception import (
    MethodNotImplementedException
)

from reinforcement_learning.ai import agent as agentModule
from reinforcement_learning.ai.history import History
from reinforcement_learning.ai.state import State
from reinforcement_learning.ai.action import Action
from reinforcement_learning.ai.event import Event
from reinforcement_learning.ai.reward import Reward
from reinforcement_learning.ai.agent import Agent, AgentConstants



class RandomAgent(Agent):

    def __init__(self, *args, **kwargs):
        Agent.__init__(self, *args, **kwargs)

    def getAction(self, state: State, possibleActions: List) -> tuple:
        return List(), RandomHelper.sample(possibleActions)

    def _update(self, history: History, isFinalState: bool):
        ...


class MonteCarloEpisodeAgent(Agent):

    FULL_EXPLORATION: float = 1.0
    NO_EXPLORATION: float = 0.0
    FULL_RETENTION: float = 1.0
    NO_RETENTION: float = 0.0
    DEFAULT_EXPLORATION_TARGET: float  = float(0.05)
    DEFAULT_TRAINNING_ITERATIONS: int = 0
    DEFAULT_TRAINNING_BATCH_SIZE: int = 0

    def __init__(self,
        *args,
        exploration: float = FULL_EXPLORATION,
        retention: float = FULL_RETENTION,
        explorationTarget: float  = DEFAULT_EXPLORATION_TARGET,
        **kwargs
    ):
        Agent.__init__(self, *args, **kwargs)
        self.exploration: float = float(exploration)
        self.explorationTarget: float = float(explorationTarget)
        self.retention: float = float(retention)
        self.initialExploration: float = float(self.exploration)
        self.initialExplorationTarget: float = float(self.explorationTarget)
        self.initialRetention: float = float(self.retention)

    def getAction(self, state: State, possibleActions: List) -> tuple:
        if self.exploration > RandomHelper.integer(minimum=0, maximum=100) / 100:
            return List(), RandomHelper.sample(possibleActions)
        visits = self.actionTable.get(state.getHash(), self._getDefaultStateActions())[AgentConstants.ACTIONS]
        actionList = [
            visit[AgentConstants.ACTION] for visit in sorted(visits, key=itemgetter(AgentConstants.ACTION_VALUE)) if visit[AgentConstants.ACTION].getHash() in [
                action.getHash() for action in possibleActions
            ]
        ]
        action: Action = actionList[-1] if ObjectHelper.isNotEmpty(actionList) else RandomHelper.sample(
            [action for action in possibleActions]
        )
        return List(visits), action.getCopy()

    def _getDefaultStateActions(self):
        return Dictionary({
            AgentConstants.ID: getId(),
            AgentConstants.ACTIONS: List([])
        })

    def _update(self, history: History, isFinalState: bool):
        if isFinalState:
            rewardValue = self._getRewardValue(history)
            filteredHistory = self._getAgentHistory(history)
            # print(f'filteredHistory: {filteredHistory}, len(filteredHistory): {len(filteredHistory)}')
            # print(f'Agent: {self.key}, Reward value: {rewardValue}, len(history): {len(history)}')
            for deepnes, event in enumerate(filteredHistory[::-1]):
                rewardTamed: float = rewardValue * (self.retention**deepnes)
                # print(f'rewardTamed: {rewardTamed}')
                # print(self.actionTable)
                stateHash: str = event.fromState.getHash()
                if stateHash not in self.actionTable:
                    self.actionTable[stateHash] = self._getDefaultStateActions()
                    self._appendFirstVisit(stateHash, event, rewardTamed)
                else:
                    visit = self.accessByAction(stateHash, event.action)
                    if ObjectHelper.isEmpty(visit):
                        self._appendFirstVisit(stateHash, event, rewardTamed)
                    else:
                        visit[AgentConstants.ACTION_VISITS] += 1
                        visit[AgentConstants.ACTION_VALUE] += (rewardTamed - visit[AgentConstants.ACTION_VALUE]) / visit[AgentConstants.ACTION_VISITS]

    def _appendFirstVisit(self, stateHash: str, event: Event, rewardTamed: float):
        self.actionTable[stateHash][AgentConstants.ACTIONS].append(
            Dictionary({
                AgentConstants.ACTION: event.action,
                AgentConstants.ACTION_VALUE: rewardTamed,
                AgentConstants.ACTION_VISITS: 1
            })
        )

    def _getAgentHistory(self, history: History) -> History:
        return History([
            event for event in history if event.getAgent().getKey() == self.getKey()
        ])

    def _getRewardValue(self, history: History):
        return history[-1].reward.get(self.key)

    def freezeInternalState(self):
        self.currentExploration: float = float(self.exploration)
        self.exploration: float = float(self.NO_EXPLORATION)
        self.currentRetention: float = float(self.retention)
        self.retention: float = float(self.NO_RETENTION)
        self.byPassUpdate()

    def updateInternalState(self):
        self.exploration: float = self.currentExploration * self.explorationReducingRatio
        self.retention: float = float(self.currentRetention)
        self.activateUpdate()

    def _newTrainning(self, totalTrainningIterations: int, trainningBatchSize: int):
        self.totalTrainningIterations: int = int(totalTrainningIterations)
        self.trainningBatchSize: int = int(trainningBatchSize)
        self.currentExploration = float(self.initialExploration)
        self.exploration = float(self.initialExploration)
        self.explorationTarget = float(self.initialExplorationTarget)
        self.retention = float(self.initialRetention)
        self.explorationReducingRatio = agentModule.getExplorationReducingRatio(self.exploration, self.explorationTarget, self.totalTrainningIterations)
        log.debug(self._newTrainning, f'Exploration reducing ratio: {self.explorationReducingRatio}')

    def _finishTrainning(self):
        self.exploration = float(self.NO_EXPLORATION)
        self.retention = float(self.FULL_RETENTION)
        self.totalTrainningIterations: int = int(self.DEFAULT_TRAINNING_ITERATIONS)
        self.trainningBatchSize: int = int(self.DEFAULT_TRAINNING_BATCH_SIZE)

    def getInternalStateDescription(self):
        return f'{self.getKey()} agent -> isTrainning: {self.isTrainning()}, exploration: {self.exploration}, retention: {self.retention}'
