import gym

from python_helper import Constant as c
from python_helper import ObjectHelper, StringHelper, RandomHelper, log, SettingHelper

from reinforcement_learning import value as valueModule
from reinforcement_learning import MonteCarloEpisodeAgent, RandomAgent, Agent, Action, Environment, Episode, History, State, Reward, List, Tuple, Set, Dictionary, Id
from reinforcement_learning import environmentModule


# env = gym.make('CartPole-v1')
# env is created, now we can use it:
# for episode in range(10):
#     obs = env.reset()
#     for step in range(50):
#         action = env.action_space.sample()  # or given a custom model, action = policy(observation)
#         print(env.action_space, action)
#         nobs, reward, done, info = env.step(action)
#         print(nobs, reward, done, info)
#         env.render()


class CartPoleV1EnvironmentImpl(Environment):

    ZERO_REWARD: float = float(0.0)
    CART_POLE_V1 = 'CartPole-v1'

    def __init__(
        self,
        *args,
        agentKey: str = None,
        **kwargs
    ):
        self.__originalArgs__ = [
        ]
        self.__originalKwargs__ = {
            'agentKey': agentKey
        }
        self.currentAgentKey: str = str(agentKey)
        self.gymEnvironment = gym.make(self.CART_POLE_V1)
        Environment.__init__(self, *args, **kwargs)

    def getInitialState(self, state: State):
        if ObjectHelper.isNotNone(state):
            return state
        self._reset()
        return self._getCurrentStateFromGymState()


    def setState(self, state: State):
        self.state = state.getCopy()

    def getCurrentAgentKey(self) -> str:
        return str(self.currentAgentKey)

    def getPossibleActions(self):
        return List([Action([(n,)]) for n in range(self.gymEnvironment.action_space.n)])

    def _getCurrentStateFromGymState(self):
        return State([round(r, 2) for r in self.gymState])

    def updateState(self, action: Action, episode: environmentModule.ShouldBeEpisode) -> tuple:
        fromState = self.getState()
        self._validateEpisodeNotFinishedWhileUpdating(fromState)

        self._stepFoward(action=action)
        toState: State = State(self._getCurrentStateFromGymState(), validate=False)
        self.setState(toState)

        isFinalState: bool = self.isFinalState(state=toState, episode=episode)
        reward: Reward = self.getReward(fromState, toState, episode, isFinalState)
        return toState, reward, isFinalState

    def prepareNextState(self):
        ...

    def getReward(self, fromState: State, toState: State, episode: environmentModule.ShouldBeEpisode, isFinalState: bool) -> Reward:
        return Reward({
            agentKey: float(self.gymReward) if (
                episode.isMaxHistoryLenght() or not isFinalState
            ) else float(self.ZERO_REWARD) for agentKey, agent in episode.agents.items()
        })

    def _isInFinalStateCondition(self, state: State) -> bool:
        return bool(self.gymDone)

    def printState(self, data: str = c.BLANK):
        if StringHelper.isNotBlank(data) :
            print(data)
        self.gymEnvironment.render()
        # print('end of print state')

    def _validateEpisodeNotFinishedWhileUpdating(self, fromState: State):
        if self._isInFinalStateCondition(state=fromState):
            raise Exception(f'Episode should be finished: {fromState}')

    def _stepFoward(self, action: Action = None):
        if ObjectHelper.isNotNone(action):
            self.gymState, self.gymReward, self.gymDone, self.gymInfo = self.gymEnvironment.step(action[0][0])
        else:
            self.gymState, self.gymReward, self.gymDone, self.gymInfo = self.gymEnvironment.step(self.gymEnvironment.action_space.sample())

    def _reset(self):
        self.gymEnvironment.reset()
        self._stepFoward()
