import gym

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


from python_helper import Constant as c
from python_helper import ObjectHelper, StringHelper, RandomHelper, log, SettingHelper
from reinforcement_learning import value as valueModule
from reinforcement_learning import MonteCarloEpisodeAgent, RandomAgent, Agent, Action, Environment, Episode, History, State, Reward, List, Tuple, Set, Dictionary, Id

CART_POLE_V1 = 'CartPole-v1'


class CartPoleV1EnvironmentImpl(Environment):

    def __init__(
        self,
        *args,
        **kwargs
    ):
        self.__originalArgs__ = [
        ]
        self.__originalKwargs__ = {
        }
        self.gymEnvironment = gym.make(CART_POLE_V1)
        self._reset()
        Environment.__init__(self, self._getCurrentStateFromGymState(), *args, **kwargs)

    def getPossibleActions(self):
        return List([Tuple((n,)) for n in range(self.gymEnvironment.action_space.n)])

    def _getCurrentStateFromGymState(self):
        return State([round(r, 2) for r in self.gymState])

    def updateState(self, action: Action, agents: List, willBeEpisodeMaxHistoryLenght: bool) -> tuple:
        # print('updating state')
        fromState = self.getState()
        self._validateNotFinished(fromState)
        self._stepFoward(action=action)
        toState: State = State(self._getCurrentStateFromGymState(), validate=False)
        # print(fromState, action, toState)
        self.setState(toState)
        isFinalState: bool = self.isFinalState(state=toState, isEpisodeMaxHistoryLenght=willBeEpisodeMaxHistoryLenght)
        reward: Reward = self.getReward(fromState, toState, agents, isFinalState, willBeEpisodeMaxHistoryLenght)
        # print(f'isFinalState: {isActualyFinalState}')
        # print('updating state finished')
        return toState, reward, isFinalState

    def _stepFoward(self, action: Action = None):
        if ObjectHelper.isNotNone(action):
            self.gymState, self.gymReward, self.gymDone, self.gymInfo = self.gymEnvironment.step(action[0])
        else:
            self.gymState, self.gymReward, self.gymDone, self.gymInfo = self.gymEnvironment.step(self.gymEnvironment.action_space.sample())

    def prepareNextState(self):
        ...

    def getReward(self, fromState: State, toState: State, agents: Dictionary, isFinalState: bool, isEpisodeMaxHistoryLenght: bool) -> Reward:
        # print('getting reward')
        # print(f"Agents: {agents}")
        # self._validateNotFinished(fromState)
        # print('getting reward finished')

        # print(Reward({
        #     k: float(self.gymReward) if self.finalStateReached else float(0.0) for k, v in agents.items()
        # }))
        reward: Reward = Reward({
            k: float(self.gymReward) if (
                isEpisodeMaxHistoryLenght
            ) or (
                isEpisodeMaxHistoryLenght and isFinalState
            ) or (
                not isFinalState
            ) else float(0.0) for k, v in agents.items()
        })
        # print(f'      ---> reward: {reward}, isFinalState: {isFinalState}, isEpisodeMaxHistoryLenght: {isEpisodeMaxHistoryLenght}')
        return reward

    def isFinalState(self, state: State = None, isEpisodeMaxHistoryLenght: bool = None) -> bool:
        self.finalStateReached: bool = self.finalStateReached or bool(self.gymDone) or (
            bool(False if ObjectHelper.isNone(isEpisodeMaxHistoryLenght) else isEpisodeMaxHistoryLenght)
        )
        # print(f'      ---> is final state: {bool(self.finalStateReached)}')
        return bool(self.finalStateReached)

    def printState(self, data: str = c.BLANK):
        print(f'{c.NEW_LINE} {self.key}')
        if StringHelper.isNotBlank(data) :
            print(data)
        self.gymEnvironment.render()
        # print('end of print state')

    def _validateNotFinished(self, fromState: State):
        if self.isFinalState(state=fromState):
            raise Exception(f'Episode should be finished: {fromState}')

    def _reset(self):
        self.gymEnvironment.reset()
        self._stepFoward()
        self.finalStateReached: bool = False
