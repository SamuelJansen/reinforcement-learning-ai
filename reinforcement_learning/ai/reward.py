from python_helper import Constant as c
from python_helper import ReflectionHelper, ObjectHelper, StringHelper
from reinforcement_learning.framework.object import Object, Id
from reinforcement_learning.framework.value import Dictionary
from reinforcement_learning.ai.action import Action


def raiseInvalidAgentKeyException():
    raise Exception("Invalid agent key")


class Reward(Dictionary):

    def __init__(
        self,
        reward: Dictionary = None,
        id: Id = None
    ):
        Dictionary.__init__(self, reward, id=id)
        # self.setValue(reward)

    # def getValue(self, agentKey: str) -> float:
    #     # print(agentKey)
    #     # print(self.value)
    #     return self.value.get(agentKey) if agentKey in self.value else raiseInvalidAgentKeyException()

    # def setValue(self, value: Dictionary):
    #     self.value = Dictionary(value)

    def getCopy(self):
        return Reward(
            reward=self,
            id=self.getId()
        )

    # def asJson(self):
    #     return self.value.getCopy()

    # def __str__(self):
    #     return StringHelper.join(
    #         [
    #             c.OPEN_DICTIONARY,
    #             StringHelper.join(
    #                 [f'{agentKey}{c.COLON_SPACE}{reward}' for agentKey, reward in self.value.items()],
    #                 character=c.COMA_SPACE
    #             ),
    #             c.CLOSE_DICTIONARY
    #         ],
    #         character=c.BLANK
    #     )
    
    # def __repr__(self):
    #     return self.__str__()
