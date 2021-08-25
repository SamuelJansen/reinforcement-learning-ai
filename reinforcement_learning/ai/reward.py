from python_helper import ReflectionHelper, ObjectHelper
from reinforcement_learning.framework.object import Object, Id
from reinforcement_learning.ai.action import Action


def raiseInvalidAgentKeyException():
    raise Exception("Invalid agent key")


class Reward(Object):

    def __init__(
        self,
        value: dict,
        id: Id = None
    ):
        Object.__init__(self, id=id)
        self.setValue(value)

    def getValue(self, agentKey: str) -> float:
        # print(agentKey)
        # print(self.value)
        return self.value.get(agentKey) if agentKey in self.value else raiseInvalidAgentKeyException()

    def setValue(self, value: dict):
        self.value = value

    def getCopy(self):
        return Reward(
            self.value,
            id=self.getId()
        )
