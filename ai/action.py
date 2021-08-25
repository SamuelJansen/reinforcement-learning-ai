from reinforcement_learning.framework.object import Object
from reinforcement_learning.framework.value import List


class Action(List):

    def __init__(self, actionValue: List, id: str = None):
        List.__init__(self, actionValue, id=id)

    def getCopy(self):
        return Action(self, id=self.getId())

    def __eq__(self, other):
        # print(f'here: {other.getId()} == {self.getId()}')
        return isinstance(other, Action) and other.getHash() == self.getHash()

    def __ne__(self, other):
        return not self.__eq__(other)
