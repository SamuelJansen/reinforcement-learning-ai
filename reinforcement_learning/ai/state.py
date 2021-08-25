from reinforcement_learning.framework.object import Id
from reinforcement_learning.framework import value
from reinforcement_learning.framework.value import Value, List


class State(List):

    def __init__(self, initialState: List, *args, id: Id = None, **kwargs):
        List.__init__(self, initialState, *args, id=id, **kwargs)

    def __eq__(self, other):
        return super(List, self).__eq__(other)

    def __ne__(self, other):
        return not self.__eq__(other)
