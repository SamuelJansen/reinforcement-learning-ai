import secrets
from python_helper import ReflectionHelper, ObjectHelper, RandomHelper
from reinforcement_learning.framework import hash
from reinforcement_learning.framework.exception import (
    MethodNotImplementedException
)

DEFAULT_ID_SIZE: int = 24


class Id(str):
    ...


class Object:

    def __init__(self, id: Id = None):
        self.__id__ = getId(id=id)
        self.updateHash()

    def getId(self):
        return str(self.__id__)

    def getHash(self):
        self.updateHash()
        return str(self.__ai_hash__)

    def getType(self):
        return ReflectionHelper.getClassName(self)

    def getCopy(self):
        raise MethodNotImplementedException()

    def updateHash(self):
        self.__ai_hash__ = hash.get(self)

    def __eq__(self, other):
        # print(f'here: {other.getId()} == {self.getId()}')
        return isinstance(other, Object) and other.getHash() == self.getHash() and other.getId() == self.getId()

    def __ne__(self, other):
        return not self.__eq__(other)

    # def __str__(self):
    #     return f'{ReflectionHelper.getClassName(self)}(id: {self.getId()})'
    #
    # def __repr__(self):
    #     return self.__str__()


def getId(id: str = None):
    size = DEFAULT_ID_SIZE // 2
    return id if ObjectHelper.isNotNone(id) else Id(
        secrets.token_hex((DEFAULT_ID_SIZE - size) // 2) + RandomHelper.string(minimum=size, maximum=size)
    )
