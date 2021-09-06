import numpy as np
from operator import itemgetter
from python_helper import Constant as c
from python_helper import ObjectHelper, ReflectionHelper
from reinforcement_learning.framework.object import Object, Id
from reinforcement_learning.framework.exception import (
    IncorrectTypeException
)
from copy import deepcopy


def sortedBy(collectionList, valueKey):
    return sorted(collectionList, key=itemgetter(valueKey))


def getSimpleName(givenType):
    name = ReflectionHelper.getName(givenType)
    return name if c.DOT not in name else name.split(c.DOT)[-1]


def getExplorationReducingRatio(originalExploration: float, target: float, totalTrainningIterations: float):
    return (target * originalExploration) ** (originalExploration / float(totalTrainningIterations))


class Value(Object):

    def __init__(self, instance, id: Id = None):
        # print(f'instance - value: {instance}')
        Object.__init__(self, id=id)
        # print('yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy')
        # print(f'getting copy of {instance}')
        self.__original__ = None ###- self.__makeCopy__(instance, validate=False)
        # print('yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy')
        ### - did not work self.__hash__ = str(self.__ai_hash__)

    def getCopy(self):
        return self.__makeCopy__(self, validate=True)

    def getOriginal(self):
        return self.__makeCopy__(self.__original__, validate=False)

    def __makeCopy__(self, instance, validate: bool = True):
        return makeCopy(instance, validate)

    def __eq__(self, other):
        return super(Object, self).__eq__(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return self.__str__()


class Set(set, Value):

    def __init__(self, instance=None, validate=True, id=None):
        # print(f'====================== init Set, validate: {validate}')
        if ObjectHelper.isNone(instance):
            instance = set()
        validateType(instance, set, validate)
        set.__init__(self, instance)
        Value.__init__(self, instance, id=id)
        # print('====================== end Set')

    def __eq__(self, other):
        return super(Value, self).__eq__(other)

    def __ne__(self, other):
        return not self.__eq__(other)


class Tuple(tuple, Value):

    def __init__(self, instance=None, validate=True, id=None):
        # print(f'====================== init Tuple, validate: {validate}')
        if ObjectHelper.isNone(instance):
            instance = tuple()
        validateType(instance, tuple, validate)
        tuple.__init__(instance)
        Value.__init__(self, instance, id=id)
        # print('====================== end Tuple')

    def __eq__(self, other):
        return super(Value, self).__eq__(other)

    def __ne__(self, other):
        return not self.__eq__(other)


class List(list, Value):

    def __init__(self, instance=None, validate=True, id=None):
        # print(f'instance - list: {instance}')
        # print(f'====================== init List, validate: {validate}')
        if ObjectHelper.isNone(instance):
            instance = list()
        validateType(instance, list, validate)
        list.__init__(self, instance)
        Value.__init__(self, instance, id=id)
        # print('====================== end List')

    def __eq__(self, other):
        return super(Value, self).__eq__(other)

    def __ne__(self, other):
        return not self.__eq__(other)


class NpArrayList(type(np.array([])), Value):

    def __init__(self, instance=None, validate=True, id=None):
        # print(f'instance - list: {instance}')
        # print(f'====================== init List, validate: {validate}')
        if ObjectHelper.isNone(instance):
            instance = list()
        validateType(instance, list, validate)
        np.array.__init__(self, instance)
        Value.__init__(self, instance, id=id)
        # print('====================== end List')

    def __eq__(self, other):
        return super(Value, self).__eq__(other)

    def __ne__(self, other):
        return not self.__eq__(other)


class NpNdArrayList(type(np.ndarray([])), Value):

    def __init__(self, instance=None, validate=True, id=None):
        # print(f'instance - list: {instance}')
        # print(f'====================== init List, validate: {validate}')
        if ObjectHelper.isNone(instance):
            instance = list()
        validateType(instance, list, validate)
        np.ndarray.__init__(self, instance)
        Value.__init__(self, instance, id=id)
        # print('====================== end List')

    def __eq__(self, other):
        return super(Value, self).__eq__(other)

    def __ne__(self, other):
        return not self.__eq__(other)


class Dictionary(dict, Value):

    def __init__(self, instance=None, validate=True, id=None):
        # print(f'====================== init Dictionary, validate: {validate}')
        if ObjectHelper.isNone(instance):
            instance = dict()
        validateType(instance, dict, validate)
        dict.__init__(self, instance)
        Value.__init__(self, instance, id=id)
        # print('====================== end Dictionary')

    def __eq__(self, other):
        return super(Value, self).__eq__(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        return self.__id__ < other.__id__

    def __le__(self, other):
        return self.getId() <= other.__id__

    def __gt__(self, other):
        return self.getId() > other.__id__

    def __ge__(self, other):
        return self.__id__ >= other.__id__


VALUE_IMPLEMENTATIONS = {
    getSimpleName(set): Set,
    getSimpleName(tuple): Tuple,
    getSimpleName(list): List,
    getSimpleName(dict): Dictionary,
    getSimpleName(object): Value
}

IMPLEMENTATIONS_MAP = {
    ** VALUE_IMPLEMENTATIONS,
    getSimpleName(Set): Set,
    getSimpleName(Tuple): Tuple,
    getSimpleName(List): List,
    getSimpleName(Dictionary): Dictionary,
    getSimpleName(Value): Value
}


def indexOf(element, collection, byKey=False) :
    if ObjectHelper.isCollection(collection):
        if ObjectHelper.isSet(collection):
            raise Exception('Set is not indexable')
        if ObjectHelper.isDictionary(collection):
            collectionValues = List(collection.keys()) if byKey else List(collection.values())
        else:
            collectionValues = List(collection)
        for index in range(len(collectionValues)):
            if collectionValues[index] == element and collectionValues[index] is element :
                return index
        raise Exception(f'{element} not found in {collection}')
    raise Exception(f'{collection} is not a collection')


def makeCopy(instance, validate):
    # print(f'making copy of {type(instance)}')
    if ObjectHelper.isNone(instance):
        return None
    elif isinstance(instance, Dictionary):
        # print('here')
        return type(instance)(
            {makeCopy(k, validate): makeCopy(v, validate) for k, v in instance.items()},
            validate=False,
            id=instance.getId()
        )
    elif isImplementation(instance):
        return type(instance)(
            [makeCopy(element, validate) for element in instance],
            validate=False,
            id=instance.getId()
        )
    elif isinstance(instance, dict):
        return {makeCopy(k, validate): makeCopy(v, validate) for k, v in instance.items()}
    elif getSimpleName(type(instance)) in VALUE_IMPLEMENTATIONS.keys():
        return type(instance)(
            [makeCopy(element, validate) for element in instance]
        )
    # print(f'         type: {type(instance)}')
    # print(f'making deepcopy of {type(instance)}')
    return deepcopy(instance)


def validateType(instance, givenType, validate):
    # print(instance)
    # print(givenType)
    if validate:
        if ObjectHelper.isNone(
            givenType
        ) or ObjectHelper.isNone(
            getImplementationOf(givenType)
            # VALUE_IMPLEMENTATIONS.get(givenType)
        ) or not (isinstance(
            instance,
            givenType
        ) or isinstance(
            instance,
            getImplementationOf(givenType)
            # VALUE_IMPLEMENTATIONS.get(givenType)
        )):
            raise IncorrectTypeException(instance, givenType)


def getImplementationOf(givenType, default=False):
    typeName = getSimpleName(givenType)
    if typeName.endswith('Impl'):
        typeName = typeName[:-len('Impl')]
    return IMPLEMENTATIONS_MAP.get(typeName, givenType if default else None)


def isImplementation(instance):
    # print(f'{type(instance)} in VALUE_IMPLEMENTATIONS.values(): {type(instance) in VALUE_IMPLEMENTATIONS.values()}')
    return type(instance) in VALUE_IMPLEMENTATIONS.values()



###- Test
# a_b_c = Tuple((
#     List([
#         Dictionary({
#             '1': Set({'a', 'b', 'c'}),
#             '2': Set({'d', 'e', 'f'}),
#             '3': Set({'g', 'h', 'i'})
#         }),
#         Dictionary({
#             '4': Set({'a', 'b', 'c'}),
#             '5': Set({'d', 'e', 'f'}),
#             '6': Set({'g', 'h', 'i'})
#         }),
#         Dictionary({
#             '7': Set({'a', 'b', 'c'}),
#             '8': Set({'d', 'e', 'f'}),
#             '9': Set({'g', 'h', 'i'})
#         })
#     ]),
#     List([
#         Dictionary({
#             '1': Set({'j', 'k', 'l'}),
#             '2': Set({'m', 'n', 'o'}),
#             '3': Set({'p', 'q', 'r'})
#         }),
#         Dictionary({
#             '4': Set({'j', 'k', 'l'}),
#             '5': Set({'m', 'n', 'o'}),
#             '6': Set({'p', 'q', 'r'})
#         }),
#         Dictionary({
#             '7': Set({'j', 'k', 'l'}),
#             '8': Set({'m', 'n', 'o'}),
#             '9': Set({'p', 'q', 'r'})
#         })
#     ]),
#     List([
#         Dictionary({
#             '1': Set({'s', 't', 'u'}),
#             '2': Set({'v', 'w', 'x'}),
#             '3': Set({'y', 'z', '0'})
#         }),
#         Dictionary({
#             '4': Set({'s', 't', 'u'}),
#             '5': Set({'v', 'w', 'x'}),
#             '6': Set({'y', 'z', '0'})
#         }),
#         Dictionary({
#             '7': Set({'s', 't', 'u'}),
#             '8': Set({'v', 'w', 'x'}),
#             '9': Set({'y', 'z', '0'})
#         })
#     ])
# ))
# a_b_cCopy = a_b_c.getCopy()
# print(a_b_c.__ai_hash__)
# print(a_b_cCopy.__ai_hash__)
# print(a_b_c.__ai_hash__ == a_b_cCopy.__ai_hash__)
# print(a_b_c.getHash())
# print(a_b_cCopy.getHash())
# print(a_b_c.getHash() == a_b_cCopy.getHash())
# # print(a_b_c.getHash(), a_b_cCopy.getHash())
# assert a_b_c == a_b_cCopy, 'a_b_c == a_b_cCopy should be equals'
# a_b_cCopy[0][0]['1'].add(None)
# ###- assert not a_b_c == a_b_cCopy, 'a_b_c == a_b_cCopy after added None should not be equals'
# print(a_b_c == a_b_cCopy)
#
# print(isinstance(dict(), Dictionary))
# print(isinstance(Dictionary(), dict))
# print(Dictionary({'b': 'c', 'a': 'd'}).getHash() == Dictionary({'a': 'd', 'b': 'c'}).getHash())
# print(f"{Dictionary({'b': 'c', 'a': 'd'}).getHash()} == {Dictionary({'a': 'd', 'b': 'c'}).getHash()}: {Dictionary({'b': 'c', 'a': 'd'}).getHash() == Dictionary({'a': 'd', 'b': 'c'}).getHash()}")
#
# print(List(['b', 'c', 'a', 'd']).getHash() == List(['a', 'd', 'b', 'c']).getHash())
# print(f"{List(['b', 'c', 'a', 'd']).getHash()} == {List(['a', 'd', 'b', 'c']).getHash()}: {List(['b', 'c', 'a', 'd']).getHash() == List(['a', 'd', 'b', 'c']).getHash()}")
#
# someDict = Dictionary({'a': 'b'})
# someDictCopy = someDict.getCopy()
# print(f'{someDict.getId()} == {someDictCopy.getId()}: {someDict.getId() == someDictCopy.getId()}')
