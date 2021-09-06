from python_helper import ObjectHelper
from reinforcement_learning import hash as hashModule
from reinforcement_learning import Set, Dictionary, Tuple, List

def getWorksAsIntended():
    #arrange
    simpleSet = {1,2,3}
    anotherSimpleSet = {1,2,3}
    anotherSimpleSetInAnotherOrder = {1,3,2}
    differentSet = {4,5,6}
    bigSet = Set(simpleSet)
    anotherBigSet = Set(anotherSimpleSet)
    anotherBigSetInAnotherOrder = Set(anotherSimpleSetInAnotherOrder)
    differentBigSet = Set(differentSet)

    bigDictionary = Dictionary({
        1: simpleSet,
        2: bigSet,
        3: differentSet,
        4: anotherSimpleSetInAnotherOrder,
        5: anotherBigSetInAnotherOrder,
        6: differentBigSet
    })
    anotherBigDictionary = Dictionary({
        1: simpleSet,
        2: bigSet,
        3: differentSet,
        4: anotherSimpleSetInAnotherOrder,
        5: anotherBigSetInAnotherOrder,
        6: differentBigSet
    })
    anotherBigDictionaryInADifferentOrder = Dictionary({
        1: simpleSet,
        5: anotherBigSetInAnotherOrder,
        2: bigSet,
        3: differentSet,
        4: anotherSimpleSetInAnotherOrder,
        6: differentBigSet
    })

    complexObject = List([
        # Set({
        #     tuple(Dictionary({
        #         1: simpleSet,
        #         2: bigSet,
        #         3: differentSet,
        #         4: anotherSimpleSetInAnotherOrder,
        #         5: anotherBigSetInAnotherOrder,
        #         6: differentBigSet
        #     })),
        #     tuple(Dictionary({
        #         1: simpleSet,
        #         2: bigSet,
        #         3: differentSet,
        #         4: anotherSimpleSetInAnotherOrder,
        #         5: anotherBigSetInAnotherOrder,
        #         6: differentBigSet
        #     })),
        #     tuple(Dictionary({
        #         7: simpleSet,
        #         8: bigSet,
        #         9: differentSet,
        #         10: anotherSimpleSetInAnotherOrder,
        #         11: anotherBigSetInAnotherOrder,
        #         12: differentBigSet
        #     }))
        # })
        # , bigSet
        # , Dictionary({
        #     1: simpleSet,
        #     2: bigSet,
        #     3: differentSet,
        #     4: anotherSimpleSetInAnotherOrder,
        #     5: anotherBigSetInAnotherOrder,
        #     6: differentBigSet
        # })
        # , Dictionary({
        #     7: simpleSet,
        #     8: bigSet,
        #     9: differentSet,
        #     10: anotherSimpleSetInAnotherOrder,
        #     11: anotherBigSetInAnotherOrder,
        #     12: differentBigSet
        # })
        Dictionary({
            f'[[{hash(1)},{hash(2)}],[{hash(3)},{hash(4)}]]': Dictionary({
                1: simpleSet,
                2: bigSet,
                3: differentSet,
                4: anotherSimpleSetInAnotherOrder,
                5: Tuple(tuple(t for t in anotherBigSetInAnotherOrder)),
                6: differentBigSet
            })
        })
    ])


    #assert
    assert hashModule.get(simpleSet) == hashModule.get(anotherSimpleSet), 'hashModule.get(simpleSet) == hashModule.get(anotherSimpleSet) sould be equals'
    assert hashModule.get(bigSet) == hashModule.get(anotherBigSet), 'hashModule.get(bigSet) == hashModule.get(anotherBigSet) sould be equals'
    assert hashModule.get(bigSet) == hashModule.get(anotherBigSetInAnotherOrder), 'hashModule.get(bigSet) == hashModule.get(anotherBigSetInAnotherOrder) sould be equals'
    assert hashModule.get(bigSet) == hashModule.get(bigSet.getCopy()), 'hashModule.get(bigSet) == hashModule.get(bigSet.getCopy()) sould be equals'
    assert bigSet == bigSet.getCopy(), 'bigSet == bigSet.getCopy() sould not be equals'
    assert not simpleSet == differentSet, 'simpleSet == differentSet sould not be equals'
    assert not bigSet == differentBigSet, 'bigSet == differentBigSet sould not be equals'

    assert ObjectHelper.isCollection(bigDictionary), 'ObjectHelper.isCollection(bigDictionary) should be a colection'
    assert ObjectHelper.isDictionary(bigDictionary), 'ObjectHelper.isDictionary(bigDictionary) should be a dictionary'

    assert bigDictionary == bigDictionary.getCopy(), 'bigDictionary == bigDictionary.getCopy() should be equals'
    assert not bigDictionary == anotherBigDictionary, 'bigDictionary == anotherBigDictionary should be equals'
    assert hashModule.get(bigDictionary) == hashModule.get(anotherBigDictionaryInADifferentOrder), 'hashModule.get(bigDictionary) == hashModule.get(anotherBigDictionaryInADifferentOrder) should be equals'

    assert bigDictionary > anotherBigDictionary or bigDictionary >= anotherBigDictionary or bigDictionary < anotherBigDictionary or bigDictionary <= anotherBigDictionary
    assert ObjectHelper.getSortedCollection([bigDictionary])

    assert tuple({1,2}) == tuple({1:3,2:4})
    assert complexObject == complexObject.getCopy(), 'complexObject == complexObject.getCopy() should be equals'

    print(hashModule.get(simpleSet))
    print(hashModule.get(anotherSimpleSet))
    print(hashModule.get(bigSet))
    print(hashModule.get(bigSet.getCopy()))
    print(hashModule.get(anotherBigSet))
    print(hashModule.get(anotherBigSetInAnotherOrder))
    print(hashModule.get(bigDictionary))
    print(hashModule.get(complexObject))

###- HashTest
getWorksAsIntended()
