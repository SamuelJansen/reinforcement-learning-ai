from numbers import Number
from python_helper import Constant as c
from python_helper import StringHelper, log, ObjectHelper, ReflectionHelper

def get(value: object):
    # return newHash(value)
    # return oldHash(value)
    # return oldAndSafeHash(value)
    return simpleHash(value)


def oldHash(value: object):
    return StringHelper.join(
        str(value).strip().replace(c.NEW_LINE, c.BLANK).split(),
        character=c.BLANK
    )


def simpleHash(value: object):
    try:
        if ObjectHelper.isSet(value):
            return StringHelper.join([
                    c.OPEN_SET,
                    StringHelper.join([simpleHash(v) for v in sortIt(value)], character=c.COMA),
                    c.CLOSE_SET
                ],
                character=c.BLANK
            )
        elif ObjectHelper.isDictionary(value):
            return StringHelper.join([
                    c.OPEN_DICTIONARY,
                    StringHelper.join([f'{simpleHash(k)}{c.COLON}{simpleHash(v)}' for k,v in value.items()], character=c.COMA),
                    c.CLOSE_DICTIONARY
                ],
                character=c.BLANK
            )
        elif ObjectHelper.isTuple(value):
            return StringHelper.join([
                    c.OPEN_TUPLE,
                    StringHelper.join([simpleHash(v) for v in value], character=c.COMA),
                    c.CLOSE_TUPLE
                ],
                character=c.BLANK
            )
        elif ObjectHelper.isList(value):
            return StringHelper.join([
                    c.OPEN_LIST,
                    StringHelper.join([simpleHash(v) for v in value], character=c.COMA),
                    c.CLOSE_LIST
                ],
                character=c.BLANK
            )
        elif ObjectHelper.isCollection(value):
            return StringHelper.join([
                    c.OPEN_LIST,
                    *[simpleHash(v) for v in value],
                    c.OPEN_LIST
                ],
                character=c.COMA
            )
        elif ObjectHelper.isNativeClassInstance(value):
            hash = str(value).strip() if isinstance(value, str) else str(value).strip()
            return hash if c.BLANK not in hash else StringHelper.join(hash.strip().split(), character=c.BLANK)
        else:
            # try:
            #     return value.getHash()
            # except:
            #     return oldHash(value)
            return oldHash(value)
            # return StringHelper.join(
            #     [simpleHash(v) for v in ReflectionHelper.getAttributePointerList(value)],
            #     character=c.BLANK
            # )
    except Exception as exception:
        print(value)
        raise exception


def sortIt(thing) :
    if ObjectHelper.isDictionary(thing) :
        sortedDictionary = {}
        for key in getSortedCollection(thing) :
            sortedDictionary[key] = sortIt(thing[key])
        return sortedDictionary
    elif ObjectHelper.isCollection(thing) :
        newCollection = []
        for innerValue in thing :
            newCollection.append(sortIt(innerValue))
        return getSortedCollection(newCollection)
    else :
        return thing

def getSortedCollection(thing) :
    return thing if (
        ObjectHelper.isNotCollection(thing) or ObjectHelper.isEmpty(thing)
    ) or (
        ObjectHelper.isNotDictionary(thing) and ObjectHelper.isNotSet(thing) and ObjectHelper.isDictionary(thing[0])
    ) else sorted(
        thing,
        key=lambda x: (
            x is not None, c.NOTHING if isinstance(x, Number) else type(x).__name__, x
        )
    )


# def simpleHash(value: object):
#     if ObjectHelper.isDictionary(value):
#         return StringHelper.join(
#             [f'{simpleHash(k)}:{simpleHash(v)}' for k,v in sortIt(value).items()],
#             character=c.COMA
#         )
#     elif ObjectHelper.isCollection(value):
#         return StringHelper.join(
#             [simpleHash(v) for v in sortIt(value)],
#             character=c.COMA
#         )
#     elif ObjectHelper.isNativeClassInstance(value):
#         return str(value)
#     else:
#         return oldHash(value)
#         # return StringHelper.join(
#         #     [simpleHash(v) for v in ReflectionHelper.getAttributePointerList(value)],
#         #     character=c.BLANK
#         # )


# def oldAndSafeHash(value: object):
#     hash = None
#     try :
#         hash = StringHelper.join(
#             str(value).strip().replace(c.NEW_LINE, c.BLANK).split(),
#             character=c.BLANK
#         )
#     except Exception as exception:
#         errorMessage = f'Not possible do get hash of {value.getId()}'
#         log.failure(get, errorMessage, exception)
#         raise Exception(f'{errorMessage}. Cause: {exception}')
#     return hash


# def newHash(value: object):
#     hash = None
#     try :
#         if ObjectHelper.isNativeClassInstance(value):
#             hash = str(value)
#         elif ObjectHelper.isCollection(value):
#             hash = StringHelper.join(
#                 [newHash(v) for v in sortIt(value)],
#                 character=c.BLANK
#             )
#         else:
#             hash = StringHelper.join(
#                 [newHash(v) for v in ReflectionHelper.getAttributePointerList(value)],
#                 character=c.BLANK
#             )
#     except Exception as exception:
#         errorMessage = f'Not possible do get hash of {value.getId()}'
#         log.failure(get, errorMessage, exception)
#         raise Exception(f'{errorMessage}. Cause: {exception}')
#     return hash
