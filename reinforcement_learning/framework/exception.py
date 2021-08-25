from python_helper import ReflectionHelper


class GlobalException(Exception):

    def __init__(self, message: str):
        self.message = message
        Exception.__init__(self, message)

    def __str__(self):
        return f'{self.message}.'

    def __repr__(self):
        return self.__str__()


class MethodNotImplementedException(GlobalException):

    def __init__(self, message: str = 'Method not implemented'):
        GlobalException.__init__(self, message)


class BadImplementation(GlobalException):

    def __init__(self, message: str = 'Bad implementation'):
        GlobalException.__init__(self, message)


class IncorrectTypeException(BadImplementation):

    TOKEN_GIVEN_TYPE: str = '__TOKEN_GIVEN_TYPE__'
    TOKEN_IS_TYPE: str = '__TOKEN_IS_TYPE__'
    DEFAULT_MESSAGE: str = f'Incorrect type. It is {TOKEN_IS_TYPE}, but should be {TOKEN_GIVEN_TYPE}'

    def __init__(
        self,
        instance: object,
        givenType: type,
        message: str = DEFAULT_MESSAGE
    ):
        BadImplementation.__init__(
            self,
            message.replace(
                self.TOKEN_IS_TYPE,
                ReflectionHelper.getClassName(instance)
            ).replace(
                self.TOKEN_GIVEN_TYPE,
                ReflectionHelper.getName(givenType)
            )
        )
