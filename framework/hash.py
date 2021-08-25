from python_helper import Constant as c
from python_helper import StringHelper


def get(value: object):
    return StringHelper.join(
        str(value).strip().replace(c.NEW_LINE, c.BLANK).split(),
        character=c.BLANK
    )
