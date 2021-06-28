from com.core import *
from com.utility import *

from abc import ABC
from abc import ABCMeta, abstractmethod


class CNN(BaseModel, ABC):
    def __init__(self):
        super.__init__()
