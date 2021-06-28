from abc import ABCMeta, abstractmethod
from com.utility import *


class BaseModel(metaclass=ABCMeta):

    def __init__(self):
        self.checkpoint = None
        self.y_train = None
        self.y_test = None
        self.y_val = None

        self.X_train = None
        self.X_test = None
        self.X_val = None

        self.history = None
        self.processData()

    @abstractmethod
    def processData(self):
        """
        tldr
        """

    @abstractmethod
    def fit(self):
        ''''
        verrà implementato dai modelli
        '''

    @abstractmethod
    def plot(self):
        """

        :return:
        """
