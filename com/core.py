from abc import ABCMeta, abstractmethod
from tensorflow.keras.utils import to_categorical
from utility import *

import numpy as np


class BaseModel(metaclass=ABCMeta):

    def __init__(self):
        X_train, y_train, X_test, y_test = loadData()
        self.processData(X_train, y_train, X_test, y_test)

    @abstractmethod
    def processData(self):
        """

        :return:
        """
        self.X_train, self.y_train, self.X_test, self.y_test, self.X_val, self.y_val


    @abstractmethod
    def fit(self):
        ''''
        same thing
        '''
