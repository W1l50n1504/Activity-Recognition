from abc import ABCMeta, abstractmethod
from tensorflow.keras.utils import to_categorical
from utility import *

import numpy as np


class BaseModel(metaclass=ABCMeta):

    def __init__(self, X_train_signals_paths, X_test_signals_paths):
        self.X_train = load_X(X_train_signals_paths)
        self.X_test = load_X(X_test_signals_paths)

    def encode(self):
        # forse da eliminare
        self.train_y = self.train_y - 1
        self.test_y = self.test_y - 1
        # one hot encode y
        self.train_y = to_categorical(self.train_y)
        self.test_y = to_categorical(self.test_y)

    @abstractmethod
    def processData(self):
        """

        :return:
        """


    @abstractmethod
    def fit(self):
        ''''
        same thing
        '''
