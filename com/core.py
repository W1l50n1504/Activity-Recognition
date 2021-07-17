from abc import ABCMeta, abstractmethod, ABC

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.python.keras.layers import Bidirectional, LSTM
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

from mlxtend.plotting import plot_confusion_matrix

from com.utility import *


class BaseModel(metaclass=ABCMeta):

    def __init__(self):

        self.X = None
        self.y = None

        self.checkpoint = None
        self.y_train = None
        self.y_test = None
        self.y_val = None

        self.X_train = None
        self.X_test = None
        self.X_val = None
        self.history = None
        self.loadData()
        self.dataProcessing()

    def loadData(self, flag):
        """:cvar"""
        if(flag):
            self.X, self.y = loadData()
        else:
            self.X, self.y = loadSavedData()
    @abstractmethod
    def dataProcessing(self):
        """
        tldr
        """

    @abstractmethod
    def modelCreation(self):
        """:cvar"""

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

    def main(self):
        self.modelCreation()
        self.fit()
        self.plot()
