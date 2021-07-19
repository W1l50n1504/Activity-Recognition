import tensorflow as tf

from abc import ABCMeta, abstractmethod, ABC
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.python.keras.layers import Bidirectional, LSTM, MaxPooling1D
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv2D, Conv1D, MaxPool1D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

from com.utility import *

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


class BaseModel(metaclass=ABCMeta):

    def __init__(self):
        self.X = None
        self.y = None
        self.model = None
        self.checkpoint = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X_val = None
        self.y_val = None
        self.history = None
        self.epochs = 10

        self.loadData()
        self.dataProcessing()

    def loadData(self):
        """:cvar"""
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
