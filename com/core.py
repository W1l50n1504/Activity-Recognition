import numpy as np
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
        self.dsConfig = 0

        self.loadData()
        self.dataProcessing()

    def loadData(self):
        """:cvar"""

        x1, y1 = loadUCIHAR()
        x2, y2 = loadUMAFall()
        x3, y3 = loadWISDM()

        if (self.dsConfig == 0):
            self.X = pd.concat([x1, x2])
            self.y = pd.concat([y1, y2])

            x3 = np.array(x3)
            y3 = np.array(y3)

            self.X_test = x3
            self.y_test = y3

        elif (self.dsConfig == 1):
            self.X = pd.concat([x3, x2])
            self.y = pd.concat([y3, y2])

            x1 = np.array(x1)
            y1 = np.array(y1)

            self.X_test = x1
            self.y_test = y1

        elif (self.dsConfig == 2):
            self.X = pd.concat([x1, x3])
            self.y = pd.concat([y1, y3])

            x2 = np.array(x2)
            y2 = np.array(y2)

            self.X_test = x2
            self.y_test = y2

        elif (self.dsConfig == 3):
            self.X, self.y = loadData()

    def dataProcessing(self):
        """
        tldr
        """
        print('elaborazione dei dati...')

        self.X = np.array(self.X)

        if (self.dsConfig == 3):
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3,
                                                                                    random_state=42)
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train,
                                                                                  test_size=0.1, random_state=42)
        else:
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X, self.y,
                                                                                  test_size=0.1, random_state=42)

        enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
        enc = enc.fit(self.y_train)

        self.y_train = enc.transform(self.y_train)
        self.y_test = enc.transform(self.y_test)
        self.y_val = enc.transform(self.y_val)

        # print('dimensione reshape', self.X_train[..., np.newaxis].shape)
        # print('dimensione reshape', self.X_test[..., np.newaxis].shape)
        # print('dimensione reshape', self.X_val[..., np.newaxis].shape)

        if (self.dsConfig == 0):
            # valori da utilizzare se si utilizza UCIHAR e UMAFALL
            self.X_train = self.X_train.reshape(22449, 4, 1)
            self.X_test = self.X_test.reshape(1098204, 4, 1)
            self.X_val = self.X_val.reshape(2495, 4, 1)

        elif (self.dsConfig == 1):
            # UMAFALL WISDM
            self.X_train = self.X_train.reshape(1007126, 4, 1)
            self.X_test = self.X_test.reshape(4119, 4, 1)
            self.X_val = self.X_val.reshape(111903, 4, 1)

        elif (self.dsConfig == 2):
            # UCIHAR WISDM
            self.X_train = self.X_train.reshape(992090, 4, 1)
            self.X_test = self.X_test.reshape(20825, 4, 1)
            self.X_val = self.X_val.reshape(110233, 4, 1)

        elif (self.dsConfig == 3):
            # valori da utilizzare se tutti e tre i dataset sono uniti
            self.X_train = self.X_train.reshape(707582, 4, 1)
            self.X_test = self.X_test.reshape(336945, 4, 1)
            self.X_val = self.X_val.reshape(78621, 4, 1)

        print('Fine elaborazione dati.')
        self.y = np.array(self.y)

    @abstractmethod
    def modelCreation(self):
        """:cvar"""

    @abstractmethod
    def fit(self):
        ''''
        verrà implementato dai modelli
        '''

    def plot(self):
        rounded_labels = np.argmax(self.y_test, axis=1)
        y_pred = self.model.predict_classes(self.X_test)

        print('round', rounded_labels.shape)
        print('y', y_pred.shape)

        mat = confusion_matrix(rounded_labels, y_pred)
        plot_confusion_matrix(conf_mat=mat, show_normed=True, figsize=(10, 10))

        plt.figure(figsize=(10, 10))
        array = confusion_matrix(rounded_labels, y_pred)
        df_cm = pd.DataFrame(array, range(6), range(6))
        df_cm.columns = ["Walking", "W_Upstairs", "W_Downstairs", "Sitting", "Standing", "Laying"]
        df_cm.index = ["Walking", "W_Upstairs", "W_Downstairs", "Sitting", "Standing", "Laying"]
        # sn.set(font_scale=1)#for label size
        # sns.heatmap(df_cm, annot=True, annot_kws={"size": 12},
        # yticklabels=("Walking", "W_Upstairs", "W_Downstairs", "Sitting", "Standing", "Laying"),
        # xticklabels=("Walking", "W_Upstairs", "W_Downstairs", "Sitting", "Standing", "Laying"))  # font size
        plt.savefig(confusionMatrixBLSTM)
        # Plot training & validation accuracy values
        plt.figure(figsize=(15, 8))
        epoch_range = range(1, self.epochs + 1)
        plt.plot(epoch_range, self.history.history['accuracy'])
        plt.plot(epoch_range, self.history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.savefig(trainingValAccBLSTM)
        # plt.show()

        # Plot training & validation auc values
        plt.figure(figsize=(15, 8))
        epoch_range = range(1, self.epochs + 1)
        plt.plot(epoch_range, self.history.history['auc'])
        plt.plot(epoch_range, self.history.history['val_auc'])
        plt.title('Model auc')
        plt.ylabel('auc')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.savefig(TrainingValAucBLSTM)
        # plt.show()

        # Plot training & validation loss values
        plt.figure(figsize=(15, 8))
        plt.plot(epoch_range, self.history.history['loss'])
        plt.plot(epoch_range, self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.savefig(ModelLossBLSTM)
        # plt.show()

    def main(self):
        self.modelCreation()
        self.fit()
        self.plot()
