import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from abc import ABCMeta, abstractmethod, ABC
# from dbn_libraries.tensorflow import SupervisedDBNClassification
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

# import required module
from playsound import playsound

# in f1score cambia il numero di classes se cambi il numero di dataset che carichi, se si tratta solo di UCIHAR
# metti 6, 7 negli altri casi

METRICS = [
    tf.keras.metrics.Accuracy(name="Accuracy"),
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall"),
    tfa.metrics.F1Score(num_classes=6, threshold=0.5),
    tf.keras.metrics.AUC(name="auc")
    # tf.keras.metrics.BinaryAccuracy(name="binaryAcc"),

]

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


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
        self.epochs = 30
        self.dsConfig = 4

        self.loadData()
        self.dataProcessing()

    def loadData(self):
        """:cvar"""

        x1, y1 = loadUCIHAR()
        x2, y2 = loadKUHAR()
        x3, y3 = loadMotionSense()

        if self.dsConfig == 0:
            # train uci + ku test motion
            x3 = np.array(x3)
            y3 = np.array(y3)

            self.X = pd.concat([x1, x2])
            self.y = pd.concat([y1, y2])
            self.X_test = x3
            self.y_test = y3

        elif self.dsConfig == 1:
            # train ku + motion test uci
            x1 = np.array(x1)
            y1 = np.array(y1)

            self.X = pd.concat([x3, x2])
            self.y = pd.concat([y3, y2])
            self.X_test = x1
            self.y_test = y1

        elif self.dsConfig == 2:
            # train uci + motion test ku
            y2 = np.array(y2)
            x2 = np.array(x2)

            self.X = pd.concat([x1, x3])
            self.y = pd.concat([y1, y3])
            self.X_test = x2
            self.y_test = y2

        elif self.dsConfig == 3:
            self.X, self.y = loadData()

        elif self.dsConfig == 4:
            self.X, self.y = loadKUHAR()

    def dataProcessing(self):
        """
        tldr
        """
        print('elaborazione dei dati...')

        self.X = np.array(self.X)

        if self.dsConfig == 3 or self.dsConfig == 4:
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
        """
        self.y_train = np.array(self.y_train).reshape(-1, 1)
        self.y_test = np.array(self.y_test).reshape(-1, 1)
        self.y_val = np.array(self.y_val).reshape(-1, 1)
        """
        print('dimensione reshape', self.X_train[..., np.newaxis].shape)
        print('dimensione reshape', self.X_test[..., np.newaxis].shape)
        print('dimensione reshape', self.X_val[..., np.newaxis].shape)

        if self.dsConfig == 0:
            # valori da utilizzare se si utilizza UCIHAR e KUHAR
            self.X_train = self.X_train.reshape(11991, 12, 1)
            self.X_test = self.X_test.reshape(5414, 12, 1)
            self.X_val = self.X_val.reshape(1333, 12, 1)

        elif self.dsConfig == 1:
            # MotionSense KUHAR
            self.X_train = self.X_train.reshape(20590, 12, 1)
            self.X_test = self.X_test.reshape(10299, 12, 1)
            self.X_val = self.X_val.reshape(2288, 12, 1)

        elif self.dsConfig == 2:
            # UCIHAR KUHAR
            self.X_train = self.X_train.reshape(20781, 12, 1)
            self.X_test = self.X_test.reshape(10087, 12, 1)
            self.X_val = self.X_val.reshape(2309, 12, 1)

        elif self.dsConfig == 3:
            # valori da utilizzare se tutti e tre i dataset sono uniti
            self.X_train = self.X_train.reshape(20900, 12, 1)
            self.X_test = self.X_test.reshape(9954, 12, 1)
            self.X_val = self.X_val.reshape(2323, 12, 1)

        elif self.dsConfig == 4:

            # UCIHAR y_train.shape = 6
            self.X_train = self.X_train.reshape(6488, 12, 1)
            self.X_test = self.X_test.reshape(3090, 12, 1)
            self.X_val = self.X_val.reshape(721, 12, 1)
            """
     
            
            # MotionSense y_train.shape = 5
            self.X_train = self.X_train.reshape(7423, 12, 1)
            self.X_test = self.X_test.reshape(3536, 12, 1)
            self.X_val = self.X_val.reshape(825, 12, 1)
                        # KUHAR y_train.shape = 5
            self.X_train = self.X_train.reshape(5207, 12, 1)
            self.X_test = self.X_test.reshape(2480, 12, 1)
            self.X_val = self.X_val.reshape(579, 12, 1)
            """

        print('Fine elaborazione dati.')
        self.y = np.array(self.y)

    @abstractmethod
    def modelCreation(self):
        """:cvar"""
        pass

    @abstractmethod
    def fit(self):
        """
        verrà implementato dai modelli
        """
        pass

    def plot(self):

        rounded_labels = np.argmax(self.y_test, axis=1)
        y_pred = self.model.predict_classes(self.X_test)

        mat = confusion_matrix(rounded_labels, y_pred)
        plot_confusion_matrix(conf_mat=mat, show_normed=True, figsize=(10, 10))

        plt.figure(figsize=(10, 10))
        array = confusion_matrix(rounded_labels, y_pred)

        mat = confusion_matrix(rounded_labels, y_pred)
        plot_confusion_matrix(conf_mat=mat, show_normed=True, figsize=(10, 10))

        plt.figure(figsize=(10, 10))
        array = confusion_matrix(rounded_labels, y_pred)

        if self.dsConfig == 4:

            df_cm = pd.DataFrame(array, range(6), range(6))
            df_cm.columns = ["Walking", "W_Upstairs", "W_Downstairs", "Sitting", "Standing", "Laying"]
            df_cm.index = ["Walking", "W_Upstairs", "W_Downstairs", "Sitting", "Standing", "Laying"]

            sns.set(font_scale=1)  # for label size
            sns.heatmap(df_cm, annot=True, annot_kws={"size": 12},
                        yticklabels=("Walking", "W_Upstairs", "W_Downstairs", "Sitting", "Standing", "Laying"),
                        xticklabels=("Walking", "W_Upstairs", "W_Downstairs", "Sitting", "Standing", "Laying"))

        else:
            df_cm = pd.DataFrame(array, range(7), range(7))
            df_cm.columns = ["Walking", "W_Upstairs", "W_Downstairs", "Sitting", "Standing", "Laying", "Jogging"]
            df_cm.index = ["Walking", "W_Upstairs", "W_Downstairs", "Sitting", "Standing", "Laying", "Jogging"]

            sns.set(font_scale=1)  # for label size
            sns.heatmap(df_cm, annot=True, annot_kws={"size": 12},
                        yticklabels=(
                        "Walking", "W_Upstairs", "W_Downstairs", "Sitting", "Standing", "Laying", "Jogging"),
                        xticklabels=(
                        "Walking", "W_Upstairs", "W_Downstairs", "Sitting", "Standing", "Laying", "Jogging"))

        plt.show()

        # Plot training & validation accuracy values
        plt.figure(figsize=(15, 8))
        epoch_range = range(1, self.epochs + 1)
        plt.plot(epoch_range, self.history.history['Accuracy'])
        plt.plot(epoch_range, self.history.history['val_Accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        # plt.savefig(trainingValAccBLSTM)
        plt.show()

        # Plot training & validation auc values
        plt.figure(figsize=(15, 8))
        epoch_range = range(1, self.epochs + 1)
        plt.plot(epoch_range, self.history.history['auc'])
        plt.plot(epoch_range, self.history.history['val_auc'])
        plt.title('Model auc')
        plt.ylabel('auc')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        # plt.savefig(TrainingValAucBLSTM)
        plt.show()

        # Plot training & validation loss values
        plt.figure(figsize=(15, 8))
        plt.plot(epoch_range, self.history.history['loss'])
        plt.plot(epoch_range, self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        # plt.savefig(ModelLossBLSTM)
        plt.show()

    def main(self):
        self.modelCreation()
        self.fit()
        self.plot()
