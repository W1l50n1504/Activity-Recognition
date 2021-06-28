import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import tensorflow

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, Bidirectional, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from mlxtend.plotting import plot_confusion_matrix

absPath_ = os.getcwd()

checkPointPathCNN = absPath_ + 'com/checkpoint/CNN'
checkPointPathBLSTM = absPath_ + 'com/checkpoint/BLSTM'

nameXtrain = ''
nameYtrain = ''
nameZtrain = ''

nameXtest = ''
nameYtest = ''
nameZtest = ''

y_train_path = ''
y_test_path = ''

# posizione salvataggio immagini dei grafici dei modelli
# grafici BLSTM
confusionMatrixBLSTM = 'com/graphs/cnn/confusionMatrixBLSTM.png'
trainingValAccBLSTM = 'com/graphs/cnn/trainingValAccBLSTM.png'
TrainingValAucBLSTM = 'com/graphs/cnn/trainingValAucBLSTM.png'
ModelLossBLSTM = 'com/graphs/cnn/modelLossBLSTM.png'

# grafici CNN
confusionMatrixCNN = 'com/graphs/cnn/confusionMatrixCNN.png'
trainingValAccCNN = 'com/graphs/cnn/trainingValAccCNN.png'
trainingValAucCNN = 'com/graphs/cnn/trainingValAucCNN.png'
modelLossCNN = 'com/graphs/cnn/modelLossCNN.png'

# grafici HMM


labelDict = {'WALKING': 0, 'WALKING_UPSTAIRS': 1, 'WALKING_DOWNSTAIRS': 2,
             'SITTING': 3, 'STANDING': 4, 'LAYING': 5}


def norm(data):
    return (data - data.mean()) / data.std() + np.finfo(np.float32).eps


def produceMagnitude(flag, path):
    magnitude = []
    if flag:

        x = norm(load_X(path + nameXtrain))
        y = norm(load_X(path + nameYtrain))
        z = norm(load_X(path + nameZtrain))

    else:
        x = norm(load_X(path + nameXtest))
        y = norm(load_X(path + nameYtest))
        z = norm(load_X(path + nameZtest))

    for i in range(0, len(x)):
        magnitude.append(np.sqrt(x[i] ** 2 + y[i] ** 2 + z[i] ** 2))

    # print('\n', magnitude)

    return magnitude


def load_X(X_signals_paths):
    X_signals = []

    file = open(X_signals_paths, 'r')
    X_signals.append(
        [np.array(serie, dtype=np.float32) for serie in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]]
    )
    file.close()

    return np.transpose(np.array(X_signals), (1, 2, 0))


def load_y(y_path):
    file = open(y_path, 'r')
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]],
        dtype=np.int32
    )
    file.close()

    return y_ - 1


def loadData():
    print('caricamento dei dati di training e test')
    X_train = produceMagnitude(0)
    X_test = produceMagnitude(1)

    y_train = load_y(y_train_path)
    y_test = load_y(y_test_path)

    print('fine caricamento')
    # return X_train, y_train, X_test, y_test

    print('hello')


def dataProcessingHMM(X_train, y_train, X_test, y_test):
    print('elaborazione dei dati...')

    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))

    X = np.log(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    # enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    # enc = enc.fit(y_train)
    """
    y_train = enc.transform(y_train)
    y_test = enc.transform(y_test)
    y_val = enc.transform(y_val)
    X_train = X_train.reshape(6488, 128)
    X_test = X_test.reshape(3090, 128)
    X_val = X_val.reshape(721, 128)
    """
    X_train = X_train.reshape((X_train.shape[0] * X_train.shape[1]), X_train.shape[2])
    X_test = X_test.reshape((X_test.shape[0] * X_test.shape[1]), X_test.shape[2])
    X_val = X_val.reshape((X_val.shape[0] * X_val.shape[1]), X_val.shape[2])

    X_train = X_train.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)
    X_val = X_val.reshape(-1, 1)

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)

    print('fine elaborazione dati')
    return X_train, y_train, X_test, y_test, X_val, y_val


def dataProcessingBLSTM(X_train, y_train, X_test, y_test):
    print('elaborazione dei dati...')

    X = np.concatenate((X_train, X_test))

    y = np.concatenate((y_train, y_test))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    enc = enc.fit(y_train)

    y_train = enc.transform(y_train)
    y_test = enc.transform(y_test)
    y_val = enc.transform(y_val)
    print('fine elaborazione dati')

    return X_train, y_train, X_test, y_test, X_val, y_val


def plotHMM(X_train, y_train, X_test, y_test, X_val, y_val):
    # non va bene come criterio di controllo della precisione del modello,
    # e' necessario trovare un metodo migliore per valutare la precisione del sistema per compararla con gli altri approcci

    lr = loadModel()
    print('Calcolo degli score del modello...')
    train_scores = []
    test_scores = []
    val_scores = []
    X_train = X_train.reshape(1, -1).tolist()
    X_test = X_test.reshape(1, -1).tolist()
    X_val = X_val.reshape(1, -1).tolist()

    for i in range(0, len(y_train)):
        train_score = lr.score(X_train)
        train_scores.append(train_score)

    for i in range(0, len(y_test)):
        test_score = lr.score(X_test)
        test_scores.append(test_score)

    for i in range(0, len(y_val)):
        val_score = lr.score(X_val)
        val_scores.append(val_score)

    length_train = len(train_scores)
    length_val = len(val_scores) + length_train
    length_test = len(test_scores) + length_val

    plt.figure(figsize=(7, 5))
    plt.scatter(np.arange(length_train), train_scores, c='b', label='trainset')
    plt.scatter(np.arange(length_train, length_val), val_scores, c='r', label='testset - imitation')
    plt.scatter(np.arange(length_val, length_test), test_scores, c='g', label='testset - original')
    plt.title(f'User: 1 | HMM states: 6 | GMM components: 2')
    plt.legend(loc='lower right')

    plt.savefig(hmmGraph)
    plt.show()


if __name__ == '__main__':
    print('hello')
