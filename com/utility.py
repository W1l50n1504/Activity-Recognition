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

#absPath_ = os.getcwd()
absPath_ = 'C:/Users/david/PycharmProjects/ActivityRecognition683127/com'

# UCI HAR dataset path
featuresPath = absPath_ + '/dataset/UCI HAR Dataset/features.txt'
xTrainPathUCI = absPath_ + '/dataset/UCI HAR Dataset/train/X_train.txt'
yTrainPathUCI = absPath_ + '/dataset/UCI HAR Dataset/train/y_train.txt'
xTestPathUCI = absPath_ + '/dataset/UCI HAR Dataset/test/X_test.txt'
yTestPathUCI = absPath_ + '/dataset/UCI HAR Dataset/test/y_test.txt'
xacc = absPath_ + '/dataset/UCI HAR Dataset/test/Inertial Signals/body_acc_x_test.txt'
yacc = absPath_ + '/dataset/UCI HAR Dataset/test/Inertial Signals/body_acc_y_test.txt'
zacc = absPath_ + '/dataset/UCI HAR Dataset/test/Inertial Signals/body_acc_z_test.txt'
x = 'tBodyAcc-mean()-X'
y = 'tBodyAcc-mean()-Y'
z = 'tBodyAcc-mean()-Z'

# posizione di salvataggio checkpoint dei modelli
checkPointPathCNN = absPath_ + '/checkpoint/CNN'
checkPointPathBLSTM = absPath_ + '/checkpoint/BLSTM'
checkPointPathHMM = absPath_ + '/checkpoint/HMM'

# posizione salvataggio immagini dei grafici dei modelli
# grafici BLSTM
confusionMatrixBLSTM = absPath_ + '/graphs/cnn/confusionMatrixBLSTM.png'
trainingValAccBLSTM = absPath_ + '/graphs/cnn/trainingValAccBLSTM.png'
TrainingValAucBLSTM = absPath_ + '/graphs/cnn/trainingValAucBLSTM.png'
ModelLossBLSTM = absPath_ + '/graphs/cnn/modelLossBLSTM.png'

# grafici CNN
confusionMatrixCNN = absPath_ + '/graphs/cnn/confusionMatrixCNN.png'
trainingValAccCNN = absPath_ + '/graphs/cnn/trainingValAccCNN.png'
trainingValAucCNN = absPath_ + '/graphs/cnn/trainingValAucCNN.png'
modelLossCNN = absPath_ + 'graphs/cnn/modelLossCNN.png'

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

    for signal_type_path in X_signals_paths:
        file = open(signal_type_path, 'r')
        # Read dataset from disk, dealing with text files' syntax
        X_signals.append(
            [np.array(serie, dtype=np.float32) for serie in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]]
        )
        file.close()

    return np.transpose(np.array(X_signals), (1, 2, 0))


def load_X1(X_signals_paths):
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


"""

def loadData():
    print('caricamento dei dati di training e test')
    X_train = produceMagnitude(0)
    X_test = produceMagnitude(1)

    y_train = load_y(y_train_path)
    y_test = load_y(y_test_path)

    print('fine caricamento')
    return X_train, y_train, X_test, y_test"""


def loadData():
    # obiettivi, caricare i tre dataset,downsampling di ucihar da 50Hz a 20Hz, rimappare la label
    # per unificarle e avere tutte le attivita' in sincrono

    print('Inizio caricamento dataset...')

    # UCI HAR Dataset caricato correttamente con il nome di ogni feature
    feature_name_df = pd.read_csv(featuresPath, sep='\s+', header=None, names=['column_index', 'column_name'])

    feature_name = feature_name_df.iloc[:, 1].values.tolist()
    feature_dup_df = feature_name_df.groupby('column_name').count()
    feature_dup_df[feature_dup_df['column_index'] > 1].head()
    X_trainUCI, X_testUCI, y_trainUCI, y_testUCI = get_human_dataset()

    print(X_trainUCI.shape)

    # per il downsampling setta ogni riga a 50hz
    # X_trainUCI.index = pd.to_datetime(X_trainUCI.index, unit='s')
    # X_trainUCI.resample('20T')

    X_train = np.array(np.sqrt((X_trainUCI[x] ** 2) + (X_trainUCI[y] ** 2) + (X_trainUCI[z] ** 2)))
    X_train = X_train.reshape(-1, 1)

    X_test = np.array(np.sqrt((X_testUCI[x] ** 2) + (X_testUCI[y] ** 2) + (X_testUCI[z] ** 2)))
    X_test = X_test.reshape(-1, 1)

    y_train = np.array(y_trainUCI)
    y_test = np.array(y_testUCI)

    # copia ed elaborazione dei dati contenuti nell'UCIHAR

    return X_train, y_train, X_test, y_test


def get_new_feature_name_df(old_feature_name_df):
    feature_dup_df = pd.DataFrame(data=old_feature_name_df.groupby('column_name').cumcount(),
                                  columns=['dup_cnt'])
    feature_dup_df = feature_dup_df.reset_index()
    new_feature_name_df = pd.merge(old_feature_name_df.reset_index(),
                                   feature_dup_df,
                                   how='outer')
    new_feature_name_df['column_name'] = new_feature_name_df[['column_name', 'dup_cnt']].apply(
        lambda x: x[0] + '_' + str(x[1]) if x[1] > 0 else x[0], axis=1)
    new_feature_name_df = new_feature_name_df.drop(['index'], axis=1)
    return new_feature_name_df


def get_human_dataset():
    feature_name_df = pd.read_csv(featuresPath,
                                  sep='\s+',
                                  header=None,
                                  names=['column_index', 'column_name'])

    new_feature_name_df = get_new_feature_name_df(feature_name_df)

    feature_name = new_feature_name_df.iloc[:, 1].values.tolist()

    X_train = pd.read_csv(xTrainPathUCI, sep='\s+', names=feature_name)
    X_test = pd.read_csv(xTestPathUCI, sep='\s+', names=feature_name)

    y_train = pd.read_csv(yTrainPathUCI, sep='\s+', header=None, names=['action'])
    y_test = pd.read_csv(yTestPathUCI, sep='\s+', header=None, names=['action'])

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    print(absPath_)