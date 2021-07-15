import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

from sklearn.utils import resample

# absPath_ = os.getcwd()
absPath_ = 'C:/Users/david/PycharmProjects/ActivityRecognition683127/com'

# etichetta dataset
activity = ['Activity']

# UCI HAR dataset path
featuresPath = absPath_ + '/dataset/UCI HAR Dataset/features.txt'

xTrainPathUCI = absPath_ + '/dataset/UCI HAR Dataset/train/X_train.txt'
xTestPathUCI = absPath_ + '/dataset/UCI HAR Dataset/test/X_test.txt'

yTrainPathUCI = absPath_ + '/dataset/UCI HAR Dataset/train/y_train.txt'
yTestPathUCI = absPath_ + '/dataset/UCI HAR Dataset/test/y_test.txt'

xacc = absPath_ + '/dataset/UCI HAR Dataset/test/Inertial Signals/body_acc_x_test.txt'
yacc = absPath_ + '/dataset/UCI HAR Dataset/test/Inertial Signals/body_acc_y_test.txt'
zacc = absPath_ + '/dataset/UCI HAR Dataset/test/Inertial Signals/body_acc_z_test.txt'

# etichette dell'UCIHAR utili
xUCI = 'tBodyAcc-mean()-X'
yUCI = 'tBodyAcc-mean()-Y'
zUCI = 'tBodyAcc-mean()-Z'
magUCI = 'magnitude'

# WISDM dataset path

wisdmPath = absPath_ + '/dataset/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt'

xWISDM = 'x-accel'
yWISDM = 'y-accel'
zWISDM = 'z-accel'
magWISDM = 'magnitude'

# UMAFALL dataset path, con etichette dei dataset

X_trainUMAFall = absPath_ + '/dataset/UMAFall_Dataset'

# etichette per i dataset che carico

x = 'x-accel'
y = 'y-accel'
z = 'z-accel'
mag = 'magnitude'

finalColumns = [x, y, z, mag]
# posizione di salvataggio checkpoint dei modelli
checkPointPathCNN = absPath_ + '/checkpoint/CNN'
checkPointPathBLSTM = absPath_ + '/checkpoint/BLSTM'
checkPointPathHMM = absPath_ + '/checkpoint/HMM'

# posizione salvataggio immagini dei grafici dei modelli

# grafici CNN
confusionMatrixCNN = absPath_ + '/graphs/cnn/confusionMatrixCNN.png'
trainingValAccCNN = absPath_ + '/graphs/cnn/trainingValAccCNN.png'
trainingValAucCNN = absPath_ + '/graphs/cnn/trainingValAucCNN.png'
modelLossCNN = absPath_ + '/graphs/cnn/modelLossCNN.png'

# grafici BLSTM
confusionMatrixBLSTM = absPath_ + '/graphs/cnn/heatMapBLSTM.png'
trainingValAccBLSTM = absPath_ + '/graphs/cnn/trainingValAccBLSTM.png'
TrainingValAucBLSTM = absPath_ + '/graphs/cnn/trainingValAucBLSTM.png'
ModelLossBLSTM = absPath_ + '/graphs/cnn/modelLossBLSTM.png'

# dizionari riguardanti le attività registrate dai dataset
labelDictUCI = {'WALKING': 0, 'WALKING_UPSTAIRS': 1, 'WALKING_DOWNSTAIRS': 2,
                'SITTING': 3, 'STANDING': 4, 'LAYING': 5}

labelDictWISDM = {'Walking': 0, 'Upstairs': 1, 'Downstairs': 2, 'Sitting': 3, 'Standing': 4, 'Jogging': 6}

labelDictUMAFALL = {}


def norm(data):
    # da eliminare
    return (data - data.mean()) / data.std() + np.finfo(np.float32).eps


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

    X = pd.concat([X_train, X_test])
    y = pd.concat([y_train, y_test])

    return X, y


def reduceSample(X_train, y_train, X_test, y_test):
    # riduce la frequenza dei campioni da 50 Hz a 20Hz
    XTrainReduced = resample(X_train, replace=True, n_samples=int((len(X_train) * 20) / 50))
    yTrainReduced = resample(y_train, replace=True, n_samples=int((len(y_train) * 20) / 50))
    XTestReduced = resample(X_test, replace=True, n_samples=int((len(X_test) * 20) / 50))
    yTestReduced = resample(y_test, replace=True, n_samples=int((len(y_test) * 20) / 50))

    return XTrainReduced, yTrainReduced, XTestReduced, yTestReduced


def loadUCIHAR():
    # copia ed elaborazione dei dati contenuti nell'UCIHAR
    # UCI HAR Dataset caricato correttamente con il nome di ogni feature
    # restituisce due dataset

    X, Y = get_human_dataset()

    X_df = pd.DataFrame(columns=finalColumns, dtype='float64')
    Y_df = pd.DataFrame(columns=activity, dtype='int32')

    X_df[x] = X[xUCI]
    X_df[y] = X[yUCI]
    X_df[z] = X[zUCI]

    X_df[mag] = np.sqrt((X[xUCI] ** 2) + (X[yUCI] ** 2) + (X[zUCI] ** 2))

    X_df = X_df.reset_index(drop=True)
    Y_df = Y.reset_index(drop=True)

    return X_df, Y_df


def loadUMAFall():
    # carica i dati contenuti nei vari file del dataset (e' stata fatta una selezione dei file) e dovrebbe restituire due dataset
    columns = ['TimeStamp', 'Sample No', 'X - Axis', 'Y - Axis', 'Z - Axis', 'Sensor Type', 'Sensor ID']

    X_df = pd.DataFrame(columns=finalColumns)

    return X_df, Y_df


def loadWISDM():
    # carica il dataset WISDM e ne estrapola le etichette delle attivita' convertendole in numeri,
    # estrae le misurazioni lungo i tre assi e ne calcola la magnitude il tutto all'interno di due
    # dataset
    # restituisce un dataset e una lista

    y_labelConverted = []
    columns = ['user', 'activity', 'timestamp', 'x-accel', 'y-accel', 'z-accel']
    df = pd.read_csv(wisdmPath, header=None, names=columns)

    # estrazione delle etichette delle attivita' presenti nel dataset e conversione utilizzando il dizionario
    y_label = df['activity'].copy()
    y_labelList = y_label.tolist()

    # traduzione delle etichette testuali in numeri int
    for i in range(0, len(y_labelList)):
        y_labelConverted.append(labelDictWISDM[y_labelList[i]])

    X_df = pd.DataFrame(columns=finalColumns, dtype='float64')

    X_df[xWISDM] = df[xWISDM]
    X_df[yWISDM] = df[yWISDM]
    # pulizia dei dati caricati, assieme ai numeri viene caricato anche il simbolo ;
    # e quindi non viene riconosciuto come un valore numerico, in questa maniera lo si rimuove
    X_df[zWISDM] = df[zWISDM].str.replace(';', '').astype(float)

    # calcolo della magnitudine
    X_df[magWISDM] = np.sqrt((X_df[xWISDM] ** 2) + (X_df[yWISDM] ** 2) + (X_df[zWISDM] ** 2))

    return X_df.copy(), y_labelConverted


def loadData():
    # obiettivi, caricare i tre dataset,downsampling di ucihar da 50Hz a 20Hz, rimappare la label
    # per unificarle e avere tutte le attivita' in sincrono

    print('Inizio caricamento dataset...')

    # carico UCIHAR
    XDataUCI, yDataUCI = loadUCIHAR()

    # carico UMAFall
    X_trainUMAFallArray, y_trainUMAFallArray, X_testUMAFallArray, y_testUMAFallArray = loadUMAFall()

    # carico WISDM
    # restituisce lo stesso numero di elementi in entrambe le parti
    yDataWISDM, XdataWISDM = loadWISDM()

    # unione dei tre dataset

    flag = 0

    if (flag == 0):
        X_train = np.concatenate((X_trainUCIArray, X_trainWISDMArray))
        y_train = np.concatenate((y_trainUCIArray, y_trainWISDMArray))

        X_test = np.concatenate((X_testUCIArray, X_testWISDMArray))
        y_test = np.concatenate((y_testUCIArray, y_testWISDMArray))

        X_val = X_trainUMAFallArray
        y_val = y_trainUMAFallArray

    elif (flag == 1):
        X_train = np.concatenate((X_trainWISDMArray, X_trainUMAFallArray))
        y_train = np.concatenate((y_trainWISDMArray, y_trainUMAFallArray))

        X_test = np.concatenate((X_testWISDMArray, X_testUMAFallArray))
        y_test = np.concatenate((y_testWISDMArray, y_testUMAFallArray))

        X_val = X_trainUCIArray
        y_val = y_trainUCIArray

    elif (flag == 2):
        X_train = np.concatenate((X_trainUCIArray, X_trainUMAFallArray))
        y_train = np.concatenate((y_trainUCIArray, y_trainUMAFallArray))

        X_test = np.concatenate((X_testUCIArray, X_testUMAFallArray))
        y_test = np.concatenate((y_testUCIArray, y_testUMAFallArray))

        X_val = X_trainWISDMArray
        y_val = y_trainWISDMArray

    return X_train, y_train, X_test, y_test, X_val, y_val


if __name__ == '__main__':
    # X_train, y_train, X_test, y_test = loadData()

    # X_trainUMAFallArray, y_trainUMAFallArray, X_testUMAFallArray, y_testUMAFallArray = loadUMAFall()

    # restituisce lo stesso numero di elementi in entrambe le parti
    # y_label, XtrainWISDM = loadWISDM()

    # print(y_label, XtrainWISDM)
    # print(len(y_label))

    x, y = loadWISDM()

    print(x)
