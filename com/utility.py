import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import seaborn as sns

from sklearn.utils import resample

# absPath_ = os.getcwd()
absPath_ = 'C:/Users/david/PycharmProjects/ActivityRecognition683127/com'
# absPath_ = '/home/w1l50n/PycharmProjects/ActivityRecognition683127-main/com'
# percorso che contiene tutti i dati precaricati, in modo da evitare di dover ricalcolarli tutti ogni volta
xPath = absPath_ + '/dataset/DataProcessed/xData.csv'
yPath = absPath_ + '/dataset/DataProcessed/yData.csv'

# etichetta dataset
activity = ['Activity']

# UCI HAR dataset path
featuresPath = absPath_ + '/dataset/UCI HAR Dataset/features.txt'

xTrainPathUCI = absPath_ + '/dataset/UCI HAR Dataset/train/X_train.txt'
xTestPathUCI = absPath_ + '/dataset/UCI HAR Dataset/test/X_test.txt'

yTrainPathUCI = absPath_ + '/dataset/UCI HAR Dataset/train/y_train.txt'
yTestPathUCI = absPath_ + '/dataset/UCI HAR Dataset/test/y_test.txt'

xaccFile = absPath_ + '/dataset/UCI HAR Dataset/test/Inertial Signals/body_acc_x_test.txt'
yaccFile = absPath_ + '/dataset/UCI HAR Dataset/test/Inertial Signals/body_acc_y_test.txt'
zaccFile = absPath_ + '/dataset/UCI HAR Dataset/test/Inertial Signals/body_acc_z_test.txt'

# etichette dell'UCIHAR utili con mean si indica il valore medio della misurazione
xUCIacc = 'tBodyAcc-mean()-X'
yUCIacc = 'tBodyAcc-mean()-Y'
zUCIacc = 'tBodyAcc-mean()-Z'
magUCIacc = 'magnitudeAcc'

xUCIgyro = 'tBodyGyro-mean()-X'
yUCIgyro = 'tBodyGyro-mean()-Y'
zUCIgyro = 'tBodyGyro-mean()-Z'
magUCIgyro = 'magnitudeAcc'
# WISDM dataset path

wisdmPath = absPath_ + '/dataset/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt'

xWISDM = 'x-accel'
yWISDM = 'y-accel'
zWISDM = 'z-accel'
magWISDM = 'magnitude'

# UMAFALL dataset path, con etichette dei dataset

umafallPath = absPath_ + '/dataset/UMAFall_Dataset'

# etichette per i dataset che carico
columnsUMAFALL = ['TimeStamp', 'Sample No', 'X - Axis', 'Y - Axis', 'Z - Axis', 'Sensor Type', 'Sensor ID']

xacc = 'x-accel-acc'
yacc = 'y-accel-acc'
zacc = 'z-accel-acc'
magacc = 'magnitude-acc'
std = 'standard-deviation'

finalColumns = [xacc, yacc, zacc, magacc, std]
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
confusionMatrixBLSTM = absPath_ + '/graphs/blstm/heatMapBLSTM.png'
trainingValAccBLSTM = absPath_ + '/graphs/blstm/trainingValAccBLSTM.png'
TrainingValAucBLSTM = absPath_ + '/graphs/blstm/trainingValAucBLSTM.png'
ModelLossBLSTM = absPath_ + '/graphs/blstm/modelLossBLSTM.png'

# dizionari riguardanti le attività registrate dai dataset
labelDictUCI = {'WALKING': 0, 'WALKING_UPSTAIRS': 1, 'WALKING_DOWNSTAIRS': 2,
                'SITTING': 3, 'STANDING': 4, 'LAYING': 5}

labelDictWISDM = {'Walking': 0, 'Upstairs': 1, 'Downstairs': 2, 'Sitting': 3, 'Standing': 4, 'Jogging': 6}

labelDictUMAFALL = {'Walking': 0, 'Laying': 5, 'Jogging': 6, 'Falling': 7}


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


def reduceSample(Xdf, yDf):
    # riduce la frequenza dei campioni da 50 Hz a 20Hz, da utilizzare con UCIHAR
    reduce = pd.DataFrame(columns=[xacc, yacc, zacc, magacc, 'label'])

    reduce[xacc] = Xdf[xacc]
    reduce[yacc] = Xdf[yacc]
    reduce[zacc] = Xdf[zacc]
    reduce[magacc] = Xdf[magacc]
    reduce[std] = Xdf[std]

    reduce['label'] = yDf['Activity']

    reduce = resample(reduce, replace=True, n_samples=int((len(reduce) * 20 / 50)), random_state=0)
    reduce = reduce.reset_index(drop=True)

    finalX = pd.DataFrame(columns=finalColumns)
    finalY = pd.DataFrame(columns=['Activity'])

    finalX[xacc] = reduce[xacc]
    finalX[yacc] = reduce[yacc]
    finalX[zacc] = reduce[zacc]
    finalX[magacc] = reduce[magacc]
    finalX[std] = reduce[std]

    finalY['Activity'] = reduce['label']

    return finalX.copy(), finalY.copy()


def loadNmerge(X_df, Y_df, path, label):
    # Funzione che carica i dati contenuti nei file del dataset UMAFALL ne carica i dati selezionando solo le feature utili
    # e li concatena nel dataset finale di UMAFALL

    df = pd.read_csv(umafallPath + path, header=None, names=columnsUMAFALL, sep=';')

    # ho preso solo le misurazioni dell'accelerometro e della posizione che mi interessa prendere anche il giroscopio 1
    # TODO bisogna creare due dataset uno che contiene i dati relativi all'accelerometro e l'altro riguardante il giroscopio
    df = df.loc[(df['Sensor Type'] == 0) & (df['Sensor ID'] == 0)]
    finalDf = pd.DataFrame(columns=finalColumns)

    finalDf[xacc] = df['X - Axis']
    finalDf[yacc] = df['Y - Axis']
    finalDf[zacc] = df['Z - Axis']
    finalDf[magacc] = np.sqrt((finalDf[xacc] ** 2) + (finalDf[yacc] ** 2) + (finalDf[zacc] ** 2))

    X_df = pd.concat([X_df, finalDf])
    X_df = X_df.reset_index(drop=True)

    length = len(X_df)

    for i in range(len(Y_df), length):
        Y_df.append(labelDictUMAFALL[label])

    return X_df, Y_df


def loadUCIHAR():
    # copia ed elaborazione dei dati contenuti nell'UCIHAR
    # UCI HAR Dataset caricato correttamente con il nome di ogni feature
    # restituisce due dataset
    # FUNZIONA

    X, Y = get_human_dataset()

    X_df = pd.DataFrame(columns=finalColumns, dtype='float32')
    Y_df = pd.DataFrame(columns=activity, dtype='int32')

    X_df[xacc] = X[xUCIacc]
    X_df[yacc] = X[yUCIacc]
    X_df[zacc] = X[zUCIacc]
    X_df[magacc] = np.sqrt((X[xUCIacc] ** 2) + (X[yUCIacc] ** 2) + (X[zUCIacc] ** 2))
    X_df[std] = X_df.std(axis=1, skipna=True)

    Y_df['Activity'] = Y['action']

    X_df = X_df.reset_index(drop=True)
    Y_df = Y_df.reset_index(drop=True)
    X_df, Y_df = reduceSample(X_df, Y_df)

    return X_df.copy(), Y_df.copy()


def loadUMAFall():
    # carica i dati contenuti nei vari file del dataset (e' stata fatta una selezione dei file) e dovrebbe restituire due
    # % Accelerometer = 0 sensor type da utilizzare
    # % Gyroscope = 1 sensor type da utilizzare

    Y_label = []

    X_df = pd.DataFrame(columns=finalColumns, dtype='float32')

    # caricato il dataset levando ; che univa tutte le colonne
    X_df, Y_label = loadNmerge(X_df, Y_label, '/UMAFall_Subject_01_ADL_Walking_1_2017-04-14_23-25-52.csv',
                               'Walking')

    X_df, Y_label = loadNmerge(X_df, Y_label, '/UMAFall_Subject_02_ADL_Jogging_1_2016-06-13_20-40-29.csv',
                               'Jogging')
    X_df, Y_label = loadNmerge(X_df, Y_label,
                               '/UMAFall_Subject_02_ADL_LyingDown_OnABed_1_2016-06-13_20-32-16.csv',
                               'Laying')

    X_df, Y_label = loadNmerge(X_df, Y_label,
                               '/UMAFall_Subject_02_Fall_backwardFall_1_2016-06-13_20-51-32.csv',
                               'Falling')

    X_df, Y_label = loadNmerge(X_df, Y_label,
                               '/UMAFall_Subject_02_Fall_forwardFall_1_2016-06-13_20-43-52.csv',
                               'Falling')

    X_df, Y_label = loadNmerge(X_df, Y_label,
                               '/UMAFall_Subject_02_Fall_lateralFall_1_2016-06-13_20-49-17.csv',
                               'Falling')

    X_df[std] = X_df.std(axis=1, skipna=True)
    Y_df = pd.DataFrame(Y_label, columns=activity, dtype='int32')

    return X_df.copy(), Y_df.copy()


def loadWISDM():
    # carica il dataset WISDM e ne estrapola le etichette delle attivita' convertendole in numeri,
    # estrae le misurazioni lungo i tre assi e ne calcola la magnitude il tutto all'interno di due
    # dataset
    # restituisce un dataset e una lista
    columns = ['user', 'activity', 'timestamp', 'x-accel', 'y-accel', 'z-accel']

    X_df = pd.DataFrame(columns=finalColumns, dtype='float32')
    Y_df = pd.DataFrame(columns=activity, dtype='int32')

    df = pd.read_csv(wisdmPath, header=None, names=columns)

    X_df[xacc] = df[xWISDM]
    X_df[yacc] = df[yWISDM]
    # pulizia dei dati caricati, assieme ai numeri viene caricato anche il simbolo ;
    # e quindi non viene riconosciuto come un valore numerico, in questa maniera lo si rimuove
    X_df[zacc] = df[zWISDM].str.replace(';', '').astype(float)

    # calcolo della magnitudine
    X_df[magacc] = np.sqrt((X_df[xWISDM] ** 2) + (X_df[yWISDM] ** 2) + (X_df[zWISDM] ** 2))
    X_df[std] = X_df.std(axis=1, skipna=True)

    # rimpiazza le stringhe che indicano le attivita' con dei valori numerici
    df = df.replace('Walking', labelDictWISDM['Walking'], regex=True)
    df = df.replace('Upstairs', labelDictWISDM['Upstairs'], regex=True)
    df = df.replace('Downstairs', labelDictWISDM['Downstairs'], regex=True)
    df = df.replace('Sitting', labelDictWISDM['Sitting'], regex=True)
    df = df.replace('Standing', labelDictWISDM['Standing'], regex=True)
    df = df.replace('Jogging', labelDictWISDM['Jogging'], regex=True)

    Y_df['Activity'] = df['activity']
    Y_df = Y_df.astype('int64')

    return X_df.copy(), Y_df.copy()


def loadData():
    # obiettivi, caricare i tre dataset,downsampling di ucihar da 50Hz a 20Hz, rimappare la label
    # per unificarle e avere tutte le attivita' in sincrono

    print('Inizio caricamento dataset...')

    # carico UCIHAR
    XDataUCI, yDataUCI = loadUCIHAR()

    # carico UMAFall
    XDataUMAFall, yDataUMAFall = loadUMAFall()

    # carico WISDM
    XdataWISDM, yDataWISDM = loadWISDM()

    # unione dei tre dataset

    X_df = pd.concat([XDataUCI, XdataWISDM, XDataUMAFall])
    X_df = X_df.reset_index(drop=True)

    y_df = pd.concat([yDataUCI, yDataWISDM, yDataUMAFall])
    y_df = y_df.reset_index(drop=True)

    # X_df = torch.tensor(X_df.values)

    return X_df, y_df


def saveData(X, Y):
    print('Salvataggio dati in DataProcessed...')
    X.to_csv(xPath, index=False)
    Y.to_csv(yPath, index=False)


def loadSavedData():
    print('Caricamento dati da DataProcessed...')
    x = pd.read_csv(xPath)
    y = pd.read_csv(yPath)

    # x = torch.tensor(x.values)

    return x, y


if __name__ == '__main__':
    # caricamento e concatenazione dei vari dataset eseguita con successo, inserire 0,1 o 2 come argument di loadData per otterenere una diversa combinazione di dataset
    # XData, YData, x_val, y_val = loadData(0)

    # print(XData)
    # print('\n')
    # print(len(YData))
    # print('\n')
    # print(x_val)
    # print('\n')
    # print(len(y_val))
    x, y = loadUCIHAR()

    print(x)
