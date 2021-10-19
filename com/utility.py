import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
# import seaborn as sns

from sklearn.utils import resample

# absPath_ = os.getcwd()
absPath_ = 'C:/Users/david/PycharmProjects/Activity-Recognition/com'
# absPath_ = '/home/w1l50n/PycharmProjects/Activity-Recognition/com'

# percorso che contiene tutti i KUHARIntero precaricati, in modo da evitare di dover ricalcolarli tutti ogni volta
xPath = absPath_ + '/dataset/ShieldApp/xData.csv'
yPath = absPath_ + '/dataset/ShieldApp/yData.csv'

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
# magUCIgyro = 'magnitudeGyro'

# MotionSense dataset data & labels

motionPath = absPath_ + '/dataset/MotionSense/'

dws = 'dws/'
jog = 'jog/'
sitt = 'sit/'
upss = 'ups/'
wlk = 'wlk/'
stdd = 'std/'

activityListMotionSense = [dws, sitt, upss, wlk, stdd]

xMSacc = 'userAcceleration.x'
yMSacc = 'userAcceleration.y'
zMSacc = 'userAcceleration.z'
xMSgyro = 'rotationRate.x'
yMSgyro = 'rotationRate.y'
zMSgyro = 'rotationRate.z'

# KUHAR dataset data & labels
kuharPath = absPath_ + '/dataset/KU HAR Dataset/'

stand = '0.Stand/'
sit = '1.Sit/'
lay = '5.Lay/'
walk = '11.Walk/'
ups = '15.Stair-up/'
downs = '16.Stair-down/'

activityListKUHAR = [stand, sit, lay, walk, ups, downs]

time1 = 'time1'
xKUacc = 'userAcceleration.x'
yKUacc = 'userAcceleration.y'
zKUacc = 'userAcceleration.z'
time2 = 'time2'
xKUgyro = 'rotationRate.x'
yKUgyro = 'rotationRate.y'
zKUgyro = 'rotationRate.z'

columnsKUHAR = [time1, xKUacc, yKUacc, zKUacc, time2, xKUgyro, yKUgyro, zKUgyro]

# labels per il trainset
xacc = 'x-acc'
yacc = 'y-acc'
zacc = 'z-acc'
magacc = 'magnitude-acc'
std = 'standard-deviation'
xgyro = 'x-gyro'
ygyro = 'y-gyro'
zgyro = 'z-gyro'
maggyro = 'magnitude-gyro'

xAngle = 'angle(X,gravityMean)'
yAngle = 'angle(Y,gravityMean)'
zAngle = 'angle(Z,gravityMean)'

cols = [xacc, yacc, zacc, magacc, xgyro, ygyro, zgyro, maggyro, std, xAngle, yAngle, zAngle, 'Activity']
finalColumns = [xacc, yacc, zacc, magacc, xgyro, ygyro, zgyro, maggyro, std, xAngle, yAngle, zAngle]

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
labelDict = {'WALKING': 1, 'WALKING_UPSTAIRS': 2, 'WALKING_DOWNSTAIRS': 3, 'SITTING': 4, 'STANDING': 5, 'LAYING': 6}


# funzioni utili per il caricamento di UCIHAR dataset
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


# funzioni da utilizzare per selezionare le feature dai vari dataset
def reduceSample(Xdf, yDf):
    # riduce la frequenza dei campioni da 100 Hz a 50Hz, da utilizzare con KUHAR
    reduce = pd.DataFrame(
        columns=[xacc, yacc, zacc, magacc, xgyro, ygyro, zgyro, maggyro, std, xAngle, yAngle, zAngle, 'label'])

    finalX = pd.DataFrame(columns=finalColumns)
    finalY = pd.DataFrame(columns=['Activity'])

    reduce[xacc] = Xdf[xacc]
    reduce[yacc] = Xdf[yacc]
    reduce[zacc] = Xdf[zacc]
    reduce[magacc] = Xdf[magacc]
    reduce[std] = Xdf[std]
    reduce[xgyro] = Xdf[xgyro]
    reduce[ygyro] = Xdf[ygyro]
    reduce[zgyro] = Xdf[zgyro]
    reduce[maggyro] = np.sqrt((Xdf[xgyro] ** 2) + (Xdf[ygyro] ** 2) + (Xdf[zgyro] ** 2))
    reduce[xAngle] = Xdf[xAngle]
    reduce[yAngle] = Xdf[yAngle]
    reduce[zAngle] = Xdf[zAngle]

    reduce['label'] = yDf['Activity']

    reduce = resample(reduce, replace=True, n_samples=int((len(reduce) / 2)), random_state=0)
    reduce = reduce.reset_index(drop=True)

    finalX[xacc] = reduce[xacc]
    finalX[yacc] = reduce[yacc]
    finalX[zacc] = reduce[zacc]
    finalX[magacc] = reduce[magacc]

    finalX[std] = reduce[std]
    finalX[xgyro] = reduce[xgyro]
    finalX[ygyro] = reduce[ygyro]
    finalX[zgyro] = reduce[zgyro]
    finalX[maggyro] = reduce[maggyro]

    finalX[xAngle] = reduce[xAngle]
    finalX[yAngle] = reduce[yAngle]
    finalX[zAngle] = reduce[zAngle]

    finalY['Activity'] = reduce['label']

    return finalX.copy(), finalY.copy()


def loadNmergeMS(X_df, Y_df, path, label):
    # Funzione che carica i KUHARIntero contenuti nei file del dataset UMAFALL ne carica i KUHARIntero selezionando solo le feature utili
    # e li concatena nel dataset finale di MotionSense

    df = pd.read_csv(path)

    finalDf = pd.DataFrame(columns=finalColumns, dtype='float64')

    dfMagnitude = pd.DataFrame(columns=['magXY', 'magYZ', 'magXZ'], dtype='float64')

    dfArcsin = pd.DataFrame(columns=['arcsinx', 'arcsiny', 'arcsinz'], dtype='float64')

    finalDf[xacc] = df[xMSacc]
    finalDf[yacc] = df[yMSacc]
    finalDf[zacc] = df[zMSacc]
    finalDf[magacc] = np.sqrt((finalDf[xacc] ** 2) + (finalDf[yacc] ** 2) + (finalDf[zacc] ** 2))

    dfMagnitude['magXY'] = normalizeData(np.sqrt(np.abs((finalDf[xacc] ** 2) + (finalDf[yacc] ** 2))))
    dfMagnitude['magYZ'] = normalizeData(np.sqrt(np.abs((finalDf[yacc] ** 2) + (finalDf[zacc] ** 2))))
    dfMagnitude['magXZ'] = normalizeData(np.sqrt(np.abs((finalDf[zacc] ** 2) + (finalDf[zacc] ** 2))))

    dfArcsin['arcsinx'] = normalizeData(finalDf[xacc] / np.sqrt(finalDf[magacc]))
    dfArcsin['arcsiny'] = normalizeData(finalDf[yacc] / np.sqrt(finalDf[magacc]))
    dfArcsin['arcsinz'] = normalizeData(finalDf[zacc] / np.sqrt(finalDf[magacc]))

    finalDf[xAngle] = np.arcsin(dfArcsin['arcsinx'])
    finalDf[yAngle] = np.arcsin(dfArcsin['arcsiny'])
    finalDf[zAngle] = np.arcsin(dfArcsin['arcsinz'])

    finalDf[xgyro] = df[xMSgyro]
    finalDf[ygyro] = df[yMSgyro]
    finalDf[zgyro] = df[zMSgyro]
    finalDf[maggyro] = np.sqrt((finalDf[xgyro] ** 2) + (finalDf[ygyro] ** 2) + (finalDf[zgyro] ** 2))

    finalDf[std] = finalDf.std(axis=1, skipna=True)

    X_df = pd.concat([X_df, finalDf])
    X_df = X_df.reset_index(drop=True)

    for i in range(len(Y_df), len(X_df)):
        Y_df.append(labelDict[label])

    return X_df, Y_df


def normalizeData(df):
    return (df - df.mean()) / df.std()


def loadNmergeKU(X_df, Y_df, path, label):
    # Funzione che carica i KUHARIntero contenuti nei file del dataset UMAFALL ne carica i KUHARIntero selezionando solo le feature utili
    # e li concatena nel dataset finale di MotionSense

    df = pd.read_csv(path, header=None, names=columnsKUHAR)

    finalDf = pd.DataFrame(columns=finalColumns, dtype='float32')

    dfMagnitude = pd.DataFrame(columns=['magXY', 'magYZ', 'magXZ'], dtype='float32')

    dfArcsin = pd.DataFrame(columns=['arcsinx', 'arcsiny', 'arcsinz'], dtype='float32')

    finalDf[xacc] = df[xKUacc]
    finalDf[yacc] = df[yKUacc]
    finalDf[zacc] = df[zKUacc]
    finalDf[magacc] = np.sqrt((finalDf[xacc] ** 2) + (finalDf[yacc] ** 2) + (finalDf[zacc] ** 2))

    dfMagnitude['magXY'] = np.sqrt(np.abs((finalDf[xacc] ** 2) + (finalDf[yacc] ** 2)))
    dfMagnitude['magYZ'] = np.sqrt(np.abs((finalDf[yacc] ** 2) + (finalDf[zacc] ** 2)))
    dfMagnitude['magXZ'] = np.sqrt(np.abs((finalDf[zacc] ** 2) + (finalDf[zacc] ** 2)))

    # angolo = arcsin(y/radquadra(magnitudine))
    dfArcsin['arcsinx'] = normalizeData(finalDf[xacc] / np.sqrt(finalDf[magacc]))
    dfArcsin['arcsiny'] = normalizeData(finalDf[yacc] / np.sqrt(finalDf[magacc]))
    dfArcsin['arcsinz'] = normalizeData(finalDf[zacc] / np.sqrt(finalDf[magacc]))

    finalDf[xAngle] = np.arcsin(dfArcsin['arcsinx'])
    finalDf[yAngle] = np.arcsin(dfArcsin['arcsiny'])
    finalDf[zAngle] = np.arcsin(dfArcsin['arcsinz'])

    finalDf[xgyro] = df[xKUgyro]
    finalDf[ygyro] = df[yKUgyro]
    finalDf[zgyro] = df[zKUgyro]
    finalDf[maggyro] = np.sqrt((finalDf[xgyro] ** 2) + (finalDf[ygyro] ** 2) + (finalDf[zgyro] ** 2))

    finalDf[std] = finalDf.std(axis=1, skipna=True)

    X_df = pd.concat([X_df, finalDf])
    X_df = X_df.reset_index(drop=True)

    for i in range(len(Y_df), len(X_df)):
        Y_df.append(labelDict[label])

    return X_df, Y_df


def loadUCIHAR():
    # copia ed elaborazione dei KUHARIntero contenuti nell'UCIHAR
    # UCI HAR Dataset caricato correttamente con il nome di ogni feature
    # restituisce due dataset
    # FUNZIONA

    X, Y = get_human_dataset()

    X_df = pd.DataFrame(columns=finalColumns, dtype='float64')
    Y_df = pd.DataFrame(columns=activity, dtype='int64')

    X_df[xacc] = X[xUCIacc]
    X_df[yacc] = X[yUCIacc]
    X_df[zacc] = X[zUCIacc]
    X_df[magacc] = np.sqrt((X[xUCIacc] ** 2) + (X[yUCIacc] ** 2) + (X[zUCIacc] ** 2))

    X_df[xgyro] = X[xUCIgyro]
    X_df[ygyro] = X[yUCIgyro]
    X_df[zgyro] = X[zUCIgyro]
    X_df[maggyro] = np.sqrt((X[xUCIgyro] ** 2) + (X[yUCIgyro] ** 2) + (X[zUCIgyro] ** 2))
    X_df[xAngle] = X[xAngle]
    X_df[yAngle] = X[yAngle]
    X_df[zAngle] = X[zAngle]

    X_df[std] = X_df.std(axis=1, skipna=True)

    Y_df['Activity'] = Y['action']

    X_df = X_df.reset_index(drop=True)
    Y_df = Y_df.reset_index(drop=True)

    # X_df, Y_df = reduceSample(X_df, Y_df)

    return X_df.copy(), Y_df.copy()


def loadMotionSense():
    X_df = pd.DataFrame(columns=finalColumns, dtype='float64')
    Y_df = pd.DataFrame(columns=activity, dtype='int64')
    Y_label = []

    X_df, Y_label = loadNmergeMS(X_df, Y_label, motionPath + activityListMotionSense[0], 'WALKING_DOWNSTAIRS')

    # X_df, Y_label = loadNmergeMS(X_df, Y_label, motionPath + activityListMotionSense[1], 'LAYING') #in realta' si tratta di jogging, controllare se esiste

    X_df, Y_label = loadNmergeMS(X_df, Y_label, motionPath + activityListMotionSense[2], 'SITTING')

    X_df, Y_label = loadNmergeMS(X_df, Y_label, motionPath + activityListMotionSense[3], 'WALKING_UPSTAIRS')

    X_df, Y_label = loadNmergeMS(X_df, Y_label, motionPath + activityListMotionSense[4], 'WALKING')

    X_df, Y_label = loadNmergeMS(X_df, Y_label, motionPath + activityListMotionSense[5], 'STANDING')

    yTemp = pd.DataFrame(Y_label, columns=activity, dtype='int64')

    # per qualche motivo droppa intere categorie e si perdono dei KUHARIntero riguardanti le attività 3 e 4
    X_df['Activity'] = yTemp['Activity']

    X_df.dropna(subset=[xAngle, yAngle, zAngle], inplace=True)

    Y_df = pd.DataFrame(columns=activity, dtype='int64')
    Y_df['Activity'] = X_df['Activity']
    X_df.drop('Activity', axis='columns', inplace=True)

    X_df = X_df.reset_index(drop=True)
    Y_df = Y_df.reset_index(drop=True)

    X_df, Y_df = reduceSample(X_df, Y_df)

    return X_df.copy(), Y_df.copy()


def loadKUHAR():
    # dataset finali che conterranno i KUHARIntero per come ci servono
    X_df = pd.DataFrame(columns=finalColumns, dtype='float32')
    Y_df = pd.DataFrame(columns=activity, dtype='int32')

    Y_label = []

    X_df, Y_label = loadNmergeKU(X_df, Y_label, kuharPath + activityListKUHAR[0], 'STANDING')

    X_df, Y_label = loadNmergeKU(X_df, Y_label, kuharPath + activityListKUHAR[1], 'SITTING')

    X_df, Y_label = loadNmergeKU(X_df, Y_label, kuharPath + activityListKUHAR[2], 'LAYING')

    X_df, Y_label = loadNmergeKU(X_df, Y_label, kuharPath + activityListKUHAR[3], 'WALKING')

    X_df, Y_label = loadNmergeKU(X_df, Y_label, kuharPath + activityListKUHAR[4], 'WALKING_UPSTAIRS')

    X_df, Y_label = loadNmergeKU(X_df, Y_label, kuharPath + activityListKUHAR[5], 'WALKING_DOWNSTAIRS')

    yTemp = pd.DataFrame(Y_label, columns=activity, dtype='int64')
    X_df['Activity'] = yTemp['Activity']
    X_df.dropna(subset=[xAngle, yAngle, zAngle], inplace=True)

    Y_df = pd.DataFrame(columns=activity, dtype='int64')
    Y_df['Activity'] = X_df['Activity']
    X_df.drop('Activity', axis='columns', inplace=True)

    X_df = X_df.reset_index(drop=True)
    Y_df = Y_df.reset_index(drop=True)

    X_df, Y_df = reduceSample(X_df, Y_df)

    return X_df.copy(), Y_df.copy()


def loadData():
    # carica i KUHARIntero in memoria dei tre dataset
    print('Inizio caricamento dataset...')

    # carico UCIHAR
    XDataUCI, yDataUCI = loadUCIHAR()
    # print('UCIHAR\n', XDataUCI, yDataUCI)
    # carico MotionSense
    XDataMS, yDataMS = loadMotionSense()
    # print('MotionSense\n', XDataMS, yDataMS)
    # carico KUHAR
    XDataKU, yDataKU = loadKUHAR()
    # print('KUHAR\n', XDataKU, yDataKU)
    # unione dei tre dataset

    X_df = pd.concat([XDataUCI, XDataKU, XDataMS])
    X_df = X_df.reset_index(drop=True)

    y_df = pd.concat([yDataUCI, yDataKU, yDataMS])
    y_df = y_df.reset_index(drop=True)

    return X_df, y_df


def saveData(X, Y, path):
    print('Salvataggio KUHARIntero in ShieldApp...')
    X.to_csv(path + 'xData.csv', index=False)
    Y.to_csv(path + 'yData.csv', index=False)


def loadSavedData():
    print('Caricamento KUHARIntero da ShieldApp...')
    x = pd.read_csv(xPath)
    y = pd.read_csv(yPath)

    # x = torch.tensor(x.values)

    return x, y
