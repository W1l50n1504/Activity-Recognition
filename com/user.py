from com.utility import *

standK = '0.Stand/'
sitK = '1.Sit/'
layK = '5.Lay/'
walkK = '11.Walk/'
upsK = '15.Stair-up/'
downsK = '16.Stair-down/'

fileListStandK = os.listdir(kuharPath + standK)
fileListSitK = os.listdir(kuharPath + sitK)
fileListLayK = os.listdir(kuharPath + layK)
fileListWalkK = os.listdir(kuharPath + walkK)
fileListUpsK = os.listdir(kuharPath + upsK)
fileListDownsk = os.listdir(kuharPath + downsK)

standMS = 'std/'
sitMS = 'sit/'
walkMS = 'wlk/'
upsMS = 'ups/'
downsMS = 'dws/'

fileListDownsMS = os.listdir(motionPath + activityListMotionSense[0])
fileListSitMS = os.listdir(motionPath + activityListMotionSense[1])
fileListUpsMS = os.listdir(motionPath + activityListMotionSense[2])
fileListWalkMS = os.listdir(motionPath + activityListMotionSense[3])
fileListStandMS = os.listdir(motionPath + activityListMotionSense[4])


def loadKUHARUser():
    # dataset finali che conterranno i KUHARIntero con tutti gli utenti
    X_df = pd.DataFrame(columns=finalColumns, dtype='float32')
    Y_df = pd.DataFrame(columns=activity, dtype='int32')

    Y_label = []

    for i in range(0, len(fileListStandK)):
        X_df, Y_label = loadNmergeKU(X_df, Y_label, kuharPath + activityListKUHAR[0] + fileListStandK[i], 'STANDING')

    for i in range(0, len(fileListSitK)):
        X_df, Y_label = loadNmergeKU(X_df, Y_label, kuharPath + activityListKUHAR[1] + fileListSitK[i], 'SITTING')

    for i in range(0, len(fileListLayK)):
        X_df, Y_label = loadNmergeKU(X_df, Y_label, kuharPath + activityListKUHAR[2] + fileListLayK[i], 'LAYING')

    for i in range(0, len(fileListWalkK)):
        X_df, Y_label = loadNmergeKU(X_df, Y_label, kuharPath + activityListKUHAR[3] + fileListWalkK[i], 'WALKING')

    for i in range(0, len(fileListUpsK)):
        X_df, Y_label = loadNmergeKU(X_df, Y_label, kuharPath + activityListKUHAR[4] + fileListUpsK[i],
                                     'WALKING_UPSTAIRS')

    for i in range(0, len(fileListDownsk)):
        X_df, Y_label = loadNmergeKU(X_df, Y_label, kuharPath + activityListKUHAR[5] + fileListDownsk[i],
                                     'WALKING_DOWNSTAIRS')

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


def loadUCIHARUser():
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


def loadMotionSenseUser():
    X_df = pd.DataFrame(columns=finalColumns, dtype='float64')
    Y_df = pd.DataFrame(columns=activity, dtype='int64')
    Y_label = []

    for i in range(0, len(fileListDownsMS)):
        X_df, Y_label = loadNmergeMS(X_df, Y_label, motionPath + activityListMotionSense[0] + fileListDownsMS[i],
                                     'WALKING_DOWNSTAIRS')

    for i in range(0, len(fileListSitMS)):
        X_df, Y_label = loadNmergeMS(X_df, Y_label, motionPath + activityListMotionSense[2] + fileListSitMS[i],
                                     'SITTING')

    for i in range(0, len(fileListUpsMS)):
        X_df, Y_label = loadNmergeMS(X_df, Y_label, motionPath + activityListMotionSense[3] + fileListUpsMS[i],
                                     'WALKING_UPSTAIRS')

    for i in range(0, len(fileListWalkMS)):
        X_df, Y_label = loadNmergeMS(X_df, Y_label, motionPath + activityListMotionSense[4] + fileListWalkMS[i],
                                     'WALKING')

    for i in range(0, len(fileListStandMS)):
        X_df, Y_label = loadNmergeMS(X_df, Y_label, motionPath + activityListMotionSense[4] + fileListStandMS[i],
                                     'STANDING')

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


if __name__ == '__main__':
    x2, y2 = loadData(kuharPath)
    x1, y1 = loadUCIHAR()

    y1 = np.array(y1)
    y2 = np.array(y2)

    y = np.concatenate((y1, y2), axis=0)

    print(y)
