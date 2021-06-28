import numpy as np
import os
import tensorflow

from tensorflow.keras.utils import to_categorical

absPath_ = os.getcwd()

X_train_signals_paths = absPath_ + '/dataset/train/X_train.txt'
X_test_signals_paths = absPath_ + '/dataset/test/X_test.txt'

y_train_path = absPath_ + '/dataset/train/y_train.txt'
y_test_path = absPath_ + '/dataset/test/y_test.txt'

pathToSignalTrain = absPath_ + '/dataset/train/Inertial Signals/'
pathToSignalTest = absPath_ + '/dataset/test/Inertial Signals/'

nameXtrain = 'total_acc_x_train.txt'
nameYtrain = 'total_acc_y_train.txt'
nameZtrain = 'total_acc_z_train.txt'

nameXtest = 'total_acc_x_test.txt'
nameYtest = 'total_acc_y_test.txt'
nameZtest = 'total_acc_z_test.txt'

hmmGraph = absPath_ + '/grafici/Markov/grafico.png'
hmmDistribution = absPath_ + '/grafici/Markov/distribution.png'

checkPointPathCNN = absPath_ + '/checkpoint/CNN'
checkPointPathBLSTM = absPath_ + '/checkpoint/BLSTM'

trainingValAccCNN = absPath_ + '/grafici/CNN/CNNAcc.png'
trainingValAccBLSTM = absPath_ + '/grafici/BLSTM/BLSTMAcc.png'

TrainingValAucCNN = absPath_ + '/grafici/CNN/CNNAuc.png'
TrainingValAucBLSTM = absPath_ + '/grafici/BLSTM/BLSTMAuc.png'

ModelLossCNN = absPath_ + '/grafici/CNN/ModelLossCNN.png'
ModelLossBLSTM = absPath_ + '/grafici/BLSTM/ModelLossBLSTM.png'

checkPointPathHMM = absPath_ + '/checkpoint/HMM'

labelDict = {'WALKING': 0, 'WALKING_UPSTAIRS': 1, 'WALKING_DOWNSTAIRS': 2,
             'SITTING': 3, 'STANDING': 4, 'LAYING': 5}


def norm(data):
    return (data - data.mean()) / data.std() + np.finfo(np.float32).eps


def produceMagnitude(flag):
    magnitude = []
    if flag:

        x = norm(load_X(pathToSignalTrain + nameXtrain))
        y = norm(load_X(pathToSignalTrain + nameYtrain))
        z = norm(load_X(pathToSignalTrain + nameZtrain))

    else:
        x = norm(load_X(pathToSignalTest + nameXtest))
        y = norm(load_X(pathToSignalTest + nameYtest))
        z = norm(load_X(pathToSignalTest + nameZtest))

    for i in range(0, len(x)):
        magnitude.append(np.sqrt(x[i] ** 2 + y[i] ** 2 + z[i] ** 2))

    # print('\n', magnitude)

    return magnitude


def encode(train_X, train_y, test_X, test_y):
    # forse da eliminare
    train_y = train_y - 1
    test_y = test_y - 1
    # one hot encode y
    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)

    return train_X, train_y, test_X, test_y


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


def loadDataHMM():
    print('caricamento dei dati di training e test')
    X_train = produceMagnitude(0)
    X_test = produceMagnitude(1)

    y_train = load_y(y_train_path)
    y_test = load_y(y_test_path)

    # print('X_train', X_train)
    # print('y_train', y_train)
    # print('X_test', X_test)
    # print('y_test', y_test)

    print('fine caricamento')
    return X_train, y_train, X_test, y_test


def loadDataCNN():
    print('caricamento dei dati di training e test')

    X_train = load_X(X_train_signals_paths)
    X_test = load_X(X_test_signals_paths)

    y_train = load_y(y_train_path)
    y_test = load_y(y_test_path)

    print('fine caricamento')
    return X_train, y_train, X_test, y_test


def loadDataBLSTM():
    print('caricamento dei dati di training e test')

    X_train = load_X(X_train_signals_paths)
    X_test = load_X(X_test_signals_paths)

    y_train = load_y(y_train_path)
    y_test = load_y(y_test_path)

    print('fine caricamento')
    return X_train, y_train, X_test, y_test


def dataProcessingHMM(X_train, y_train, X_test, y_test):
    print('elaborazione dei dati...')

    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))

    X = np.log(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    #enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    #enc = enc.fit(y_train)
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


def dataProcessingCNN(X_train, y_train, X_test, y_test):
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

    # print('dimensione reshape', X_val[..., np.newaxis].shape)

    X_train = X_train.reshape(6488, 561, 1, 1)
    X_test = X_test.reshape(3090, 561, 1, 1)
    X_val = X_val.reshape(721, 561, 1, 1)

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


def plot_learningCurveCNN(history, epochs):
    # Plot training & validation accuracy values
    plt.figure(figsize=(15, 8))
    epoch_range = range(1, epochs + 1)
    plt.plot(epoch_range, history.history['accuracy'])
    plt.plot(epoch_range, history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(trainingValAccCNN)
    # plt.show()

    # Plot training & validation auc values
    plt.figure(figsize=(15, 8))
    epoch_range = range(1, epochs + 1)
    plt.plot(epoch_range, history.history['auc'])
    plt.plot(epoch_range, history.history['val_auc'])
    plt.title('Model auc')
    plt.ylabel('auc')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(TrainingValAucCNN)
    # plt.show()

    # Plot training & validation loss values
    plt.figure(figsize=(15, 8))
    plt.plot(epoch_range, history.history['loss'])
    plt.plot(epoch_range, history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(ModelLossCNN)
    # plt.show()


def plot_learningCurveBLSTM(history, epochs):
    # Plot training & validation accuracy values
    plt.figure(figsize=(15, 8))
    epoch_range = range(1, epochs + 1)
    plt.plot(epoch_range, history.history['accuracy'])
    plt.plot(epoch_range, history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(trainingValAccBLSTM)
    # plt.show()

    # Plot training & validation auc values
    plt.figure(figsize=(15, 8))
    epoch_range = range(1, epochs + 1)
    plt.plot(epoch_range, history.history['auc'])
    plt.plot(epoch_range, history.history['val_auc'])
    plt.title('Model auc')
    plt.ylabel('auc')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(TrainingValAucBLSTM)
    # plt.show()

    # Plot training & validation loss values
    plt.figure(figsize=(15, 8))
    plt.plot(epoch_range, history.history['loss'])
    plt.plot(epoch_range, history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(ModelLossBLSTM)
    # plt.show()
