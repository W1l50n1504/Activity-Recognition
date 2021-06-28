from com.utility import *

np.random.seed(42)

absPath_ = os.getcwd()


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


def loadModel():
    return None


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
