from com.core import *
from com.utility import *
from hmmlearn.hmm import GaussianHMM, GMMHMM, MultinomialHMM

np.random.seed(42)

absPath_ = os.getcwd()


class HMM(BaseModel, ABC):

    def __init__(self):
        super().__init__()
        self.X_train, self.y_train, self.X_test, self.y_test = loadDataHMM()
        self.processData()

    def createModel(self):
        self.n_mix = 16
        self.n_components = 6
        print('Creazioni matrici di prob...')
        startprob = np.zeros(self.n_components)
        startprob[0] = 1

        transmat = np.zeros((self.n_components, self.n_components))
        transmat[0, 0] = 1
        transmat[-1, -1] = 1

        for i in range(transmat.shape[0] - 1):
            if i != transmat.shape[0]:
                for j in range(i, i + 2):
                    transmat[i, j] = 0.5

        print('Creazione modello...')
        self.model = MultinomialHMM(n_components=self.n_components)

        self.model.startprob_ = np.array(startprob)
        self.model.transmat_ = np.array(transmat)
        print('fine creazione modello')

    def fit(self):
        print('Inizio fitting del modello...')
        print(self.X_train.shape)
        print(self.y_train.shape)
        self.model.fit(self.X_train, self.y_train)
        print('fine fitting')
        print('Salvataggio del modello...')
        self.saveModel()

    def processData(self):
        print('elaborazione dei dati...')

        X = np.concatenate((self.X_train, self.X_test))
        y = np.concatenate((self.y_train, self.y_test))

        X = np.log(X)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=0.1,
                                                                              random_state=42)

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

        self.X_train = self.X_train.reshape(-1, 1)
        self.X_test = self.X_test.reshape(-1, 1)
        self.X_val = self.X_val.reshape(-1, 1)

        self.y_train = self.y_train.reshape(-1, 1)
        self.y_test = self.y_test.reshape(-1, 1)
        self.y_val = self.y_val.reshape(-1, 1)

        print('fine elaborazione dati')

    def loadModel():
        return None

    def plot(self):
        # non va bene come criterio di controllo della precisione del modello,
        # e' necessario trovare un metodo migliore per valutare la precisione del sistema per compararla con gli altri approcci
        # model = loadModel()
        rounded_labels = np.argmax(self.y_test, axis=1)
        y_pred = self.model.predict(self.X_test)

        print('rounded', rounded_labels.shape)
        print('y_pred', y_pred.shape)
        print('y_test', self.y_test.shape)

        print('y_test', self.y_test)
        print('rounded', rounded_labels)
        print('y_pred', y_pred)

        mat = confusion_matrix(rounded_labels, y_pred)
        plot_confusion_matrix(conf_mat=mat, show_normed=True, figsize=(10, 10))

        plt.figure(figsize=(10, 10))
        array = confusion_matrix(rounded_labels, y_pred)
        df_cm = pd.DataFrame(array, range(6), range(6))
        df_cm.columns = ["Walking", "W_Upstairs", "W_Downstairs", "Sitting", "Standing", "Laying"]
        df_cm.index = ["Walking", "W_Upstairs", "W_Downstairs", "Sitting", "Standing", "Laying"]
        # sn.set(font_scale=1)#for label size
        sns.heatmap(df_cm, annot=True, annot_kws={"size": 12},
                    yticklabels=("Walking", "W_Upstairs", "W_Downstairs", "Sitting", "Standing", "Laying"),
                    xticklabels=("Walking", "W_Upstairs", "W_Downstairs", "Sitting", "Standing", "Laying"))  # font size
        # plt.savefig()
        plt.show()

    def saveModel(self):
        return None


if __name__ == '__main__':
    hmm = HMM()
    hmm.createModel()
    hmm.fit()
    hmm.plot()
