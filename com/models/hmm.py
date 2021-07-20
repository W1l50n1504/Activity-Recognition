from com.utility import *
from com.core import *

checkPointPathHMM = absPath_ + '/checkpoint/HMM'


class HMM(BaseModel, ABC):

    def __init__(self):
        super().__init__()
        self.n_mix = 16
        self.n_components = 6
        print('Creazioni matrici di prob...')
        self.startprob = np.zeros(self.n_components)
        self.startprob[0] = 1

        self.transmat = np.zeros((self.n_components, self.n_components))
        self.transmat[0, 0] = 1
        self.transmat[-1, -1] = 1

        for i in range(transmat.shape[0] - 1):
            if i != self.transmat.shape[0]:
                for j in range(i, i + 2):
                    self.transmat[i, j] = 0.5

    def saveModel(self):
        with open(checkPointPathHMM + '/best_model.pkl', "wb") as file:
            pickle.dump(self.model, file)

    def loadModel():
        with open(checkPointPathHMM + '/best_model.pkl', "rb") as file:
            self.model = pickle.load(file)

    def fit(self):


        print('Inizio fitting del modello...')
        model = GMMHMM(n_components=n_components,
                       n_mix=n_mix,
                       covariance_type="diag",
                       init_params="cm", params="cm", verbose=True)

        model.startprob_ = np.array(startprob)
        model.transmat_ = np.array(transmat)

        model.fit(X_train)

        print('Salvataggio del modello...')
        saveModel(model)

    def plot(self):
        self.model = loadModel()
        rounded_labels = np.argmax(y_test, axis=1)
        y_pred = model.predict(X_test)

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


if __name__ == '__main__':
    hmm = HMM()
    hmm.main
