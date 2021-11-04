from com.dbn_libraries import SupervisedDBNClassification
from com.core import *
from com.utility import *

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


class DeepBeliefNetwork(BaseModel, ABC):
    def __init__(self):
        super().__init__()

    def dataProcessing(self):
        self.X_train = np.abs(self.X) + self.X.mean()
        self.X_train = np.array(self.X).astype('float64')

        print(self.y)

        self.y_train = np.array(self.y)
        print(self.y_train)
        self.y_train = self.y_train.flatten()

        # Splitting data
        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3,
        #                                                                       random_state=42)

    def modelCreation(self):
        """self.model = SupervisedDBNClassification(hidden_layers_structure=[1024, 1024],
                                                 learning_rate_rbm=0.1,
                                                 learning_rate=0.1,
                                                 n_epochs_rbm=10,
                                                 n_iter_backprop=self.epochs,
                                                 batch_size=32,
                                                 activation_function='relu',
                                                 dropout_p=0.02)
        """
        self.model = SupervisedDBNClassification(hidden_layers_structure=[256, 256],
                                                 learning_rate_rbm=0.05,
                                                 learning_rate=0.1,
                                                 n_epochs_rbm=10,
                                                 n_iter_backprop=100,
                                                 batch_size=32,
                                                 activation_function='relu',
                                                 dropout_p=0.2)

    def fit(self):
        self.history = self.model.fit(self.X_train, self.y_train)
        self.saveModel()

    def saveModel(self):
        self.model.save('C:/Users/david/PycharmProjects/Activity-Recognition/com/checkpoint/DBN/bestDBN.pkl')

    def loadModel(self):
        self.model = SupervisedDBNClassification.load(
            'C:/Users/david/PycharmProjects/Activity-Recognition/com/checkpoint/DBN/bestDBN.pkl')

    def plot(self):
        y_pred = self.model.predict(self.X_test)

        mat = confusion_matrix(self.y_test, y_pred)
        plot_confusion_matrix(conf_mat=mat, show_normed=True, figsize=(10, 10))

        plt.figure(figsize=(10, 10))
        array = confusion_matrix(self.y_test, y_pred)

        df_cm = pd.DataFrame(array, range(6), range(6))
        df_cm.columns = ["Walking", "W_Upstairs", "W_Downstairs", "Sitting", "Standing", "Laying"]
        df_cm.index = ["Walking", "W_Upstairs", "W_Downstairs", "Sitting", "Standing", "Laying"]

        sns.set(font_scale=1)  # for label size
        sns.heatmap(df_cm, annot=True, annot_kws={"size": 12},
                    yticklabels=("Walking", "W_Upstairs", "W_Downstairs", "Sitting", "Standing", "Laying"),
                    xticklabels=("Walking", "W_Upstairs", "W_Downstairs", "Sitting", "Standing", "Laying"))

        plt.show()

    def main(self):
        self.modelCreation()
        self.fit()
        self.saveModel()
        # self.loadModel()
        self.plot()


if __name__ == '__main__':
    dbn = DeepBeliefNetwork()
    dbn.main()
