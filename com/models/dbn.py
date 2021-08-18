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
        # elaborazione dei dati nel formato utile al funzionamento del dbn con conseguente separazione in train and test data
        self.X = np.array(self.X).astype('float64')
        self.X = np.abs(self.X) + 1

        self.y = np.array(self.y.values)
        self.y = self.y.flatten()

        # Splitting data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3,
                                                                                random_state=42)

    def modelCreation(self):
        self.model = SupervisedDBNClassification(hidden_layers_structure=[256, 256],
                                                 learning_rate_rbm=0.05,
                                                 learning_rate=0.1,
                                                 n_epochs_rbm=self.epochs,
                                                 n_iter_backprop=100,
                                                 batch_size=32,
                                                 activation_function='relu',
                                                 dropout_p=0.2)

    def fit(self):
        self.history = self.model.fit(self.X_train, self.y_train)
        self.saveModel()

    def saveModel(self):
        self.model.save('C:/Users/david/PycharmProjects/ActivityRecognition683127/com/checkpoint/DBN/bestDBN.pkl')

    def loadModel(self):
        self.model = SupervisedDBNClassification.load(
            'C:/Users/david/PycharmProjects/ActivityRecognition683127/com/checkpoint/DBN/bestDBN.pkl')

    def plot(self):
        rounded_labels = self.y_test
        # rounded_labels = np.argmax(Y_test, axis=1)

        y_pred = self.model.predict(self.X_test)

        # print('round', rounded_labels.shape)
        # print('y', y_pred.shape)

        mat = confusion_matrix(rounded_labels, y_pred)
        plot_confusion_matrix(conf_mat=mat, show_normed=True, figsize=(10, 10))

        plt.figure(figsize=(10, 10))
        array = confusion_matrix(rounded_labels, y_pred)
        df_cm = pd.DataFrame(array, range(5), range(5))

        df_cm.columns = ["Walking", "W_Upstairs", "W_Downstairs", "Sitting", "Standing"]#, "Laying", "Jogging"]
        df_cm.index = ["Walking", "W_Upstairs", "W_Downstairs", "Sitting", "Standing"]#, "Laying", "Jogging"]
        sns.set(font_scale=1)  # for label size
        sns.heatmap(df_cm, annot=True, annot_kws={"size": 12},
                    yticklabels=(
                        "Walking", "W_Upstairs", "W_Downstairs", "Sitting", "Standing", "Laying"),  # , "Jogging"),
                    xticklabels=(
                        "Walking", "W_Upstairs", "W_Downstairs", "Sitting", "Standing", "Laying"))  # , "Jogging"))

        plt.show()

    def main(self):
        self.modelCreation()
        self.fit()
        self.saveModel()
        self.plot()


if __name__ == '__main__':
    dbn = DeepBeliefNetwork()
    dbn.main()
