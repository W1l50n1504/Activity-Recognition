from com.core import *
from com.utility import *

from dbn.

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

"""
class DeepBeliefNet(BaseModel, ABC):

    def __init__(self):
        super().__init__()

    def dataProcessing(self):
        print('Elaborazione dei dati...')
        ss = standardscaler()
        self.X = ss.fit_transform(self.X)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                                                random_state=42)

        print('Fine elaborazione dati.')

    def modelCreation(self):
        print('Creazione Modello...')
        self.model = SupervisedDBNClassification(hidden_layers_structure=[256, 256],
                                                 learning_rate_rbm=0.05,
                                                 learning_rate=0.1,
                                                 n_epochs_rbm=self.epochs,
                                                 n_iter_backprop=100,
                                                 batch_size=32,
                                                 activation_function='relu',
                                                 dropout_p=0.2)
        print('Fine creazione.')

    def fit(self):
        print('Inizio fitting del modello...')
        self.model.fit(X_train, Y_train)
        print('Fine fitting.')

    def plot(self):
        print('Inizio plotting delle metriche...')
        # plotting confusion matrix
        rounded_labels = np.argmax(self.y_test, axis=1)
        y_pred = self.model.predict(self.X_test)

        mat = confusion_matrix(rounded_labels, y_pred)
        plot_confusion_matrix(conf_mat=mat, show_normed=True, figsize=(10, 10))

        plt.figure(figsize=(10, 10))
        array = confusion_matrix(rounded_labels, y_pred)
        df_cm = pd.DataFrame(array, range(6), range(6))
        df_cm.columns = ["Walking", "W_Upstairs", "W_Downstairs", "Sitting", "Standing", "Laying", 'Jogging', 'Hopping',
                         'Falling']
        df_cm.index = ["Walking", "W_Upstairs", "W_Downstairs", "Sitting", "Standing", "Laying", 'Jogging', 'Hopping',
                       'Falling']
        sns.heatmap(df_cm, annot=True, annot_kws={"size": 12},
                    yticklabels=(
                        "Walking", "W_Upstairs", "W_Downstairs", "Sitting", "Standing", "Laying", 'Jogging', 'Hopping',
                        'Falling'),
                    xticklabels=(
                        "Walking", "W_Upstairs", "W_Downstairs", "Sitting", "Standing", "Laying", 'Jogging', 'Hopping',
                        'Falling'))  # font size

        plt.show()

        # Plot training & validation accuracy values
        plt.figure(figsize=(15, 8))
        epoch_range = range(1, self.epochs + 1)
        plt.plot(epoch_range, self.history.history['accuracy'])
        plt.plot(epoch_range, self.history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.savefig(trainingValAccCNN)
        # plt.show()

        # Plot training & validation auc values
        plt.figure(figsize=(15, 8))
        epoch_range = range(1, self.epochs + 1)
        plt.plot(epoch_range, self.history.history['auc'])
        plt.plot(epoch_range, self.history.history['val_auc'])
        plt.title('Model auc')
        plt.ylabel('auc')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.savefig(trainingValAucCNN)
        # plt.show()

        # Plot training & validation loss values
        plt.figure(figsize=(15, 8))
        plt.plot(epoch_range, self.history.history['loss'])
        plt.plot(epoch_range, self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.savefig(modelLossCNN)
        # plt.show()
        print('Fine plotting.')
"""

if __name__ == '__main__':
    X, y = loadSavedData()
    print('Elaborazione dei dati...')
    ss = standardscaler()
    X = ss.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)

    print('Fine elaborazione dati.')
    print('Creazione Modello...')
    model = SupervisedDBNClassification(hidden_layers_structure=[256, 256],
                                        learning_rate_rbm=0.05,
                                        learning_rate=0.1,
                                        n_epochs_rbm=10,
                                        n_iter_backprop=100,
                                        batch_size=32,
                                        activation_function='relu',
                                        dropout_p=0.2)
    print('Fine creazione.')

    print('Inizio fitting...')
    model.fit(X_train, Y_train)
    print('Fine fitting.')
    print('Inizio plotting delle metriche...')
    # plotting confusion matrix
    rounded_labels = np.argmax(self.y_test, axis=1)
    y_pred = self.model.predict(self.X_test)

    mat = confusion_matrix(rounded_labels, y_pred)
    plot_confusion_matrix(conf_mat=mat, show_normed=True, figsize=(10, 10))

    plt.figure(figsize=(10, 10))
    array = confusion_matrix(rounded_labels, y_pred)
    df_cm = pd.DataFrame(array, range(6), range(6))
    df_cm.columns = ["Walking", "W_Upstairs", "W_Downstairs", "Sitting", "Standing", "Laying", 'Jogging', 'Hopping',
                     'Falling']
    df_cm.index = ["Walking", "W_Upstairs", "W_Downstairs", "Sitting", "Standing", "Laying", 'Jogging', 'Hopping',
                   'Falling']
    sns.heatmap(df_cm, annot=True, annot_kws={"size": 12},
                yticklabels=(
                    "Walking", "W_Upstairs", "W_Downstairs", "Sitting", "Standing", "Laying", 'Jogging', 'Hopping',
                    'Falling'),
                xticklabels=(
                    "Walking", "W_Upstairs", "W_Downstairs", "Sitting", "Standing", "Laying", 'Jogging', 'Hopping',
                    'Falling'))  # font size

    plt.show()

    # Plot training & validation accuracy values
    plt.figure(figsize=(15, 8))
    epoch_range = range(1, self.epochs + 1)
    plt.plot(epoch_range, self.history.history['accuracy'])
    plt.plot(epoch_range, self.history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(trainingValAccCNN)
    # plt.show()

    # Plot training & validation auc values
    plt.figure(figsize=(15, 8))
    epoch_range = range(1, self.epochs + 1)
    plt.plot(epoch_range, self.history.history['auc'])
    plt.plot(epoch_range, self.history.history['val_auc'])
    plt.title('Model auc')
    plt.ylabel('auc')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(trainingValAucCNN)
    # plt.show()

    # Plot training & validation loss values
    plt.figure(figsize=(15, 8))
    plt.plot(epoch_range, self.history.history['loss'])
    plt.plot(epoch_range, self.history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(modelLossCNN)
    # plt.show()
    print('Fine plotting.')
