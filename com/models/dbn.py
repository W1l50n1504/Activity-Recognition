import numpy as np

np.random.seed(42)  # for reproducibility
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from com.dbn_libraries import SupervisedDBNClassification
from com.utility import *
from com.core import *

if __name__ == '__main__':
    # Loading dataset
    X, Y = loadData()

    print(np.all(X))

    X = np.array(X).astype('float64')
    X = np.abs(X) + 1

    print(np.all(X))

    Y = np.array(Y.values)
    Y = Y.flatten()
    # Splitting data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Training
    classifier = SupervisedDBNClassification(hidden_layers_structure=[256, 256],
                                             learning_rate_rbm=0.05,
                                             learning_rate=0.1,
                                             n_epochs_rbm=10,
                                             n_iter_backprop=100,
                                             batch_size=32,
                                             activation_function='relu',
                                             dropout_p=0.2)
    classifier.fit(X_train, Y_train)
    # Save the model
    classifier.save('C:/Users/david/PycharmProjects/ActivityRecognition683127/com/checkpoint/DBN/bestDBN.pkl')

    # Restore
    classifier = SupervisedDBNClassification.load(
        'C:/Users/david/PycharmProjects/ActivityRecognition683127/com/checkpoint/DBN/bestDBN.pkl')
    # Test
    #    rounded_labels = np.argmax(Y_test, axis=1)
    rounded_labels = Y_test
    y_pred = classifier.predict(X_test)

    print('round', rounded_labels.shape)
    # print('y', y_pred.shape)

    mat = confusion_matrix(rounded_labels, y_pred)
    plot_confusion_matrix(conf_mat=mat, show_normed=True, figsize=(10, 10))

    plt.figure(figsize=(10, 10))
    array = confusion_matrix(rounded_labels, y_pred)
    df_cm = pd.DataFrame(array, range(7), range(7))

    df_cm.columns = ["Walking", "W_Upstairs", "W_Downstairs", "Sitting", "Standing", "Laying", "Jogging"]
    df_cm.index = ["Walking", "W_Upstairs", "W_Downstairs", "Sitting", "Standing", "Laying", "Jogging"]
    sns.set(font_scale=1)  # for label size
    sns.heatmap(df_cm, annot=True, annot_kws={"size": 12},
                yticklabels=(
                    "Walking", "W_Upstairs", "W_Downstairs", "Sitting", "Standing", "Laying", "Jogging"),
                xticklabels=(
                    "Walking", "W_Upstairs", "W_Downstairs", "Sitting", "Standing", "Laying", "Jogging"))

    plt.show()
    # Plot training & validation accuracy values
    plt.figure(figsize=(15, 8))
    epoch_range = range(1, 10 + 1)
    plt.plot(epoch_range, history.history['Accuracy'])
    plt.plot(epoch_range, history.history['val_Accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    # plt.savefig(trainingValAccBLSTM)
    plt.show()

    # Plot training & validation auc values
    plt.figure(figsize=(15, 8))
    epoch_range = range(1, 10 + 1)
    plt.plot(epoch_range, history.history['auc'])
    plt.plot(epoch_range, history.history['val_auc'])
    plt.title('Model auc')
    plt.ylabel('auc')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    # plt.savefig(TrainingValAucBLSTM)
    plt.show()

    # Plot training & validation loss values
    plt.figure(figsize=(15, 8))
    plt.plot(epoch_range, history.history['loss'])
    plt.plot(epoch_range, history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    # plt.savefig(ModelLossBLSTM)
    plt.show()

    Y_pred = classifier.predict(X_test)
