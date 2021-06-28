from com.core import *
from com.utility import *

from abc import ABC

import tensorflow as tf

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


class CNN(BaseModel, ABC):
    def __init__(self):
        super().__init__()
        self.epochs = 10
        print('Creazione Modello...')
        self.model = Sequential()
        self.model.add(Conv2D(64, 1, activation='relu', input_shape=self.X_train[0].shape))
        self.model.add(Dropout(0.1))

        self.model.add(Conv2D(128, 1, activation='relu', padding='valid'))
        self.model.add(MaxPool2D(1, 1))

        self.model.add(Dropout(0.5))
        self.model.add(Flatten())

        self.model.add(Dense(512, activation='relu'))

        self.model.add(Dense(6, activation='softmax'))

        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse',
                           metrics=['accuracy', tf.keras.metrics.AUC()])

        print('Fine creazione')

    def processData(self):
        print('Elaborazione dei dati...')
        self.X_train, self.y_train, self.X_test, self.y_test = loadData()
        x = np.concatenate((self.X_train, self.X_test))
        y = np.concatenate((self.y_train, self.y_test))

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.3, random_state=42)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=0.1,
                                                                              random_state=42)

        enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
        enc = enc.fit(self.y_train)

        self.y_train = enc.transform(self.y_train)
        self.y_test = enc.transform(self.y_test)
        self.y_val = enc.transform(self.y_val)

        # print('dimensione reshape', self.X_val[..., np.newaxis].shape)

        print('fine elaborazione dati')

    def fit(self):
        self.checkpoint = ModelCheckpoint(checkPointPathCNN + '/best_model.hdf5',
                                          monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto', period=1)

        self.history = self.model.fit(self.X_train, self.y_train, batch_size=64, epochs=self.epochs,
                                      validation_data=(self.X_val, self.y_val),
                                      verbose=1, callbacks=[self.checkpoint])

    def plot(self):
        # plotting confusion matrix
        rounded_labels = np.argmax(self.y_test, axis=1)
        y_pred = self.model.predict_classes(self.X_test)

        print('round', rounded_labels.shape)
        print('y', y_pred.shape)

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
        plt.savefig(confusionMatrixCNN)
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


if __name__ == '__main__':
    cnn = CNN()

    cnn.fit()
    cnn.plot()
