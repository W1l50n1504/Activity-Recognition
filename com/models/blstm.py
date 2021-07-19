from com.core import *
from com.utility import *

import tensorflow as tf

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


class BLSTM(BaseModel, ABC):
    def __init__(self):
        super().__init__()

    def dataProcessing(self):
        print('elaborazione dei dati...')

        self.X = np.array(self.X)
        self.y = np.array(self.y)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3,
                                                                                random_state=42)

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=0.1,
                                                                              random_state=42)

        enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
        enc = enc.fit(self.y_train)

        self.y_train = enc.transform(self.y_train)
        self.y_test = enc.transform(self.y_test)
        self.y_val = enc.transform(self.y_val)

        self.X_train = self.X_train.reshape(707582, 4, 1)
        self.X_test = self.X_test.reshape(336945, 4, 1)
        self.X_val = self.X_val.reshape(78621, 4, 1)

        print('fine elaborazione dati')

    def modelCreation(self):
        print('Creazione Modello...')

        self.model = Sequential()

        self.model.add(
            Bidirectional(
                LSTM(units=64, return_sequences=True, input_shape=[self.X_train.shape[1], self.X_train.shape[2]])))

        self.model.add(Dropout(rate=0.1))

        self.model.add(Bidirectional(LSTM(units=128)))

        self.model.add(Dense(units=256, activation='relu'))

        self.model.add(Dropout(rate=0.5))

        self.model.add(Dense(self.y_train.shape[1], activation='softmax'))

        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy',
                           metrics=['accuracy', tf.keras.metrics.AUC()])

        print('Fine creazione')

    def fit(self):
        self.checkpoint = ModelCheckpoint(
            checkPointPathBLSTM + '/best_model.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True,
            mode='auto',
            period=1)

        self.history = self.model.fit(self.X_train, self.y_train, batch_size=16, epochs=10,
                                      validation_data=(self.X_test, self.y_test), verbose=1,
                                      callbacks=[self.checkpoint])

    def plot(self):
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
        plt.savefig(confusionMatrixBLSTM)
        # Plot training & validation accuracy values
        plt.figure(figsize=(15, 8))
        epoch_range = range(1, self.epochs + 1)
        plt.plot(epoch_range, self.history.history['accuracy'])
        plt.plot(epoch_range, self.history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.savefig(trainingValAccBLSTM)
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
        plt.savefig(TrainingValAucBLSTM)
        # plt.show()

        # Plot training & validation loss values
        plt.figure(figsize=(15, 8))
        plt.plot(epoch_range, self.history.history['loss'])
        plt.plot(epoch_range, self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.savefig(ModelLossBLSTM)
        # plt.show()


if __name__ == '__main__':
    blstm = BLSTM()
    blstm.main()
