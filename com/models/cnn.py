from com.core import *
from com.utility import *

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


class CNN(BaseModel, ABC):
    def __init__(self):
        super().__init__()

    def dataProcessing(self):
        print('Elaborazione dei dati...')

        enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
        enc = enc.fit(self.y)

        self.y = enc.transform(self.y)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3,
                                                                                random_state=42)

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=0.1,
                                                                              random_state=42)

        # self.X_train = tf.convert_to_tensor(self.X_train, dtype=tf.float32)
        # self.X_test = tf.convert_to_tensor(self.X_test, dtype=tf.float32)
        # self.X_val = tf.convert_to_tensor(self.X_val, dtype=tf.float32)

        print('train\n', self.X_train.shape, '\n', self.y_train.shape)
        print('\ntest\n', self.X_test.shape, '\n', self.y_test.shape)
        print('\nval\n', self.X_val.shape, '\n', self.y_val.shape)

        print('Fine elaborazione dati.')

    def modelCreation(self):
        print('Creazione Modello...')
        self.model = Sequential()
        self.model.add(Conv2D(64, 1, activation='relu', input_shape=self.X_train.shape))
        self.model.add(Dropout(0.1))

        self.model.add(Conv2D(128, 1, activation='relu', padding='valid'))
        self.model.add(MaxPool2D(1, 1))

        self.model.add(Dropout(0.5))
        self.model.add(Flatten())

        self.model.add(Dense(512, activation='relu'))

        self.model.add(Dense(6, activation='softmax'))

        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse',
                           metrics=['accuracy', tf.keras.metrics.AUC()])

        print('Fine creazione.')

    def fit(self):
        print('Inizio fitting del modello...')
        self.checkpoint = ModelCheckpoint(checkPointPathCNN + '/best_model.hdf5',
                                          monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto', period=1)

        self.history = self.model.fit(self.X_train, self.y_train, batch_size=64, epochs=self.epochs,
                                      validation_data=(self.X_val, self.y_val),
                                      verbose=1, callbacks=[self.checkpoint])
        print('Fine fitting.')

    def fitWeb(self):
        verbose, epochs, batch_size = 0, 10, 32
        n_timesteps, n_features, n_outputs = self.X_train.shape[0], self.X_train.shape[1], self.y_train.shape[0]
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features)))
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
        model.add(Dropout(0.5))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(n_outputs, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # fit network
        model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
        # evaluate model
        _, accuracy = model.evaluate(self.X_test, self.y_test, batch_size=batch_size, verbose=0)

    def plot(self):
        print('Inizio plotting delle metriche...')
        # plotting confusion matrix
        rounded_labels = np.argmax(self.y_test, axis=1)
        y_pred = self.model.predict_classes(self.X_test)

        mat = confusion_matrix(rounded_labels, y_pred)
        plot_confusion_matrix(conf_mat=mat, show_normed=True, figsize=(10, 10))

        plt.figure(figsize=(10, 10))
        array = confusion_matrix(rounded_labels, y_pred)
        df_cm = pd.DataFrame(array, range(6), range(6))
        df_cm.columns = ["Walking", "W_Upstairs", "W_Downstairs", "Sitting", "Standing", "Laying"]
        df_cm.index = ["Walking", "W_Upstairs", "W_Downstairs", "Sitting", "Standing", "Laying"]
        sns.heatmap(df_cm, annot=True, annot_kws={"size": 12},
                    yticklabels=("Walking", "W_Upstairs", "W_Downstairs", "Sitting", "Standing", "Laying"),
                    xticklabels=("Walking", "W_Upstairs", "W_Downstairs", "Sitting", "Standing", "Laying"))  # font size

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


if __name__ == '__main__':
    cnn = CNN()
    cnn.fitWeb()
