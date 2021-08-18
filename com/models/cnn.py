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
                           metrics=METRICS)

        print('Fine creazione.')

    def fit(self):
        print('Inizio fitting del modello...')
        self.checkpoint = ModelCheckpoint(checkPointPathCNN + '/best_model.hdf5',
                                          monitor='val_Accuracy', verbose=1, save_best_only=True, mode='auto', period=1)

        self.history = self.model.fit(self.X_train, self.y_train, batch_size=64, epochs=self.epochs,

                                      validation_data=(self.X_val, self.y_val),
                                      verbose=1, callbacks=[self.checkpoint])
        print('Fine fitting.')


if __name__ == '__main__':
    cnn = CNN()
    cnn.main()
