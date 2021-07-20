from com.core import *
from com.utility import *

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


class BLSTM(BaseModel, ABC):
    def __init__(self):
        super().__init__()

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
                                      validation_data=(self.X_val, self.y_val), verbose=1,
                                      callbacks=[self.checkpoint])


if __name__ == '__main__':
    blstm = BLSTM()
    blstm.main()
