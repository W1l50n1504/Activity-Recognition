from com.utility import *
from com.core import *

if __name__ == '__main__':
    x, y = loadUCIHAR()

    x = np.array(x)
    y = np.array(y)

    print(x.shape)
    print(y.shape)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    print(X_train[0].shape)
    print(y_train.shape)

    n_timesteps, n_features, n_outputs = X_train.shape[0], X_train.shape[1], y_train.shape[1]
    verbose, epochs, batch_size = 0, 10, 8

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
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    _, accuracy = model.evaluate(X_train, y_train, batch_size=batch_size, verbose=0)

    print('accuracy:', accuracy)
