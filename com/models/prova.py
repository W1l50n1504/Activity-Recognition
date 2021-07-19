from com.utility import *
from com.core import *

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Useful Constants

# Those are separate normalised input features for the neural network
INPUT_SIGNAL_TYPES = [
    "body_acc_x_",
    "body_acc_y_",
    "body_acc_z_",
    "body_gyro_x_",
    "body_gyro_y_",
    "body_gyro_z_",
    "total_acc_x_",
    "total_acc_y_",
    "total_acc_z_"
]

# Output classes to learn how to classify
LABELS = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING"
]

TRAIN = "C:/Users\david\PycharmProjects\ActivityRecognition683127\com\dataset/UCI HAR Dataset/train/"
TEST = "C:/Users\david\PycharmProjects\ActivityRecognition683127\com\dataset/UCI HAR Dataset/test/"


# Load "X" (the neural network's training and testing inputs)

def load_X(X_signals_paths):
    X_signals = []

    for signal_type_path in X_signals_paths:
        file = open(signal_type_path, 'r')
        # Read dataset from disk, dealing with text files' syntax
        X_signals.append(
            [np.array(serie, dtype=np.float32) for serie in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]]
        )
        file.close()

    return np.transpose(np.array(X_signals), (1, 2, 0))


X_train_signals_paths = [TRAIN + "Inertial Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES]
X_test_signals_paths = [TEST + "Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES]


# Load "y" (the neural network's training and testing outputs)

def load_y(y_path):
    file = open(y_path, 'r')
    # Read dataset from disk, dealing with text file's syntax
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]],
        dtype=np.int32
    )
    file.close()

    # Substract 1 to each output class for friendly 0-based indexing
    return y_ - 1


def plot_learningCurve(history, epochs):
    # Plot training & validation accuracy values
    epoch_range = range(1, epochs + 1)
    plt.plot(epoch_range, history.history['accuracy'])
    plt.plot(epoch_range, history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

    # Plot training & validation auc values
    epoch_range = range(1, epochs + 1)
    plt.plot(epoch_range, history.history['auc_1'])
    plt.plot(epoch_range, history.history['val_auc_1'])
    plt.title('Model auc')
    plt.ylabel('auc')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(epoch_range, history.history['loss'])
    plt.plot(epoch_range, history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()


y_train_path = TRAIN + "y_train.txt"
y_test_path = TEST + "y_test.txt"

if __name__ == '__main__':
    X, y = loadSavedData()

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    enc = OneHotEncoder(handle_unknown='ignore', sparse=False)

    enc = enc.fit(y_train)

    y_train = enc.transform(y_train)
    y_test = enc.transform(y_test)
    y_val = enc.transform(y_val)

    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, X_val.shape, y_val.shape)

    X_train = X_train.reshape(707582, 4, 1)
    X_test = X_test.reshape(336945, 4, 1)
    X_val = X_val.reshape(78621, 4, 1)

    print(X_train[0].shape, y_train.shape, X_test.shape, y_test.shape, X_val.shape, y_val.shape)

    model = Sequential()
    model.add(Conv2D(64, 1, activation='relu', input_shape=X_train.shape))
    model.add(Dropout(0.1))

    model.add(Conv2D(128, 1, activation='relu', padding='valid'))
    model.add(MaxPool2D(1, 1))

    model.add(Conv2D(256, 1, activation='relu', padding='valid'))
    model.add(MaxPool2D(1, 1))

    model.add(Dropout(0.5))
    model.add(Flatten())

    model.add(Dense(512, activation='relu'))

    model.add(Dense(6, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['accuracy', tf.keras.metrics.AUC()])

    checkpoint = ModelCheckpoint(
        checkPointPathCNN,
        monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto', period=1)

    history = model.fit(X_train, y_train, batch_size=64, epochs=30, validation_data=(X_val, y_val), verbose=1,
                        callbacks=[checkpoint])

    model = load_model(checkPointPathCNN)

    plot_learningCurve(history, 30)

    accuracy = model.evaluate(X_test, y_test, batch_size=16, verbose=1)
    print(accuracy)

    rounded_labels = np.argmax(y_test, axis=1)
    y_pred = model.predict_classes(X_test);

    mat = confusion_matrix(rounded_labels, y_pred);
    plot_confusion_matrix(conf_mat=mat, show_normed=True, figsize=(10, 10));

    plt.figure(figsize=(10, 10))
    array = confusion_matrix(rounded_labels, y_pred);
    df_cm = pd.DataFrame(array, range(6), range(6));
    df_cm.columns = ["Walking", "W_Upstairs", "W_Downstairs", "Sitting", "Standing", "Laying"];
    df_cm.index = ["Walking", "W_Upstairs", "W_Downstairs", "Sitting", "Standing", "Laying"];
    # sn.set(font_scale=1)#for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 12},
               yticklabels=("Walking", "W_Upstairs", "W_Downstairs", "Sitting", "Standing", "Laying"),
               xticklabels=("Walking", "W_Upstairs", "W_Downstairs", "Sitting", "Standing", "Laying"));  # font size
    plt.show();

"""
    X_train = load_X(X_train_signals_paths)
    X_test = load_X(X_test_signals_paths)

    y_train = load_y(y_train_path)
    y_test = load_y(y_test_path)

    X = np.concatenate((X_train, X_test))

    y = np.concatenate((y_train, y_test))

    X.shape, y.shape
"""
