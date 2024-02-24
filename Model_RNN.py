import numpy as np
from keras import layers, models
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from Evaluation import evaluation


def Model_RNN(train_data, train_target, test_data, test_target, BS=None, sol=None):
    if sol is None:
        sol = [5, 5]
    if BS is None:
        BS = 32
    out, model = RNN_train(train_data, train_target, test_data, test_target, BS=BS, sol=sol)  # RNN
    out = np.reshape(out, test_target.shape)
    pred = np.round(out)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    Eval = evaluation(pred, test_target)
    return Eval, pred


def RNN_train(trainX, trainY, testX, testY, BS, sol=None):
    if sol is None:
        sol = [256, 5]

    input_shape = (5, 32, 32, 3)
    num_classes = testY.shape[-1]
    # trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    # testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    IMG_SIZE = 32
    Train_X = np.zeros((trainX.shape[0], 5, IMG_SIZE, IMG_SIZE, 3))
    for i in range(trainX.shape[0]):
        temp = np.resize(trainX[i], (5, IMG_SIZE * IMG_SIZE, 3))
        Train_X[i] = np.reshape(temp, (5, IMG_SIZE, IMG_SIZE, 3))

    Test_X = np.zeros((testX.shape[0], 5, IMG_SIZE, IMG_SIZE, 3))
    for i in range(testX.shape[0]):
        temp = np.resize(testX[i], (5, IMG_SIZE * IMG_SIZE, 3))
        Test_X[i] = np.reshape(temp, (5, IMG_SIZE, IMG_SIZE, 3))

    Train_X = tf.constant(Train_X, dtype=tf.float32)
    trainY = tf.constant(trainY, dtype=tf.float32)
    Test_X = tf.constant(Test_X, dtype=tf.float32)
    testY = tf.constant(testY, dtype=tf.float32)

    model = models.Sequential()

    # Assuming input_shape is (time_steps, height, width, channels)
    model.add(layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu'), input_shape=input_shape))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2, 2))))
    model.add(layers.TimeDistributed(layers.Flatten()))
    model.add(layers.LSTM(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    model.summary()
    model.fit(Train_X, trainY, epochs=5, batch_size=BS, verbose=2, steps_per_epoch=32, validation_data=(Test_X, testY))
    testPredict = model.predict(Test_X)
    return testPredict, model



