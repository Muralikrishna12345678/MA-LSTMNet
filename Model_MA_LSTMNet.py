import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from Evaluation import evaluation

def Model_MA_LSTMNet(train_data, train_target, test_data, test_target, Batch=None, sol=None):
    if sol is None:
        sol = [5]
    if Batch is None:
        Batch = 32

    out, model = MA_LSTMNet_train(train_data, train_target, test_data, test_target, sol, Batch)
    pred = np.asarray(out)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    Eval = evaluation(pred, test_target)
    return Eval, pred

# https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
def MA_LSTMNet_train(trainX, trainY, testX, testy, sol = None, Batch=None):
    if sol is None:
        sol = [5]

    if Batch is None:
        Batch = 32
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    model = Sequential()
    model.add(LSTM(int(sol[0]), input_shape=(1, trainX.shape[2])))  # hidden neuron count(5 - 255)
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=5, batch_size=Batch, verbose=2)

    testPredict = model.predict(testX)
    return testPredict, model

