import cv2 as cv
import numpy as np
# import tflearn
# from tensorflow.python.framework import ops
# from tflearn.layers.conv import conv_2d, max_pool_2d
# from tflearn.layers.core import input_data, dropout, fully_connected
# from tflearn.layers.estimator import regression
# from Evaluation import evaluation


# def DCNN(Data, Target, sol=None):
#     if sol is None:
#         sol = [128, 5, 0.01]
#     IMG_SIZE = 20
#     Train_X = np.zeros((Data.shape[0], IMG_SIZE, IMG_SIZE, 1))
#     for i in range(Data.shape[0]):
#         temp = np.resize(Data[i], (IMG_SIZE * IMG_SIZE, 1))
#         Train_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 1))
#
#     Test_X = np.zeros((Data.shape[0], IMG_SIZE, IMG_SIZE, 1))
#     for i in range(Data.shape[0]):
#         temp = np.resize(Data[i], (IMG_SIZE * IMG_SIZE, 1))
#         Test_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 1))
#     pred, weight = Model(Train_X, Target, sol)
#
#     pred = np.asarray(pred)
#     Eval = evaluation(pred, Target)
#     feat = np.asarray(weight)
#     feat = np.reshape(feat, (feat.shape[0] * feat.shape[1], feat.shape[2] * feat.shape[3]))
#     feat = np.resize(feat, (Data.shape[0], 1000))
#
#     return Eval, feat
#
#
# def Model(X, Y, sol):
#     LR = 1e-3
#     ops.reset_default_graph()
#     convnet = input_data(shape=[None, 20, 20, 1], name='input')
#
#     convnet = conv_2d(convnet, 32, 5, name='layer-conv1', activation='linear')
#     convnet = max_pool_2d(convnet, 5)
#
#     convnet = conv_2d(convnet, 64, 5, name='layer-conv2', activation='linear')
#     convnet = max_pool_2d(convnet, 5)
#
#     convnet = conv_2d(convnet, round(sol[0]), 5, name='layer-conv3', activation='linear')
#     convnet = max_pool_2d(convnet, 5)
#
#     convnetc = conv_2d(convnet, 64, 5, name='layer-conv4', activation='linear')
#     convnet = max_pool_2d(convnetc, 5)
#
#     convnet = conv_2d(convnet, 32, 5, name='layer-conv5', activation='linear')
#     convnet = max_pool_2d(convnet, 5)
#
#     convnet1 = fully_connected(convnet, 1024, name='layer-conv', activation='linear')
#     convnet2 = dropout(convnet1, 0.8)
#
#     convnet3 = fully_connected(convnet2, Y.shape[1], name='layer-conv-before-softmax', activation='linear')
#
#     regress = regression(convnet3, optimizer='sgd', learning_rate=sol[2],
#                          loss='mean_square', name='target')
#
#     model = tflearn.DNN(regress, tensorboard_dir='log')
#
#     MODEL_NAME = 'test.model'.format(LR, '6conv-basic')
#     model.fit({'input': X}, {'target': Y}, n_epoch=round(sol[1]),
#               validation_set=({'input': X}, {'target': Y}),
#               snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
#
#     pred = model.predict(X)
#     weight = model.get_weights(convnetc.W)
#     return pred, weight


from keras import layers, models
import numpy as np
from Evaluation import evaluation


def Model(X, Y, test_x, test_y, Batch_size=None):
    if Batch_size is None:
        Batch_size = 4

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.summary()
    model.add(layers.Flatten())
    model.add(layers.Dense(Y.shape[1]))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, Y, epochs=10, batch_size=Batch_size, validation_data=(test_x, test_y))
    pred = model.predict(test_x)
    Weights = model.layers[-1].get_weights()[0]
    Data = np.concatenate((X, test_x), axis=0 )
    Feature = np.resize(Weights, (Data.shape[0], Data.shape[1]))

    return pred, Feature


def DCNN(Data, Target, Batch_size=None):
    if Batch_size is None:
        Batch_size = 4
    per = round(Data.shape[0] * 0.75)
    train_data = Data[:per, :]
    test_data = Data[per:, :]
    train_target = Target[:per, :]
    test_target = Target[per:, :]
    IMG_SIZE = 32

    Train_X = np.zeros((train_data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(train_data.shape[0]):
        temp = np.resize(train_data[i], (IMG_SIZE * IMG_SIZE, 3))
        Train_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))

    Test_X = np.zeros((test_data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(test_data.shape[0]):
        temp = np.resize(test_data[i], (IMG_SIZE * IMG_SIZE, 3))
        Test_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))
    pred, feat = Model(Train_X, train_target, Test_X, test_target, Batch_size)
    Eval = evaluation(pred, test_target)
    return Eval, feat
