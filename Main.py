import numpy as np
import pandas as pd
import pywt
from numpy import matlib
from scipy.signal import find_peaks
from scipy.stats import hypsecant
import Global_Vars
from CSO import CSO
from DCNN import DCNN
from EOO import EOO
from Model_CNN import Model_CNN
from Model_LSTM import Model_LSTM
from Model_MA_LSTMNet import Model_MA_LSTMNet
from Model_RNN import Model_RNN
from PROPOSED import PROPOSED
from Plot_Results import *
from RDA import RDA
from RSO import RSO
from objfun import objfun_feat, objfun_cls

no_of_dataset = 2


def timedomain(rr):
    results = {}
    hr = 60000 / rr
    results['RMSSD (ms)'] = np.sqrt(np.mean(np.square(np.diff(rr))))
    return results


def harmonicMean(arr, n):
    # Declare sum variables and
    # initialize with zero.
    sm = 0
    for i in range(0, n):
        sm = sm + 1 / arr[i]
    return n / sm


def avg_calc(ls):
    n, mean = len(ls), 0.0
    if n <= 1:
        return ls[0]
    # calculate average
    for el in ls:
        mean = mean + float(el)
    mean = mean / float(n)
    return mean


# Read the dataset 1
an = 0
if an == 1:
    Abnormal = './Dataset/Dataset_1/ptbdb_abnormal.csv'
    Normal = './Dataset/Dataset_1/ptbdb_normal.csv'
    File1 = pd.read_csv(Abnormal)
    File2 = pd.read_csv(Normal)
    Dataset1 = File1.values
    Dataset2 = File2.values
    Dataset = np.append(Dataset1, Dataset2, axis=0)
    index = np.arange(Dataset.shape[0])
    np.random.shuffle(index)
    Dataset = Dataset[index, :]
    Data = Dataset[:, :Dataset.shape[1] - 1]
    Target = Dataset[:, Dataset.shape[1] - 1]
    Target = Target.reshape(-1, 1)
    np.save('Dataset_1.npy', Data)
    np.save('Target_1.npy', Target)

# Read Dataset 2
an = 0
if an == 1:
    Dataset = './Dataset/Dataset_2/Heart-Disease-Prediction-master/dataset.csv'
    Data = pd.read_csv(Dataset)
    Data.drop('target', inplace=True, axis=1)
    data_1 = np.asarray(Data)
    targ = pd.read_csv(Dataset, usecols=['target'])
    tar = np.asarray(targ)
    np.save('Dataset_2.npy', data_1)  # Save the Dataset_2
    np.save('Target_2.npy', tar)  # Save the Target_2

# Pre-processing
an = 0
if an == 1:
    for n in range(1, no_of_dataset):
        Dataset = np.load('Dataset_' + str(n + 1) + '.npy', allow_pickle=True)
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)

        # IQR
        Q1 = np.percentile(Dataset[:, 7], 13, interpolation='midpoint')
        Q3 = np.percentile(Dataset[:, 7], 0, interpolation='midpoint')
        IQR = Q3 - Q1

        if n == 0:
            # Upper bound
            upper = np.where(Dataset[:, 7] >= 13)
            # Lower bound
            lower = np.where(Dataset[:, 7] <= 0)

        else:
            # Upper bound
            upper = np.where(Dataset[:, 12] >= 4)
            # Lower bound
            lower = np.where(Dataset[:, 12] <= 0)

        input = pd.DataFrame(Dataset)
        ''' Removing the Outliers '''
        input.drop(upper[0], inplace=True)
        input.drop(lower[0], inplace=True)

        output = pd.DataFrame(Target)
        ''' Removing the Outliers '''
        output.drop(upper[0], inplace=True)
        output.drop(lower[0], inplace=True)

        '''Data Filling'''
        input.ffill(axis=0)
        input.ffill(axis=1)

        '''Data Filling'''
        output.ffill(axis=0)
        output.ffill(axis=1)

        # normalized_data = stats.boxcox(input)
        normalized_data = input.to_numpy()
        np.save('Preprocess_' + str(n + 1) + '.npy', normalized_data)
        np.save('tar_' + str(n + 1) + '.npy', output)

# Deep feature extraction by DeepCNN
an = 0
if an == 1:
    for n in range(no_of_dataset):
        Data = np.load('Preprocess_' + str(n + 1) + '.npy', allow_pickle=True)
        Target = np.load('tar_' + str(n + 1) + '.npy', allow_pickle=True)
        Eval, Feat = DCNN(Data, Target)
        np.save('Feat_1_' + str(n + 1) + '.npy', Feat)

# Signal processing
an = 0
if an == 1:
    for n in range(no_of_dataset):
        Data = np.load('Preprocess_' + str(n + 1) + '.npy', allow_pickle=True)
        data_count = len(Data)
        ECG_peak = [find_peaks(Data[i], height=0)[0] for i in range(len(Data))]
        length = len(max(ECG_peak, key=len))
        ECG_Peak = np.asarray([np.append(x, [0] * (length - len(x))) for x in ECG_peak])
        ECG_harmonic = np.asarray([harmonicMean(Data[i], len(Data[i])) for i in range(len(Data))]).reshape(-1, 1)
        ECG_zero_crosses_rate = [np.nonzero(np.diff(Data[i] > 0))[0] for i in range(len(Data))]
        length = len(max(ECG_zero_crosses_rate, key=len))
        ECG_zero_crosses_rate = np.asarray([np.append(x, [0] * (length - len(x))) for x in ECG_zero_crosses_rate])
        ECG_Entropy = np.asarray([hypsecant.entropy(Data[i]) for i in range(len(Data))])
        ECG_STD = np.asarray([np.std(Data[i]) for i in range(len(Data))]).reshape(-1, 1)
        ECG_RMSSD = np.asarray([timedomain(Data[i])['RMSSD (ms)'] for i in range(len(Data))]).reshape(-1, 1)
        EEG = np.random.randint(low=-100, high=2000, size=(data_count, 3400))  # EEG
        EEG_peak = [find_peaks(Data[i], height=0)[0] for i in range(len(Data))]
        length = len(max(EEG_peak, key=len))
        EEG_Peak = np.asarray([np.append(x, [0] * (length - len(x))) for x in EEG_peak])
        EEG_harmonic = np.asarray([harmonicMean(Data[i], len(Data[i])) for i in range(len(Data))]).reshape(-1, 1)
        EEG_zero_crosses_rate = [np.nonzero(np.diff(Data[i] > 0))[0] for i in range(len(Data))]
        length = len(max(ECG_zero_crosses_rate, key=len))
        EEG_zero_crosses_rate = np.asarray([np.append(x, [0] * (length - len(x))) for x in EEG_zero_crosses_rate])
        EEG_Entropy = np.asarray([hypsecant.entropy(Data[i]) for i in range(len(Data))])
        EEG_STD = np.asarray([np.std(Data[i]) for i in range(len(Data))]).reshape(-1, 1)
        EEG_RMSSD = np.asarray([timedomain(Data[i])['RMSSD (ms)'] for i in range(len(Data))]).reshape(-1, 1)
        DWT = np.zeros((data_count, 107))
        for j in range(data_count):
            print('dwt ', j)
            (cA, cD) = pywt.dwt(EEG[j, :], 'db1')
            (cA, cD) = pywt.dwt(cD, 'db1')
            (cA, cD) = pywt.dwt(cD, 'db1')
            (cA, cD) = pywt.dwt(cD, 'db1')
            (cA, cD) = pywt.dwt(cD, 'db1')
            DWT[j, :] = cD
        feat_ecg = np.append(ECG_Peak, np.append(ECG_harmonic, np.append(ECG_zero_crosses_rate, np.append(ECG_Entropy,
                                                                                                          np.append(
                                                                                                              ECG_STD,
                                                                                                              ECG_RMSSD,
                                                                                                              axis=1),
                                                                                                          axis=1),
                                                                         axis=1), axis=1), axis=1)
        feat_eeg = np.append(DWT, np.append(EEG_Peak,
                                            np.append(EEG_harmonic,
                                                      np.append(EEG_zero_crosses_rate, np.append(EEG_Entropy,
                                                                                                 np.append(
                                                                                                     EEG_STD,
                                                                                                     EEG_RMSSD,
                                                                                                     axis=1),
                                                                                                 axis=1),
                                                                axis=1), axis=1), axis=1), axis=1)

        Attributes = ['Name', 'Age', 'Sex', 'Heartbeat', 'Pressure', 'IsTired', 'IsDiaria', 'Stress',
                      'Temperature', 'Blood Test', 'Urine Test']
        Name = np.arange(data_count).reshape(-1, 1)
        Sex = np.random.randint(low=1, high=3, size=(data_count, 1))  # 1 -> Female ....   2 -> Male ....
        Age = np.random.randint(low=1, high=70, size=(data_count, 1))
        Heartbeat = np.random.randint(low=60, high=80, size=(data_count, 1))  # Normal 72
        Pressure = np.random.randint(low=80, high=120, size=(data_count, 1))  # Normal (80 - 120)
        IsTired = np.random.randint(low=0, high=2, size=(data_count, 1))
        IsDiaria = np.random.randint(low=0, high=2, size=(data_count, 1))
        Stress = np.random.randint(low=50, high=100, size=(data_count, 1))
        Temperature = np.random.randint(low=90, high=120, size=(data_count, 1))
        Blood_Test = ((1.5 - (-1.5)) * np.random.random(size=(data_count, 3))) + (-1.5)
        Urine_Test = ((1.5 - (-1.5)) * np.random.random(size=(data_count, 18))) + (-1.5)
        feat_data = np.append(Name,
                              np.append(Sex, np.append(Age, np.append(Heartbeat, np.append(Pressure, np.append(IsTired,
                                                                                                               np.append(
                                                                                                                   IsDiaria,
                                                                                                                   np.append(
                                                                                                                       Stress,
                                                                                                                       Temperature,
                                                                                                                       axis=1),
                                                                                                                   axis=1),
                                                                                                               axis=1),
                                                                                           axis=1), axis=1), axis=1),
                                        axis=1), axis=1)
        Feat = np.append(feat_data, np.append(feat_ecg, feat_eeg, axis=1), axis=1)
        np.save('Feat_2_' + str(n + 1) + '.npy', Feat)


# Optimization for Feature Selection And Weight Optimization
an = 0
if an == 1:
    for n in range(no_of_dataset):
        Feat_1 = np.load('Feat_1_' + str(n + 1) + '.npy', allow_pickle=True)
        Feat_2 = np.load('Feat_2_' + str(n + 1) + '.npy', allow_pickle=True)
        Target = np.load('tar_' + str(n + 1) + '.npy', allow_pickle=True)
        Global_Vars.Feat_1 = Feat_1
        Global_Vars.Feat_2 = Feat_2
        Global_Vars.Target = Target
        Npop = 10
        Chlen = 4 * 25  # (25 + 25) for feature 1 and 2 and (25 + 25) for weights for feature 1 and 2
        xmin = matlib.repmat(np.append(1 * np.ones(25), np.append(1 * np.ones(25), np.append(0.01 * np.ones(25), 0.01 * np.ones(25), axis=0), axis=0), axis=0), Npop, 1)
        xmax = matlib.repmat(np.append(Feat_1.shape[1] - 1 * np.ones(25), np.append(Feat_2.shape[1] - 1 * np.ones(25), np.append(0.99 * np.ones(25), 0.99 * np.ones(25), axis=0), axis=0), axis=0), Npop, 1)

        fname = objfun_feat
        initsol = np.zeros((Npop, Chlen))
        for p1 in range(initsol.shape[0]):
            for p2 in range(initsol.shape[1]):
                initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
        Max_iter = 50

        print("EOO...")
        [bestfit1, fitness1, bestsol1, time1] = EOO(initsol, fname, xmin, xmax, Max_iter)  # EOO

        print("CSO...")
        [bestfit2, fitness2, bestsol2, time2] = CSO(initsol, fname, xmin, xmax, Max_iter)  # CSO

        print("RSO...")
        [bestfit4, fitness4, bestsol4, time3] = RSO(initsol, fname, xmin, xmax, Max_iter)  # RSO

        print("RDA...")
        [bestfit3, fitness3, bestsol3, time4] = RDA(initsol, fname, xmin, xmax, Max_iter)  # RDA

        print("Proposed...")
        [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)  # PROPOSED

        BestSol_feat = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
        np.save('BestSol_FEAT_' + str(n + 1) + '.npy', BestSol_feat)  # Save the BestSol_FEAT

# feature Concatenation for Feature Fusion
an = 0
if an == 1:
    for i in range(no_of_dataset):
        Feat_1 = np.load('Feat_1_' + str(i + 1) + '.npy', allow_pickle=True)  # Load the Feat 1
        Feat_2 = np.load('Feat_2_' + str(i + 1) + '.npy', allow_pickle=True)  # Load the Feat 2
        bests = np.load('BestSol_FEAT_' + str(i + 1) + '.npy', allow_pickle=True)  # Load the Bestsol Feat
        sol_Features = np.round(bests[4, :50]).astype(np.int16)
        Sol_Weight = bests[4, 50:]
        F1 = Feat_1[:, sol_Features[:25]]
        F2 = Feat_2[:, sol_Features[25:]]
        Weighted_Feat_1 = F1 * Sol_Weight[:25]
        Weighted_Feat_2 = F2 * Sol_Weight[25:]
        Feature = np.concatenate((Weighted_Feat_1, Weighted_Feat_2), axis=1)
        np.save('Feature_' + str(i + 1) + '.npy', Feature)  # Save the Feature

# optimization for Classification
an = 0
if an == 1:
    BEST_Solution = []
    Fitness = []
    for n in range(no_of_dataset):
        Feat = np.load('Feature_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the Selected features
        Target = np.load('tar_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the Target
        Global_Vars.Feat = Feat
        Global_Vars.Target = Target
        Npop = 10
        Chlen = 100  # hidden neuron count in  MA-LSTMNet
        xmin = matlib.repmat([5], Npop, Chlen)
        xmax = matlib.repmat([255], Npop, Chlen)
        fname = objfun_cls
        initsol = np.zeros((Npop, Chlen))
        for p1 in range(initsol.shape[0]):
            for p2 in range(initsol.shape[1]):
                initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
        Max_iter = 50

        print("EOO...")
        [bestfit1, fitness1, bestsol1, time1] = EOO(initsol, fname, xmin, xmax, Max_iter)  # EOO

        print("CSO...")
        [bestfit2, fitness2, bestsol2, time2] = CSO(initsol, fname, xmin, xmax, Max_iter)  # CSO

        print("RSO...")
        [bestfit4, fitness4, bestsol4, time3] = RSO(initsol, fname, xmin, xmax, Max_iter)  # RSO

        print("RDA...")
        [bestfit3, fitness3, bestsol3, time4] = RDA(initsol, fname, xmin, xmax, Max_iter)  # RDA

        print("PROPOSED...")
        [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)  # Hybrid RSO & RDA

        BestSol_CLS = [bestsol1.squeeze(), bestsol2.squeeze(), bestsol3.squeeze(), bestsol4.squeeze(),
                       bestsol5.squeeze()]
        fitness = [fitness1.squeeze(), fitness2.squeeze(), fitness3.squeeze(), fitness4.squeeze(), fitness5.squeeze()]
        BEST_Solution.append(BestSol_CLS)
        Fitness.append(fitness)
    np.save('Fitness.npy', np.asarray(Fitness))
    np.save('BestSol_CLS.npy', np.asarray(BEST_Solution))

# Classification Batch size
an = 0
if an == 1:
    Eval_all = []
    for n in range(no_of_dataset):
        Feature = np.load('Feature_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the Selected features
        Target = np.load('tar_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the Target
        BestSol = np.load('BestSol_CLS.npy', allow_pickle=True)[n]  # Load the Bestsol Classification
        Feat = Feature
        EVAL = []
        Batch_size = [4, 8, 16, 32, 48, 64]
        for learn in range(len(Batch_size)):
            Batch_sizec = round(Feat.shape[0] * Batch_size[learn])
            Train_Data = Feat[:Batch_sizec, :]
            Train_Target = Target[:Batch_sizec, :]
            Test_Data = Feat[Batch_sizec:, :]
            Test_Target = Target[Batch_sizec:, :]
            Eval = np.zeros((10, 14))
            for j in range(BestSol.shape[0]):
                print(learn, j)
                sol = np.round(BestSol[j, :]).astype(np.int16)
                Eval[j, :], pred1 = Model_MA_LSTMNet(Train_Data, Train_Target, Test_Data, Test_Target, Batch_size[learn], sol=sol)
            Eval[5, :], pred2 = Model_RNN(Train_Data, Train_Target, Test_Data, Test_Target, Batch_size[learn])
            Eval[6, :], pred3 = Model_CNN(Train_Data, Train_Target, Test_Data, Test_Target, Batch_size=Batch_size[learn])
            Eval[7, :], pred4 = Model_LSTM(Train_Data, Train_Target, Test_Data, Test_Target, Batch_size[learn])
            Eval[8, :], pred5 = Model_MA_LSTMNet(Train_Data, Train_Target, Test_Data, Test_Target, Batch_size[learn])
            Eval[9, :], pred6 = Eval[4, :]
            EVAL.append(Eval)
        Eval_all.append(EVAL)
    np.save('Evaluate_all.npy', np.asarray(Eval_all))  # Save the Eval all


# KFOLD - Classification
an = 0
if an == 1:
    Eval_all = []
    for n in range(no_of_dataset):
        Feature = np.load('Feature_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the Selected features
        Target = np.load('tar_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the Target
        BestSol = np.load('BestSol_CLS.npy', allow_pickle=True)[n]  # Load the Bestsol Classification
        K = 5
        Per = 1 / 5
        Perc = round(Feature.shape[0] * Per)
        eval = []
        for i in range(K):
            Eval = np.zeros((10, 14))
            Feat = Feature
            Test_Data = Feat[i * Perc: ((i + 1) * Perc), :]
            Test_Target = Target[i * Perc: ((i + 1) * Perc), :]
            test_index = np.arange(i * Perc, ((i + 1) * Perc))
            total_index = np.arange(Feat.shape[0])
            train_index = np.setdiff1d(total_index, test_index)
            Train_Data = Feat[train_index, :]
            Train_Target = Target[train_index, :]
            for j in range(BestSol.shape[0]):
                sol = np.round(BestSol[j, :]).astype(np.int16)
                Eval[j, :], pred = Model_MA_LSTMNet(Train_Data, Train_Target, Test_Data, Test_Target, sol=sol)
            Eval[5, :], pred_1 = Model_RNN(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[6, :], pred_2 = Model_CNN(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[7, :], pred_3 = Model_LSTM(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[8, :], pred_4 = Model_MA_LSTMNet(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[9:], pred_5 = Eval[4, :]
            eval.append(Eval)
        Eval_all.append(eval)
    np.save('Eval_all_KFold_1.npy', np.asarray(Eval_all))

plot_Results()
plotConvResults()
Plot_ROC_Curve()
Plot_Confusion()
