import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
from sklearn.metrics import roc_curve
from itertools import cycle
import seaborn as sns
from sklearn.metrics import confusion_matrix

no_of_dataset = 2


def plot_kfold_results():
    eval = np.load('Eval_all_KFold_1.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Graph_Term = [0, 3, 4, 5, 9]
    Algorithm = ['TERMS', 'EOO-MA-LSTMNet', 'CSO-MA-LSTMNet', 'RSO-MA-LSTMNet', 'RDA-MA-LSTMNet', 'HRS-RDOA-MA-LSTMNet']
    Classifier = ['TERMS', 'RNN', 'CNN', 'LSTM', 'MA-LSTMNet', 'HRS-RDOA-MA-LSTMNet']

    learnper = [1, 2, 3, 4, 5]
    for i in range(eval.shape[0]):
        for j in range(len(Graph_Term)):
            Graph = np.zeros(eval.shape[1:3])
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    if j == 9:
                        Graph[k, l] = eval[i, k, l, Graph_Term[j] + 4]
                    else:
                        Graph[k, l] = eval[i, k, l, Graph_Term[j] + 4] * 100

            plt.plot(learnper, Graph[:, 0], color='r', linewidth=3, marker='o', markerfacecolor='blue', markersize=12,
                     label="EOO-MA-LSTMNet")
            plt.plot(learnper, Graph[:, 1], color='g', linewidth=3, marker='o', markerfacecolor='red', markersize=12,
                     label="CSO-MA-LSTMNet")
            plt.plot(learnper, Graph[:, 2], color='b', linewidth=3, marker='o', markerfacecolor='green', markersize=12,
                     label="RSO-MA-LSTMNet")
            plt.plot(learnper, Graph[:, 3], color='c', linewidth=3, marker='o', markerfacecolor='yellow', markersize=12,
                     label="RDA-MA-LSTMNet")
            plt.plot(learnper, Graph[:, 4], color='k', linewidth=3, marker='o', markerfacecolor='cyan', markersize=12,
                     label="HRS-RDOA-MA-LSTMNet")
            plt.xticks(learnper, ('1', '2', '3', '4', '5'))
            plt.xlabel('K-Fold')
            plt.ylabel(Terms[Graph_Term[j]])
            plt.legend(loc=4)
            plt.tight_layout()
            path1 = "./Results/Dataset_%s_K-Fold_%s_line.png" % (i + 1, Terms[Graph_Term[j]])
            plt.savefig(path1)
            plt.show()

            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            X = np.arange(5)

            ax.bar(X + 0.00, Graph[:, 5], color='r', width=0.10, label="RNN")
            ax.bar(X + 0.10, Graph[:, 6], color='g', width=0.10, label="CNN")
            ax.bar(X + 0.20, Graph[:, 7], color='b', width=0.10, label="LSTM")
            ax.bar(X + 0.30, Graph[:, 8], color='m', width=0.10, label="MA-LSTMNet")
            ax.bar(X + 0.40, Graph[:, 4], color='k', width=0.10, label="HRS-RDOA-MA-LSTMNet")
            plt.xticks(X + 0.10, ('1', '2', '3', '4', '5'))
            plt.xlabel('K-Fold')
            plt.ylabel(Terms[Graph_Term[j]])
            plt.legend(loc=1)
            path1 = "./Results/Dataset_%s_K-Fold_%s_bar.png" % (i + 1, Terms[Graph_Term[j]])
            plt.savefig(path1)
            plt.show()


def Plot_Confusion():
    Actual = np.load('Actual.npy', allow_pickle=True)
    Predict = np.load('Predict.npy', allow_pickle=True)
    for n in range(no_of_dataset):
        ax = plt.subplot()
        cm = confusion_matrix(np.asarray(Actual[n]), np.asarray(Predict[n]))
        sns.heatmap(cm, annot=True, fmt='g',
                    ax=ax)
        plt.title('Accuracy')
        # path = "./Results/Confusion_%s.png" % (n + 1)
        # plt.savefig(path)
        plt.show()


def plot_Results():
    eval = np.load('Evaluate_all.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Graph_Term = [0, 3, 4, 5, 9]

    Algorithm = ['TERMS', 'EOO-MA-LSTMNet', 'CSO-MA-LSTMNet', 'RSO-MA-LSTMNet', 'RDA-MA-LSTMNet', 'HRS-RDOA-MA-LSTMNet']
    Classifier = ['TERMS', 'RNN', 'CNN', 'LSTM', 'MA-LSTMNet', 'HRS-RDOA-MA-LSTMNet']
    for i in range(eval.shape[0]):
        value = eval[i, 4, :, 4:]

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], value[j, :])
        print('-------------------------------------------------- Dataset - ', i + 1, ' - 75 % - Algorithm Comparison ',
              '--------------------------------------------------')
        print(Table)

        Table = PrettyTable()
        Table.add_column(Classifier[0], Terms)
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value[len(Algorithm) + j - 1, :])
        print('-------------------------------------------------- Dataset - ', i + 1, ' - 75 % - Classifier Comparison',
              '--------------------------------------------------')
        print(Table)

    Batch_size = [4, 8, 16, 32, 48, 64]
    for i in range(eval.shape[0]):
        for j in range(len(Graph_Term)):
            Graph = np.zeros(eval.shape[1:3])
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    if j == 9:
                        Graph[k, l] = eval[i, k, l, Graph_Term[j] + 4]
                    else:
                        Graph[k, l] = eval[i, k, l, Graph_Term[j] + 4] * 100

            plt.plot(Batch_size, Graph[:, 0], color='r', linewidth=3, marker='o', markerfacecolor='blue', markersize=12,
                     label="EOO-MA-LSTMNet")
            plt.plot(Batch_size, Graph[:, 1], color='g', linewidth=3, marker='o', markerfacecolor='red', markersize=12,
                     label="CSO-MA-LSTMNet")
            plt.plot(Batch_size, Graph[:, 2], color='b', linewidth=3, marker='o', markerfacecolor='green', markersize=12,
                     label="RSO-MA-LSTMNet")
            plt.plot(Batch_size, Graph[:, 3], color='c', linewidth=3, marker='o', markerfacecolor='yellow', markersize=12,
                     label="RDA-MA-LSTMNet")
            plt.plot(Batch_size, Graph[:, 4], color='k', linewidth=3, marker='o', markerfacecolor='cyan', markersize=12,
                     label="HRS-RDOA-MA-LSTMNet")
            plt.xticks(Batch_size, ('4', '8', '16', '32', '48', '64'))
            plt.xlabel('Batch size')
            plt.ylabel(Terms[Graph_Term[j]])
            plt.legend(loc=4)
            plt.tight_layout()
            path1 = "./Results/Dataset_%s_Batch_size_%s_line.png" % (i + 1, Terms[Graph_Term[j]])
            plt.savefig(path1)
            plt.show()

            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            X = np.arange(6)
            ax.bar(X + 0.00, Graph[:, 5], color='r', width=0.10, label="RNN")
            ax.bar(X + 0.10, Graph[:, 6], color='g', width=0.10, label="CNN")
            ax.bar(X + 0.20, Graph[:, 7], color='b', width=0.10, label="LSTM")
            ax.bar(X + 0.30, Graph[:, 8], color='m', width=0.10, label="MA-LSTMNet")
            ax.bar(X + 0.40, Graph[:, 4], color='k', width=0.10, label="HRS-RDOA-MA-LSTMNet")
            plt.xticks(X + 0.10, ('4', '8', '16', '32', '48', '64'))
            plt.xlabel('Batch size')
            plt.ylabel(Terms[Graph_Term[j]])
            plt.legend(loc=1)
            path1 = "./Results/Dataset_%s_Batch_size_%s_bar.png" % (i + 1, Terms[Graph_Term[j]])
            plt.savefig(path1)
            plt.show()

def Statistical(data):
    Min = np.min(data)
    Max = np.max(data)
    Mean = np.mean(data)
    Median = np.median(data)
    Std = np.std(data)
    return np.asarray([Min, Max, Mean, Median, Std])


def plotConvResults():
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'EOO-MA-LSTMNet', 'CSO-MA-LSTMNet', 'RSO-MA-LSTMNet', 'RDA-MA-LSTMNet',
                 'HRS-RDOA-MA-LSTMNet']

    Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    for n in range(no_of_dataset):
        Conv_Graph = np.zeros((5, 5))
        for j in range(5):
            Conv_Graph[j, :] = Statistical(Fitness[n, j, :])

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('-------------------------------------------------- Dataset - ', n + 1, 'Statistical Report ',
              '--------------------------------------------------')
        print(Table)

        length = np.arange(25)
        Conv_Graph = Fitness[n]

        plt.plot(length, Conv_Graph[0, :], color='r', linewidth=3, marker='*', markerfacecolor='red',
                 markersize=12, label='EOO-MA-LSTMNet')
        plt.plot(length, Conv_Graph[1, :], color='g', linewidth=3, marker='*', markerfacecolor='green',
                 markersize=12, label='CSO-MA-LSTMNet')
        plt.plot(length, Conv_Graph[2, :], color='b', linewidth=3, marker='*', markerfacecolor='blue',
                 markersize=12, label='RSO-MA-LSTMNet')
        plt.plot(length, Conv_Graph[3, :], color='m', linewidth=3, marker='*', markerfacecolor='magenta',
                 markersize=12, label='RDA-MA-LSTMNet')
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
                 markersize=12, label='HRS-RDOA-MA-LSTMNet')
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=1)
        plt.savefig("./Results/Dataset_%s_Convergence.png" % (n + 1))
        plt.show()


def Plot_ROC_Curve():
    lw = 2
    cls = ['RNN', 'CNN', 'LSTM', 'MA-LSTMNet', 'HRS-RDOA-MA-LSTMNet']
    for a in range(no_of_dataset):
        Actual = np.load('tar_' + str(a + 1) + '.npy', allow_pickle=True)
        if a == 1:
            Actual = np.load('Targets_' + str(a + 1) + '.npy', allow_pickle=True)
        colors = cycle(["green", "blue", "darkorange", "red", "black"])
        for i, color in zip(range(len(cls)), colors):  # For all classifiers
            Predicted = np.load('Y_Score.npy', allow_pickle=True)[a][i]
            false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual.ravel(), Predicted.ravel())
            plt.plot(
                false_positive_rate1,
                true_positive_rate1,
                color=color,
                lw=lw,
                label=cls[i], )

        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title('Accuracy')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        path1 = "./Results/Dataset_%s_ROC.png" % (a + 1)
        plt.savefig(path1)
        plt.show()


if __name__ == '__main__':
    plot_kfold_results()
    plot_Results()
    plotConvResults()
    Plot_ROC_Curve()
    Plot_Confusion()
