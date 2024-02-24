import numpy as np
from Global_Vars import Global_Vars
from Model_MA_LSTMNet import Model_MA_LSTMNet

def objfun_feat(Soln):
    Feat_1 = Global_Vars.Feat_1
    Feat_2 = Global_Vars.Feat_2
    Tar = Global_Vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            sol_Features = np.round(Soln[:50]).astype(np.int16)
            Sol_Weight = Soln[50:]
            F1 = Feat_1[:, sol_Features[:25]]
            F2 = Feat_2[:, sol_Features[25:]]
            Weighted_Feat_1 = F1 * Sol_Weight[:25]
            Weighted_Feat_2 = F2 * Sol_Weight[25:]
            Feature = np.concatenate((Weighted_Feat_1, Weighted_Feat_2), axis=1)
            Corr = np.mean(np.corrcoef(Feature))
            Fitn[i] = 1 / Corr
        return Fitn
    else:
        sol_Features = np.round(Soln[:50]).astype(np.int16)
        Sol_Weight = Soln[50:]
        F1 = Feat_1[:, sol_Features[:25]]
        F2 = Feat_2[:, sol_Features[25:]]
        Weighted_Feat_1 = F1 * Sol_Weight[:25]
        Weighted_Feat_2 = F2 * Sol_Weight[25:]
        Feature = np.concatenate((Weighted_Feat_1, Weighted_Feat_2), axis=1)
        Corr = np.mean(np.corrcoef(Feature))
        Fitn = 1 / Corr
        return Fitn


def objfun_cls(Soln):
    Feat = Global_Vars.Feat
    Tar = Global_Vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        learnper = round(Feat.shape[0] * 0.75)
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            Train_Data = Feat[:learnper, :]
            Train_Target = Tar[:learnper, :]
            Test_Data = Feat[learnper:, :]
            Test_Target = Tar[learnper:, :]
            Eval = Model_MA_LSTMNet(Train_Data, Train_Target, Test_Data, Test_Target, sol)
            Fitn = 1 / (Eval[4]) + Eval[8] + Eval[9]
        return Fitn
    else:
        learnper = round(Feat.shape[0] * 0.75)
        sol = np.round(Soln).astype(np.int16)
        Train_Data = Feat[:learnper, :]
        Train_Target = Tar[:learnper, :]
        Test_Data = Feat[learnper:, :]
        Test_Target = Tar[learnper:, :]
        Eval = Model_MA_LSTMNet(Train_Data, Train_Target, Test_Data, Test_Target, sol)
        Fitn = 1 / (Eval[4]) + Eval[8] + Eval[9]
        return Fitn



