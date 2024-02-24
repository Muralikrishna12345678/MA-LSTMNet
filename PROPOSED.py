import numpy as np
import time
import random as rn


def SortArray(Matrix):
    Output = np.zeros(Matrix.shape)
    Mean = np.zeros(Matrix.shape[0])
    for i in range(Matrix.shape[0]):
        Mean[i] = np.mean(Matrix[i, :])
    index = np.argsort(Mean)
    index = index[::-1]
    for i in range(Matrix.shape[0]):
        Output[i, :] = Matrix[index[i], :]
    return Output, index


def SortPosition(Matrix, index):
    Output = np.zeros(Matrix.shape)
    for i in range(Matrix.shape[0]):
        Output[i, :] = Matrix[index[i], :]
    return Output


def Check_Bounds(s, lb, ub):
    for i in range(s.shape[0]):
        if s[i] > ub[i]:
            s[i] = ub[i]
        if s[i] < lb[i]:
            s[i] = lb[i]
    return s

def PROPOSED(Positions, objective, Lower_bound, Upper_bound, Max_iterations):
    Search_Agents, dimension = Positions.shape
    Position = np.zeros((1, dimension))
    Score = np.inf
    Convergence = np.zeros((Max_iterations, 1))
    l = 0
    x = 1
    y = 5
    R = int(np.floor(np.multiply((y - x), np.random.rand(1, 1)) + x))
    Fit = np.zeros((Search_Agents, dimension))
    Prop_Position = np.copy(Positions)
    Fnew = np.zeros((Search_Agents, dimension))
    V = np.zeros((Search_Agents, dimension))
    P = np.zeros((Search_Agents, dimension))
    offs = np.zeros((Search_Agents, dimension))
    Distance = np.zeros(Search_Agents)
    N_harem = np.zeros((Search_Agents, dimension))
    for i in range(Search_Agents):
        Fit[i, :] = objective(Positions[i, :])
    ct = time.time()

    while l < Max_iterations:
        # Using RSO
        for i in range(Positions.shape[1 - 1]):
            Flag4Upper_bound = Positions[i, :] > Upper_bound[i, :]
            Flag4Lower_bound = Positions[i, :] < Lower_bound[i, :]
            Positions[i, :] = (np.multiply(Positions[i, :], (~ (Flag4Upper_bound + Flag4Lower_bound)))) + np.multiply(
                Upper_bound[i, :], Flag4Upper_bound) + np.multiply(Lower_bound[i, :], Flag4Lower_bound)

            fitness = objective(Positions[i, :])
            if fitness < Score:
                Score = fitness
                Position = Positions[i, :]

        # using RDA
        num_of_com = round(0.5 * Search_Agents)
        num_of_stag = Search_Agents - num_of_com
        Fitness, index = SortArray(Fit)
        Solution = SortPosition(Positions, index)
        for i in range(Search_Agents):
            if np.random.uniform(0, 1) >= 0.5:
                Temp = Solution[i, :] + np.random.uniform(0, 1) * (((xmax[i, :] - xmin[i, :]) * np.random.uniform(0, 1)) + xmin[i, :])
            else:
                Temp = Solution[i, :] - np.random.uniform(0, 1) * (((xmax[i, :] - xmin[i, :]) * np.random.uniform(0, 1)) + xmin[i, :])
            a = np.mean(Temp)
            b = np.mean(Solution[i, :])
            if a > b:
                Solution[i, :] = Temp

        num_of_com = round(0.5 * Search_Agents)
        num_of_stag = Search_Agents - num_of_com
        Fitness, index = SortArray(Fit)
        Solution = SortPosition(Positions, index)

        for i in range(num_of_com):
            for j in range(num_of_stag):
                New1 = ((Solution[i, :] + Solution[num_of_com + j, :]) / 2) + np.random.uniform(0, 1) * (
                            ((xmax[i, :] - xmin[i, :]) * np.random.uniform(0, 1)) + xmin[i, :])
                New2 = ((Solution[i, :] + Solution[num_of_com + j, :]) / 2) + np.random.uniform(0, 1) * (
                            ((xmax[i, :] - xmin[i, :]) * np.random.uniform(0, 1)) + xmin[i, :])
                a = np.mean(New1)
                b = np.mean(Solution[i, :])
                if a > b:
                    Solution[i, :] = New1
                a = np.mean(New2)
                b = np.mean(Solution[num_of_com + i, :])
                if a > b:
                    Solution[num_of_com + i, :] = New2

        for i in range(Search_Agents):
            Fnew[i, :] = objective(Solution[i, :])
            Fnew[i, :] = Check_Bounds(Fnew[i, :], xmin[i, :], xmax[i, :])
            V[i, :] = Fnew[i, :] - max(Fnew[i, :])
            P[i, :] = abs(V[i, :]) / sum(Fnew[i, :])
            N_harem[i, :] = np.round(P[i, :]) * index[i]

        for i in range(num_of_com):
            N_harem[i, :] = np.round(Solution[i, :] * N_harem[i, :])
            offs[i, :] = ((Solution[i, :] + index[i]) / 2) + (xmax[i, :] - xmin[i, :]) * np.random.uniform(0, 1)
            k = rn.randrange(10)
            N_harem[i, :] = np.round(np.random.uniform(0, 1) * N_harem[k, :])

        for i in range(num_of_stag):
            Distance[i] = np.sqrt(sum((Solution[num_of_com + i, :] - index[num_of_com + i]) ** 2))
            offs[num_of_com + i, :] = ((Solution[num_of_com + i, :] + index[num_of_com + i]) / 2) + (
                        xmax[i, :] - xmin[i, :]) * np.random.uniform(0, 1)

        for i in range(Search_Agents):
            a = np.mean(offs[i, :])
            b = np.mean(Solution[i, :])
            if a > b:
                Solution[i, :] = offs[i, :]
            a = np.mean(Fnew[i, :])
            b = np.mean(Fitness[i, :])
            if a > b:
                Fitness[i, :] = Fnew[i, :]


        A = R - l * ((R) / Max_iterations)
        for i in range(Positions.shape[1 - 1]):
            for j in range(Positions.shape[2 - 1]):
                C = 2 * np.random.rand()
                P_vec = A * Positions[i, j] + np.abs(C * ((Position[j] - Positions[i, j])))
                P_final = Position[j] - P_vec
                Positions[i, j] = P_final

        # Update using proposed
        r1 = np.random.rand()
        r2 = np.random.rand()
        Prop_Position = r1 * Positions + r2 * Solution

        Convergence[l] = Score
        l += 1
    ct = time.time() - ct
    Sol, ind = SortArray(Prop_Position)
    bestfit = np.min(Sol)
    return bestfit, Convergence, Position, ct

