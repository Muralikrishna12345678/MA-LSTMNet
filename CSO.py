import numpy as np
import time

def CSO(X, Obj_func, Vmax, Vmin, K):
    [sizepop, dim] = X.shape
    delta = 0.2
    V = np.random.rand(sizepop, dim)  # Initial velocities in [0, 1]
    X_best = X.copy()  # Best-known positions
    X_best_Obj_func = np.array([Obj_func(x) for x in X_best])

    convergence = np.zeros((K))
    ct = time.time()
    for k in range(K):
        w = 0.9 - k * (0.9 - 0.4) / K  # Inertia weight decreases over time
        # Update delta based on Eq. (8)
        delta = delta * (1 - (1 - 0.001) * (k / K))

        for i in range(sizepop):
            r1, r2 = np.random.rand(), np.random.rand()  # Random values in [0, 1]
            X_rnd = X[np.random.choice(sizepop)]  # Randomly select a search agent

            # Calculate Obj_func values for the search agent and its best neighbor
            Obj_func_current = Obj_func(X[i])
            Obj_func_local_best = Obj_func(X_best[i])

            # Update the incremental function based on Eq. (4)
            if Obj_func_current < Obj_func_local_best:
                gamma = 1 - w
            else:
                gamma = 0

            # Update velocity using Eq. (2)
            V[i] = w * V[i] + delta * gamma * (X_best[i] - X[i]) + r1 * (X_best[i] - X[i]) + r2 * (X_rnd - X[i])

            # Update position using Eq. (1)
            X[i] = X[i] + V[i]

            # Clip velocity to stay within boundaries
            V[i] = np.clip(V[i], Vmin[i], Vmax[i])
            X[i] = np.clip(X[i], 0, 1)

            # Update best-known positions and Obj_func
            if Obj_func(X[i]) < X_best_Obj_func[i]:
                X_best[i] = X[i]
                X_best_Obj_func[i] = Obj_func(X[i])
        best_idx = np.argmin(X_best_Obj_func)
        x_best = X_best[best_idx]
        convergence[k] = Obj_func(x_best)
    # Find the best solution
    best_idx = np.argmin(X_best_Obj_func)
    x_best = X_best[best_idx]
    best_Fit = Obj_func(x_best)
    ct -= time.time()
    return best_Fit, convergence, x_best, ct



