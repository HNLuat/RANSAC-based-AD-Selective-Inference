from util import solveEquation # solve the inequality Ax^2 + Bx + C <= 0
from util import getIntersection # get the intersection of 2 collections of intervals
from util import getUnion # get the union of 2 collections of intervals
from util import getComplement # get the complement of a collection of intervals
import numpy as np
from numpy.linalg import pinv
    
def identifying_truncated_region(model, X, a, b):
    """
    Input:
    - model: the RANSAC model that is fitted with X, y_obs
    - X: the features matrix
    - a, b: necessary variables that are calculated and used in Selective Inference
    Output: the truncated region
    """
    B = model.B
    t = model.t
    Oobs = model.best_outliers
    n = X.shape[0]
    p = X.shape[1]
    maybe_inliers_set = model.maybe_inliers_set 
    truncatedRegion = []
    # Computing R 
    R = [[None for j in range(n + 1)] for i in range(B + 1)]
    for i in range(1, B + 1):
        I_si = np.zeros((n, n))
        for idx in maybe_inliers_set[i-1]:
            I_si[idx][idx] = 1
        X_si = np.dot(I_si, X)

        for j in range(1, n + 1):
            xj = X[j-1].reshape((p,1))
            Omega = (np.dot(np.dot(xj.T, pinv(X_si)), I_si)).T
            ej = np.zeros((n,1))
            ej[j-1] = 1

            Q = np.dot(ej, ej.T) - 2*np.dot(ej, Omega.T) + np.dot(Omega, Omega.T)

            u = np.dot(np.dot(a.T, Q), a)[0][0] - t
            v = np.dot(np.dot(a.T, (Q.T + Q)), b)[0][0]
            w = np.dot(np.dot(b.T, Q), b)[0][0]

            R[i][j] = solveEquation(w, v, u)

    # Construct S
    S = [[[[] for k in range(len(Oobs) + 1)] for j in range(n + 1)] for i in range(B + 1)]
    for i in range(1, B+1):
        for j in range(1, n+1):
            S[i][j][0] = getUnion(getComplement(R[i][j]), S[i][j-1][0])
    
    for i in range(1, B+1):
        for j in range(1, n+1):
            for o in range(1, len(Oobs)+1):
                res1 = getIntersection(R[i][j], S[i][j-1][o])
                res2 = getIntersection(getComplement(R[i][j]), S[i][j-1][o-1])
                S[i][j][o] = getUnion(res1, res2)

    # Calculate Z
    Z = []
    for b in range(1, B+1):
        Z2_b = [(-np.inf, np.inf)]
        for j in range(1, n + 1):
            if j - 1 not in Oobs:
                Z2_b = getIntersection(Z2_b, R[b][j])
            else:
                Z2_b = getIntersection(Z2_b, getComplement(R[b][j]))
        Z1_b = [(-np.inf, np.inf)]
        for u in range(1, b):
            Z1_b = getIntersection(Z1_b, S[u][n][len(Oobs)])
        for u in range(b+1, B+1):
            Z1_b = getIntersection(Z1_b, S[u][n][len(Oobs)-1])

        Z = getUnion(Z, getIntersection(Z1_b, Z2_b))
    
    return Z
