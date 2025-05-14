from mpmath import mp
import numpy as np
from numpy.linalg import pinv


def calculate_SI_essentials(X, y_obs, Oobs, j):
    n = X.shape[0]
    p = X.shape[1]
    
    variance = 1
    Sigma = np.identity(n) * variance # covariance matrix
    
    # construct eta
    ej = np.zeros((n, 1))
    ej[j][0] = 1
    xj = X[j].reshape((p, 1))
    
    I_minusOobs = np.zeros((n, n))
    for i in range(n):
        if i not in Oobs:
            I_minusOobs[i][i] = 1
    X_minusOobs = np.dot(I_minusOobs, X)
    eta = (ej.T - np.dot(np.dot(xj.T, pinv(X_minusOobs)), I_minusOobs)).T
    
    etaT_Sigma_eta = np.dot(np.dot(eta.T, Sigma), eta)[0][0]  # variance of truncated normal distribution 
    etaT_yobs = np.dot((eta.T), y_obs)[0][0]  # test statistic
    
    b = np.dot(Sigma, eta) / etaT_Sigma_eta
    a = np.dot((np.identity(n) - np.dot(b, eta.T)), y_obs)
    return etaT_yobs, etaT_Sigma_eta, a, b

def solveEquation(A, B, C):
    """
    Solve the inequaility Ax^2 + Bx + C <= 0
    """
    # Case: A = 0
    if A == 0:
        left = None
        right = None
        # Case: A, B = 0 
        if B == 0:
            if C > 0:
                return []
            else:
                return [(-np.inf, np.inf)]
        # Case: B != 0
        else: 
            # Case: B > 0
            if B > 0:
                #  [-oo, -C/B]
                left = -np.inf
                right = -C/B
            # Case: Q2 < 0
            else:
                # [-C/B, oo]
                left = -C/B
                right = np.inf
            return [(left, right)]
            
    # Case: A != 0
    else:
        Delta = B**2 - 4*A*C
        # Case: Delta <= 0
        if Delta <= 0:
            if A > 0:
                if Delta == 0:
                    return [(-B/(2*A), -B/(2*A))]
                return []
            else:
                return [(-np.inf, np.inf)]
        # Case: Delta > 0
        else:
            sol1 = (-B + np.sqrt(Delta))/(2*A)
            sol2 = (-B - np.sqrt(Delta))/(2*A)
            # Ensure sol1 < sol2
            if sol1 > sol2:
                sol1, sol2 = sol2, sol1
            # Case: Q1 > 0
            if A > 0:
                return [(sol1, sol2)]
            else:
                return [(-np.inf, sol1), (sol2, np.inf)]
            


def getIntersection(list1, list2):
    list1.sort()
    list2.sort()

    intersections = []
    i, j = 0, 0

    while i < len(list1) and j < len(list2):
        start1, end1 = list1[i]
        start2, end2 = list2[j]

        start_intersection = max(start1, start2)
        end_intersection = min(end1, end2)

        if start_intersection <= end_intersection:
            intersections.append((start_intersection, end_intersection))

        if end1 < end2:
            i += 1
        else:
            j += 1

    return intersections

def merge_ranges(ranges):
    if not ranges:
        return []
    
    ranges.sort()
    merged = [ranges[0]]

    for current in ranges[1:]:
        last_merged = merged[-1]
        
        if current[0] <= last_merged[1]:
            merged[-1] = (last_merged[0], max(last_merged[1], current[1]))
        else:
            merged.append(current)

    return merged

def getUnion(list1, list2):
    combined = list1 + list2
    return merge_ranges(combined)

def getComplement(intervals):
    result = []
    current_start = float('-inf')
    
    for interval in sorted(intervals):
        if current_start < interval[0]:
            result.append((current_start, interval[0]))
        current_start = max(current_start, interval[1])
    
    if current_start < float('inf'):
        result.append((current_start, float('inf')))
    
    return result
            


def calculate_p_value(Regions, etaT_y, etaT_Sigma_eta):
    mp.dps = 100
    numerator = 0
    denominator = 0
    mu = 0
    tn_sigma = np.sqrt(etaT_Sigma_eta)
    for i in Regions:
        left = i[0]
        right = i[1]
        denominator = denominator + mp.ncdf((right - mu)/tn_sigma) - mp.ncdf((left - mu)/tn_sigma)
        if etaT_y >= right:
            numerator = numerator + mp.ncdf((right - mu)/tn_sigma) - mp.ncdf((left - mu)/tn_sigma)
        elif (etaT_y >= left) and (etaT_y < right):
            numerator = numerator + mp.ncdf((etaT_y - mu)/tn_sigma) - mp.ncdf((left - mu)/tn_sigma)
    
    if denominator == 0:
        print("Error")
        return None
    else:
        cdf = float(numerator/denominator) 
        pvalue = 2*min(cdf, 1 - cdf)
        return pvalue
    
