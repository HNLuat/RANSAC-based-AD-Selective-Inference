import numpy as np
import matplotlib.pyplot as plt

import util
import gendata
import regressor 
import CTRL_RANSAC

def run():
    n = 100
    p = 5
    B = 10
    t = 2
    num_outliers = 0
    delta = 0
    X, y = gendata.gen(n, p, num_outliers, delta)
    model = regressor.RANSAC(B, t)
    model.fit(X, y)
    Oobs = model.best_outliers
    if len(Oobs) == 0:
        return None
    j_selected = np.random.choice(Oobs)
    etaT_yobs, etaT_Sigma_eta, a, b = util.calculate_SI_essentials(X, y, Oobs, j_selected)
    truncatedRegion = CTRL_RANSAC.identifying_truncated_region(model, X, a, b)
    selective_pvalue = util.calculate_p_value(truncatedRegion, etaT_yobs, etaT_Sigma_eta)
    return selective_pvalue

if __name__ == "__main__":
    trials = 2000
    p_values = []
    reject = 0
    detect = 0

    for trial in range(trials):
        if (trial+1)%10 == 0:
            print("trial ", trial + 1)
        p_value = run()
        if p_value is None:
            continue
        detect += 1
        if p_value < 0.05:
            reject += 1
        p_values.append(p_value)
    
    print(f"reject: {reject}, detect: {detect}, FPR: {reject/detect}")
    plt.hist(p_values)
    plt.savefig("./results/FPR_plot")
