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
    list_of_outliers = model.best_outliers
    for j in list_of_outliers:
        etaT_yobs, etaT_Sigma_eta, a, b = util.calculate_SI_essentials(X, y, list_of_outliers, j)
        truncatedRegion = CTRL_RANSAC.identifying_truncated_region(model, X, a, b)
        selective_pvalue = util.calculate_p_value(truncatedRegion, etaT_yobs, etaT_Sigma_eta)
        print(f"The p value of the {j}(th) instance is {selective_pvalue}")
    

if __name__ == "__main__":
    run()