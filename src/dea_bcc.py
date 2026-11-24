"""
Input-oriented BCC DEA (envelopment / dual form) for 12 suppliers.

For each supplier o:
    min  theta
    s.t. sum_j lambda_j * x_ij <= theta * x_io     (inputs: unit price, late delivery %, shipping error %)
         sum_j lambda_j * y_rj >= y_ro             (outputs: product-quality utility, customer-service utility)
         sum_j lambda_j = 1                        (variable returns to scale, BCC)
         lambda_j >= 0

Reproduces the Excel Solver results in DEA_Dual_BCC.xlsx using scipy.
Run: python dea_bcc.py
"""
import numpy as np
import pandas as pd
from scipy.optimize import linprog

suppliers = list("ABCDEFGHIJKL")
# Inputs (less is better)
X = np.array([
    # unit price ($), late delivery (%), shipping error (%)
    [1.34, 3, 8], [1.44, 5, 4], [2.13, 6, 6], [2.01, 9, 3],
    [3.44, 4, 7], [2.98, 4, 6], [1.34, 1, 3], [5.10, 2, 3],
    [1.40, 1, 3], [1.50, 6, 4], [1.33, 3, 3], [1.94, 9, 6],
])
# Outputs (more is better): utilities from the Evidential Reasoning assessment
Y = np.array([
    # product quality, customer service
    [0.7524, 0.6525], [0.6941, 0.6491], [0.5544, 0.6198], [0.5433, 0.5318],
    [0.4625, 0.5402], [0.5010, 0.3958], [0.1363, 0.4677], [0.7998, 0.6926],
    [0.4523, 0.6108], [0.4360, 0.6128], [0.3778, 0.5172], [0.5372, 0.5513],
])


def bcc_input_oriented(X, Y, o):
    """Return (theta, lambdas) for DMU o. Decision vector = [theta, lambda_1..lambda_n]."""
    n, m = X.shape
    s = Y.shape[1]
    c = np.r_[1, np.zeros(n)]                        # minimise theta
    A_in = np.c_[-X[o].reshape(m, 1), X.T]           # sum lambda*x - theta*x_o <= 0
    A_out = np.c_[np.zeros((s, 1)), -Y.T]            # -sum lambda*y <= -y_o
    A_ub = np.r_[A_in, A_out]
    b_ub = np.r_[np.zeros(m), -Y[o]]
    A_eq = np.r_[0, np.ones(n)].reshape(1, -1)       # convexity (BCC)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1],
                  bounds=[(0, None)] * (n + 1), method="highs")
    return res.x[0], res.x[1:]


if __name__ == "__main__":
    rows = []
    for o, name in enumerate(suppliers):
        theta, lam = bcc_input_oriented(X, Y, o)
        peers = {suppliers[j]: round(l, 3) for j, l in enumerate(lam) if l > 1e-4 and j != o}
        targets = lam @ X                                 # composite benchmark inputs
        rows.append({
            "Supplier": name,
            "theta*": round(theta, 4),
            "Status": "Efficient" if theta > 0.9999 else "Inefficient",
            "Peers (lambda)": peers if theta < 0.9999 else "-",
            "Target unit price": round(targets[0], 3),
            "Target late del %": round(targets[1], 3),
            "Target ship err %": round(targets[2], 3),
        })

    results = pd.DataFrame(rows)
    pd.set_option("display.width", 200, "display.max_colwidth", 60)
    print(results.to_string(index=False))
    print(f"\nEfficient suppliers: {', '.join(results.loc[results.Status=='Efficient','Supplier'])}")
