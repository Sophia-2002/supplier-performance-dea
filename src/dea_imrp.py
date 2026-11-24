"""
DEA-IMRP: interactive minimax reference point trade-off analysis for inefficient suppliers
(Yang & Li 2002; Yang et al. 2009; Yang, Xu & Yang 2012).

Pipeline for each inefficient supplier (DMU jo):
  1. Payoff table  -> ideal point f*  (min each input separately over Omega_jo)
  2. Initial weights w0_i = 1 / (x_i,jo - f*_i), solve minimax LP -> iter-0 efficient point
  3. Gradient projection loop with decision-maker preference direction df and step factor gamma:
       normal N = w * beta (duals of minimax constraints)
       Delta = df - (df.N / N.N) N       (project preference onto frontier tangent plane)
       alpha = gamma * alpha_max
       new target Y = f + alpha*Delta, weights w = 1 / (Y - f*), re-solve minimax LP
  4. Stop when move size ||alpha*Delta|| falls below the noise threshold (0.05), or at the
     iteration the analysis settled on.

Omega_jo = { lambda >= 0 : sum lambda_j y_rj >= y_r,jo ,  sum lambda_j x_ij <= x_i,jo ,  sum lambda_j = 1 }
(BCC frontier; no target input may exceed the supplier's current level)

Run: python dea_imrp.py
"""
import numpy as np
import pandas as pd
from scipy.optimize import linprog

from dea_bcc import X, Y, suppliers, bcc_input_oriented

INPUTS = ["Unit Price", "Late Delivery", "Ship Error"]

# Decision-maker preferences: priority Late Delivery > Ship Error > Unit Price, risk-averse
DF_SCHEDULE = [np.array([+0.05, -0.50, -0.20]),   # iter-1
               np.array([+0.04, -0.30, -0.30]),   # iter-2
               np.array([+0.03, -0.25, -0.18])]   # iter-3
GAMMA = [0.25, 0.20, 0.15]
FINAL_ITER = {"C": 3, "E": 1, "F": 2, "J": 1, "L": 2}   # stopping iteration from the analysis


def _omega(jo):
    """Constraints defining Omega_jo, for decision vector lambda (length n)."""
    n = X.shape[0]
    A_ub = np.r_[-Y.T, X.T]          # outputs: -sum lambda*y <= -y_jo ; inputs: sum lambda*x <= x_jo
    b_ub = np.r_[-Y[jo], X[jo]]
    A_eq = np.ones((1, n))
    return A_ub, b_ub, A_eq, np.array([1.0])


def payoff_ideal(jo):
    """Ideal point f*: minimum of each input objective taken separately."""
    A_ub, b_ub, A_eq, b_eq = _omega(jo)
    return np.array([
        linprog(X[:, i], A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                bounds=(0, None), method="highs").fun
        for i in range(X.shape[1])])


def minimax(jo, w, f_star):
    """min eta  s.t. w_i (f_i(lambda) - f*_i) <= eta,  lambda in Omega_jo.
    Returns (f(lambda), lambda, beta) where beta are the duals of the minimax constraints."""
    n, m = X.shape
    A_ub0, b_ub0, A_eq0, b_eq = _omega(jo)
    c = np.r_[1.0, np.zeros(n)]                                  # vars: [eta, lambda]
    A_mm = np.c_[-np.ones((m, 1)), (w[:, None] * X.T)]           # w_i*sum lambda x_i - eta <= w_i f*_i
    A_ub = np.r_[A_mm, np.c_[np.zeros((A_ub0.shape[0], 1)), A_ub0]]
    b_ub = np.r_[w * f_star, b_ub0]
    A_eq = np.c_[np.zeros((1, 1)), A_eq0]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=[(None, None)] + [(0, None)] * n, method="highs")
    lam = res.x[1:]
    beta = -res.ineqlin.marginals[:m]
    return lam @ X, lam, beta


def run_imrp(jo, n_iter):
    x0 = X[jo]
    f_star = payoff_ideal(jo)
    w = 1.0 / (x0 - f_star)
    f, lam, beta = minimax(jo, w, f_star)
    log = [{"iter": 0, "target": f.copy(), "lambda": lam, "move": np.nan}]
    for t in range(n_iter):
        N = w * beta                                             # normal vector (Yang & Li 2002, Eq. 21)
        df = DF_SCHEDULE[t]
        delta = df - (df @ N) / (N @ N) * N                      # projection (Eq. 28)
        caps = [(f[i] - f_star[i]) / -delta[i] if delta[i] < 0 else
                (x0[i] - f[i]) / delta[i] for i in range(len(f)) if abs(delta[i]) > 1e-12]
        alpha = GAMMA[t] * min(caps)
        move = np.linalg.norm(alpha * delta)
        Y_new = f + alpha * delta
        w = 1.0 / (Y_new - f_star)                               # weight update (Yang et al. 2009, Eq. 29)
        f, lam, beta = minimax(jo, w, f_star)
        log.append({"iter": t + 1, "target": f.copy(), "lambda": lam, "move": move})
    return f_star, log


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    summary = []
    for jo, name in enumerate(suppliers):
        theta, _ = bcc_input_oriented(X, Y, jo)
        if theta > 0.9999:
            continue
        f_star, log = run_imrp(jo, FINAL_ITER[name])
        print(f"\n=== Supplier {name}  (BCC theta* = {theta:.4f}) ===")
        print("Ideal point f*:", np.round(f_star, 4))
        traj = pd.DataFrame([{"iter": r["iter"], **dict(zip(INPUTS, np.round(r["target"], 4))),
                              "||alpha*Delta||": round(r["move"], 4)} for r in log])
        print(pd.concat([pd.DataFrame([{"iter": "current", **dict(zip(INPUTS, X[jo]))}]), traj]).to_string(index=False))
        final = log[-1]["target"]
        summary.append({"Supplier": name, **{f"{k} (current)": v for k, v in zip(INPUTS, X[jo])},
                        **{f"{k} (target)": round(v, 2) for k, v in zip(INPUTS, final)},
                        "Final iter": FINAL_ITER[name]})
    print("\n=== Final improvement targets ===")
    print(pd.DataFrame(summary).to_string(index=False))
