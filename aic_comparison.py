"""
Reproduces the AIC comparison in ttr_crosscheck_v3.tex, S:What-this-note-adds-beyond-SDW's-own-model.

Both models (the 7-parameter mixture model of fit_final_mixture.py and the
4-parameter SDW reversible-chain model of sdw_model_refit.py) are refit here,
independently of those two scripts, and their RSS is used directly -- this
script does not read RMSE numbers from Table 1 or from either script's own
hardcoded values, to keep the AIC comparison free of any hand-copied number.

AIC = 2k - 2*ln(Lhat). Under the standard i.i.d.-Gaussian-residual assumption
used implicitly by every least-squares fit in this project, at the Gaussian
MLE: -2*ln(Lhat) = n*ln(2*pi) + n*ln(RSS/n) + n, so
    AIC = 2k + n*ln(RSS/n) + const,
where const = n*ln(2*pi) + n is identical for both models being compared (same
n, same data) and cancels out of any AIC difference.

RSS and n are pooled across T, I, and A together at each temperature (not
curve by curve), matching how the paper describes the comparison.
"""
import csv
import math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

def load(name, Tk):
    d = {'T': [], 'I': [], 'A': []}
    with open(f'Sun2018_TTR_panel{name}_{Tk}K_digitized.csv') as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            sp, t, v = row
            if sp in d:
                d[sp].append((float(t), float(v)))
    for k in d:
        d[k] = np.array(sorted(d[k]))
    return d

panels = {'B': 298, 'C': 310, 'D': 277}
N1 = 2.0
POWER = N1 + 1.0
K_OURS, K_SDW = 7, 4

# ---- our mixture model (Stage 1 + Stage 2), same as fit_final_mixture.py ----

def logi(t, mu, r1):
    return 1.0 / (1 + np.exp(mu) * np.exp(-r1 * t))

def xfun_of(mu, r1, a, k_fast):
    x_slow_0 = logi(0.0, mu, r1)
    def xfun(t):
        x_slow = (logi(t, mu, r1) - x_slow_0) / (1.0 - x_slow_0)
        x_fast = 1.0 - np.exp(-k_fast * t)
        return (1 - a) * x_slow + a * x_fast
    return xfun

def fit_stage1(t, Tdata, ntrials=80, seed=0):
    def resid(p):
        mu, r1_log, a_logit, k_fast_log = p
        r1 = np.exp(r1_log)
        a = 1 / (1 + np.exp(-a_logit)); k_fast = np.exp(k_fast_log)
        xfun = xfun_of(mu, r1, a, k_fast)
        return (1.0 - xfun(t)) - Tdata
    lo = np.array([-15, np.log(1e-4), -5, np.log(0.001)])
    hi = np.array([15, np.log(5.0), 5, np.log(10.0)])
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(ntrials):
        p0 = rng.uniform(lo, hi)
        sol = least_squares(resid, p0, bounds=(lo, hi), max_nfev=3000, xtol=1e-13, ftol=1e-13)
        rmse = np.sqrt(np.mean(resid(sol.x) ** 2))
        if best is None or rmse < best[0]:
            best = (rmse, sol.x)
    x = best[1]
    mu, r1 = x[0], np.exp(x[1])
    a = 1 / (1 + np.exp(-x[2])); k_fast = np.exp(x[3])
    return mu, r1, a, k_fast

def m_ode(t, y, xfun, w):
    m = y[0]
    c1 = max(xfun(t) - m, 0.0)
    return [w * c1 ** POWER]

def fit_stage2(tI, Id, tA, Ad, xfun):
    tmax = max(tI.max(), tA.max())
    def resid(p):
        w = np.exp(p[0]); kI, kA = p[1], p[2]
        t_dense = np.linspace(0, tmax, 800)
        sol = solve_ivp(m_ode, [0, tmax], [0.0], t_eval=t_dense, args=(xfun, w),
                         method='RK45', rtol=1e-9, atol=1e-11, max_step=0.3)
        m_dense = sol.y[0]; x_dense = np.array([xfun(tt) for tt in t_dense])
        mI = np.interp(tI, t_dense, m_dense); xI = np.interp(tI, t_dense, x_dense)
        mA = np.interp(tA, t_dense, m_dense)
        return np.concatenate([kI * np.clip(xI - mI, 0, None) - Id, kA * mA - Ad])
    best = None
    for w0 in [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]:
        p0 = [np.log(w0), 1.0, 1.0]
        sol = least_squares(resid, p0, bounds=([-15, 0.05, 0.05], [15, 5, 5]))
        if best is None or sol.cost < best.cost:
            best = sol
    w, kI, kA = np.exp(best.x[0]), best.x[1], best.x[2]
    return w, kI, kA, best.cost

def rss_ours(name, Tk, d):
    tT, Td = d['T'][:, 0], d['T'][:, 1]
    tI, Id = d['I'][:, 0], d['I'][:, 1]
    tA, Ad = d['A'][:, 0], d['A'][:, 1]
    mu, r1, a, k_fast = fit_stage1(tT, Td, seed=Tk)
    xfun = xfun_of(mu, r1, a, k_fast)
    w, kI, kA, _ = fit_stage2(tI, Id, tA, Ad, xfun)
    tmax = max(tT.max(), tI.max(), tA.max())
    t_dense = np.linspace(0, tmax, 800)
    sol = solve_ivp(m_ode, [0, tmax], [0.0], t_eval=t_dense, args=(xfun, w),
                     method='RK45', rtol=1e-9, atol=1e-11, max_step=0.3)
    m_dense = sol.y[0]; x_dense = np.array([xfun(tt) for tt in t_dense])
    Tpred = 1.0 - np.array([xfun(tt) for tt in tT])
    Ipred = kI * np.clip(np.interp(tI, t_dense, x_dense) - np.interp(tI, t_dense, m_dense), 0, None)
    Apred = kA * np.interp(tA, t_dense, m_dense)
    rss = np.sum((Tpred - Td) ** 2) + np.sum((Ipred - Id) ** 2) + np.sum((Apred - Ad) ** 2)
    n = len(tT) + len(tI) + len(tA)
    return rss, n

# ---- SDW's reversible-chain model, same as sdw_model_refit.py ----

def rhs(t, y, k1, km1, k2, km2):
    T, I, A = y
    return [-k1 * T + km1 * I,
            k1 * T - km1 * I - k2 * I + km2 * A,
            k2 * I - km2 * A]

def model_curves(params, tT, tI, tA):
    k1, km1, k2, km2 = params
    tmax = max(tT.max(), tI.max(), tA.max())
    sol = solve_ivp(rhs, [0, tmax], [1.0, 0.0, 0.0], args=(k1, km1, k2, km2),
                     dense_output=True, method='RK45', rtol=1e-9, atol=1e-11)
    return sol.sol(tT)[0], sol.sol(tI)[1], sol.sol(tA)[2]

def rss_sdw(d, seed):
    tT, Td = d['T'][:, 0], d['T'][:, 1]
    tI, Id = d['I'][:, 0], d['I'][:, 1]
    tA, Ad = d['A'][:, 0], d['A'][:, 1]
    def residuals(params):
        T_pred, I_pred, A_pred = model_curves(params, tT, tI, tA)
        return np.concatenate([T_pred - Td, I_pred - Id, A_pred - Ad])
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(40):
        x0 = 10 ** rng.uniform(-2, 0.5, size=4)
        try:
            sol = least_squares(residuals, x0, bounds=(1e-5, 20.0), method='trf', max_nfev=5000)
        except Exception:
            continue
        if best is None or sol.cost < best.cost:
            best = sol
    T_pred, I_pred, A_pred = model_curves(best.x, tT, tI, tA)
    rss = np.sum((T_pred - Td) ** 2) + np.sum((I_pred - Id) ** 2) + np.sum((A_pred - Ad) ** 2)
    n = len(tT) + len(tI) + len(tA)
    return rss, n

if __name__ == '__main__':
    print(f"{'T(K)':>5} {'n':>5} {'RSS_ours':>10} {'RSS_SDW':>10} "
          f"{'AIC_ours':>10} {'AIC_SDW':>10} {'dAIC(ours favored by)':>22}")
    for name, Tk in panels.items():
        d = load(name, Tk)
        rss_o, n_o = rss_ours(name, Tk, d)
        rss_s, n_s = rss_sdw(d, seed=Tk)
        assert n_o == n_s, "point counts must match between the two fits"
        n = n_o
        aic_ours = 2 * K_OURS + n * math.log(rss_o / n)
        aic_sdw = 2 * K_SDW + n * math.log(rss_s / n)
        dAIC = aic_sdw - aic_ours
        print(f"{Tk:>5} {n:>5} {rss_o:>10.5f} {rss_s:>10.5f} "
              f"{aic_ours:>10.2f} {aic_sdw:>10.2f} {dAIC:>22.2f}")
