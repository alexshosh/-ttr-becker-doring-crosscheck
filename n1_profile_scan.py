import numpy as np, csv
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

def load(name, Tk):
    d = {'T':[], 'I':[], 'A':[]}
    with open(f'Sun2018_TTR_panel{name}_{Tk}K_digitized.csv') as f:
        r = csv.reader(f); next(r)
        for row in r:
            sp,t,v = row
            if sp in d: d[sp].append((float(t),float(v)))
    for k in d: d[k]=np.array(sorted(d[k]))
    return d

panels={'B':298,'C':310,'D':277}

def logi(t, mu, r1):
    return 1.0/(1+np.exp(mu)*np.exp(-r1*t))

def xfun_of(mu, r1, a, k_fast):
    x_slow_0 = logi(0.0, mu, r1)
    def xfun(t):
        x_slow = (logi(t, mu, r1) - x_slow_0) / (1.0 - x_slow_0)
        x_fast = 1.0 - np.exp(-k_fast*t)
        return (1-a)*x_slow + a*x_fast
    return xfun

def fit_stage1(t, Tdata, ntrials=80, seed=0):
    def resid(p):
        mu, r1_log, a_logit, k_fast_log = p
        r1 = np.exp(r1_log)
        a = 1/(1+np.exp(-a_logit)); k_fast = np.exp(k_fast_log)
        xfun = xfun_of(mu, r1, a, k_fast)
        pred = 1.0 - xfun(t)
        return pred - Tdata
    lo = np.array([-15, np.log(1e-4), -5, np.log(0.001)])
    hi = np.array([15, np.log(5.0),   5, np.log(10.0)])
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(ntrials):
        p0 = rng.uniform(lo, hi)
        sol = least_squares(resid, p0, bounds=(lo,hi), max_nfev=3000, xtol=1e-13, ftol=1e-13)
        rmse = np.sqrt(np.mean(resid(sol.x)**2))
        if best is None or rmse < best[0]:
            best = (rmse, sol.x)
    x = best[1]
    mu,r1 = x[0], np.exp(x[1])
    a = 1/(1+np.exp(-x[2])); k_fast = np.exp(x[3])
    return mu,r1,a,k_fast

def m_ode(t, y, xfun, w, power):
    m = y[0]
    c1 = max(xfun(t)-m, 0.0)
    return [w*c1**power]

def fit_stage2(tI, Id, tA, Ad, xfun, power):
    tmax = max(tI.max(), tA.max())
    def resid(p):
        w = np.exp(p[0]); kI, kA = p[1], p[2]
        t_dense = np.linspace(0, tmax, 800)
        sol = solve_ivp(m_ode, [0,tmax], [0.0], t_eval=t_dense, args=(xfun,w,power),
                         method='RK45', rtol=1e-9, atol=1e-11, max_step=0.3)
        m_dense = sol.y[0]; x_dense = np.array([xfun(tt) for tt in t_dense])
        mI = np.interp(tI,t_dense,m_dense); xI = np.interp(tI,t_dense,x_dense)
        mA = np.interp(tA,t_dense,m_dense)
        return np.concatenate([kI*np.clip(xI-mI,0,None)-Id, kA*mA-Ad])
    best=None
    for w0 in [1e-3,1e-2,1e-1,1.0,10.0,100.0]:
        p0=[np.log(w0),1.0,1.0]
        try:
            sol = least_squares(resid,p0,bounds=([-15,0.05,0.05],[15,5,5]))
        except Exception:
            continue
        if best is None or sol.cost<best.cost: best=sol
    if best is None:
        return None
    w,kI,kA = np.exp(best.x[0]), best.x[1], best.x[2]
    return w,kI,kA,best.cost

N1_grid = np.round(np.arange(0.5, 4.01, 0.25), 2)

for name,Tk in panels.items():
    d = load(name,Tk)
    tT,Td = d['T'][:,0], d['T'][:,1]
    tI,Id = d['I'][:,0], d['I'][:,1]
    tA,Ad = d['A'][:,0], d['A'][:,1]
    mu,r1,a,k_fast = fit_stage1(tT,Td, seed=Tk)
    xfun = xfun_of(mu,r1,a,k_fast)

    print(f"\n=== {Tk}K: profile scan over N1 (Stage 1 held fixed from mixture fit) ===")
    costs=[]
    for N1 in N1_grid:
        power = N1 + 1.0
        res = fit_stage2(tI,Id,tA,Ad,xfun,power)
        if res is None:
            print(f"  N1={N1:.2f}  FIT FAILED"); continue
        w,kI,kA,cost = res
        costs.append((N1,cost))
        print(f"  N1={N1:.2f}  cost={cost:.6f}  w={w:.4g} kI={kI:.3f} kA={kA:.3f}")
    best_N1,best_cost = min(costs, key=lambda z:z[1])
    print(f"  --> best N1 = {best_N1} (cost={best_cost:.6f})")
