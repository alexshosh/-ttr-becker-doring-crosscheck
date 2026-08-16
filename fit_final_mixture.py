"""
Stage 1 replaces the plain single logistic with an honest two-component mixture:
x(t) = (1-a)*x_slow(t) + a*x_fast(t), x_slow = the SAME closed-form reduced-manifold
logistic as before (Eq. eq:x, unchanged), x_fast = 1-exp(-k_fast t), a fast relaxation
with no claimed physical mechanism (purely empirical, disclosed as such).
T0 is fixed at its exact physical value (1) for all three temperatures -- no free/bounded
T0 parameter anywhere in this script, unlike the original fit_final_N1_2.py.
Stage 2 (I, A via m(t)) is refit against this new x(t), same procedure as before.
"""
import numpy as np, csv
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares, differential_evolution
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
N1 = 2.0
POWER = N1 + 1.0

def logi(t, mu, r1):
    return 1.0/(1+np.exp(mu)*np.exp(-r1*t))

def xfun_of(mu, r1, a, k_fast):
    x_slow_0 = logi(0.0, mu, r1)   # re-anchor: logi(0,mu,r1) = 1/(1+e^mu) != 0 in general
    def xfun(t):
        x_slow = (logi(t, mu, r1) - x_slow_0) / (1.0 - x_slow_0)
        x_fast = 1.0 - np.exp(-k_fast*t)
        return (1-a)*x_slow + a*x_fast
    return xfun

def fit_stage1(t, Tdata, ntrials=80, seed=0):
    # T0=1 exact; T_inf is NOT a free parameter -- it emerges implicitly from the
    # mixture's own rates not fully saturating within the observed time window,
    # exactly as validated standalone (adding a redundant explicit Tinf made the
    # optimization landscape harder and was a real regression, not an improvement).
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

def r2(y, yhat):
    ss_res = np.sum((y-yhat)**2); ss_tot = np.sum((y-np.mean(y))**2)
    return 1 - ss_res/ss_tot

def m_ode(t, y, xfun, w):
    m = y[0]
    c1 = max(xfun(t)-m, 0.0)
    return [w*c1**POWER]

def fit_stage2(tI, Id, tA, Ad, xfun):
    tmax = max(tI.max(), tA.max())
    def resid(p):
        w = np.exp(p[0]); kI, kA = p[1], p[2]
        t_dense = np.linspace(0, tmax, 800)
        sol = solve_ivp(m_ode, [0,tmax], [0.0], t_eval=t_dense, args=(xfun,w),
                         method='RK45', rtol=1e-9, atol=1e-11, max_step=0.3)
        m_dense = sol.y[0]; x_dense = np.array([xfun(tt) for tt in t_dense])
        mI = np.interp(tI,t_dense,m_dense); xI = np.interp(tI,t_dense,x_dense)
        mA = np.interp(tA,t_dense,m_dense)
        return np.concatenate([kI*np.clip(xI-mI,0,None)-Id, kA*mA-Ad])
    best=None
    for w0 in [1e-3,1e-2,1e-1,1.0,10.0,100.0]:
        p0=[np.log(w0),1.0,1.0]
        sol = least_squares(resid,p0,bounds=([-15,0.05,0.05],[15,5,5]))
        if best is None or sol.cost<best.cost: best=sol
    w,kI,kA = np.exp(best.x[0]), best.x[1], best.x[2]
    return w,kI,kA,best.cost

fig, axes = plt.subplots(1,3, figsize=(16.5,5.2), sharey=True)
results={}
for ax,(name,Tk) in zip(axes,panels.items()):
    d = load(name,Tk)
    tT,Td = d['T'][:,0], d['T'][:,1]
    tI,Id = d['I'][:,0], d['I'][:,1]
    tA,Ad = d['A'][:,0], d['A'][:,1]
    mu,r1,a,k_fast = fit_stage1(tT,Td, seed=Tk)
    xfun = xfun_of(mu,r1,a,k_fast)
    w,kI,kA,cost = fit_stage2(tI,Id,tA,Ad,xfun)

    tmax = max(tT.max(),tI.max(),tA.max())
    t_dense = np.linspace(0,tmax,800)
    sol = solve_ivp(m_ode,[0,tmax],[0.0],t_eval=t_dense,args=(xfun,w),method='RK45',rtol=1e-9,atol=1e-11,max_step=0.3)
    m_dense=sol.y[0]; x_dense=np.array([xfun(tt) for tt in t_dense])
    Tpred_dense = 1.0-x_dense
    Ipred_dense = kI*np.clip(x_dense-m_dense,0,None)
    Apred_dense = kA*m_dense

    Tpred_atT = 1.0-np.array([xfun(tt) for tt in tT])
    Ipred_atI = kI*np.clip(np.interp(tI,t_dense,x_dense)-np.interp(tI,t_dense,m_dense),0,None)
    Apred_atA = kA*np.interp(tA,t_dense,m_dense)
    R2_T,R2_I,R2_A = r2(Td,Tpred_atT), r2(Id,Ipred_atI), r2(Ad,Apred_atA)
    RMSE_T = np.sqrt(np.mean((Tpred_atT-Td)**2))
    RMSE_I = np.sqrt(np.mean((Ipred_atI-Id)**2))
    RMSE_A = np.sqrt(np.mean((Apred_atA-Ad)**2))

    results[Tk]=dict(mu=mu,r1=r1,a=a,k_fast=k_fast,w=w,kI=kI,kA=kA,
                      R2_T=R2_T,R2_I=R2_I,R2_A=R2_A,RMSE_T=RMSE_T,RMSE_I=RMSE_I,RMSE_A=RMSE_A)
    print(f"{Tk}K: RMSE_T={RMSE_T:.4f}({R2_T:.3f}) RMSE_I={RMSE_I:.4f}({R2_I:.3f}) RMSE_A={RMSE_A:.4f}({R2_A:.3f}) "
          f"| mu={mu:.3f} r1={r1:.4f} a={a:.4f} k_fast={k_fast:.4f} | w={w:.4g} kI={kI:.3f} kA={kA:.3f}")

    ax.scatter(tT,Td,s=10,color='#2166ac',alpha=0.55,label='T data',zorder=2)
    ax.plot(t_dense,Tpred_dense,'-',color='#08306b',lw=2.4,label='T model',zorder=3)
    ax.scatter(tI,Id,s=10,color='#b2182b',alpha=0.55,label='I data',zorder=2)
    ax.plot(t_dense,Ipred_dense,'-',color='#67000d',lw=2.4,label='I model',zorder=3)
    ax.scatter(tA,Ad,s=10,color='#1a9850',alpha=0.55,label='A data',zorder=2)
    ax.plot(t_dense,Apred_dense,'-',color='#00441b',lw=2.4,label='A model',zorder=3)
    txt = f"$R^2$(T)={R2_T:.3f}\n$R^2$(I)={R2_I:.3f}\n$R^2$(A)={R2_A:.3f}"
    ax.text(0.97,0.97,txt,transform=ax.transAxes,ha='right',va='top',fontsize=11,
            bbox=dict(boxstyle='round',fc='white',ec='0.6',alpha=0.9))
    ax.set_title(f'{Tk} K', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (hr)', fontsize=12)
    ax.set_xlim(-2,72); ax.set_ylim(-0.05,1.05)
    ax.tick_params(labelsize=10)
axes[0].set_ylabel('Normalized population', fontsize=12)
handles,labels = axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',ncol=6,fontsize=11,bbox_to_anchor=(0.5,1.06),frameon=False)
plt.tight_layout()
plt.savefig('fit_result_overlay_mixture.png', dpi=160, bbox_inches='tight')
print("\nsaved fit_result_overlay_mixture.png")
