"""
Final, corrected fit: N1=2 (the effective exponent well-constrained at 310K and 277K,
consistent with -- not contradicted by -- 298K's own weaker-signal data), fixed across all
three temperatures for consistency, replacing the earlier unexamined N1=1 default.
Same 2-stage cascade procedure throughout.
"""
import numpy as np, csv
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
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

def logistic(t, mu, r1):
    return 1.0/(1.0+np.exp(mu)*np.exp(-r1*t))

def fit_stage1(t, Tdata):
    def resid(p):
        mu, r1, T0, Tinf = p
        return (T0-(T0-Tinf)*logistic(t,mu,r1)) - Tdata
    p0 = [3.0, 0.05, Tdata[0], Tdata[-1]]
    sol = least_squares(resid, p0, bounds=([-10,1e-4,0,-0.5],[15,2,1.3,1.0]))
    return sol.x

def r2(y, yhat):
    ss_res = np.sum((y-yhat)**2); ss_tot = np.sum((y-np.mean(y))**2)
    return 1 - ss_res/ss_tot

def m_ode(t, y, xfun, w):
    m = y[0]
    c1 = max(xfun(t)-m, 0.0)
    return [w*c1**POWER]

def fit_stage2(tI, Id, tA, Ad, mu, r1):
    xfun = lambda tt: logistic(tt, mu, r1)
    tmax = max(tI.max(), tA.max())
    def resid(p):
        w = np.exp(p[0]); kI, kA = p[1], p[2]
        t_dense = np.linspace(0, tmax, 800)
        sol = solve_ivp(m_ode, [0,tmax], [0.0], t_eval=t_dense, args=(xfun,w),
                         method='RK45', rtol=1e-9, atol=1e-11, max_step=0.3)
        m_dense = sol.y[0]; x_dense = xfun(t_dense)
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
    mu,r1,T0,Tinf = fit_stage1(tT,Td)
    w,kI,kA,cost = fit_stage2(tI,Id,tA,Ad,mu,r1)

    xfun = lambda tt: logistic(tt,mu,r1)
    tmax = max(tT.max(),tI.max(),tA.max())
    t_dense = np.linspace(0,tmax,800)
    sol = solve_ivp(m_ode,[0,tmax],[0.0],t_eval=t_dense,args=(xfun,w),method='RK45',rtol=1e-9,atol=1e-11,max_step=0.3)
    m_dense=sol.y[0]; x_dense=xfun(t_dense)
    Tpred_dense = T0-(T0-Tinf)*x_dense
    Ipred_dense = kI*np.clip(x_dense-m_dense,0,None)
    Apred_dense = kA*m_dense

    Tpred_atT = T0-(T0-Tinf)*logistic(tT,mu,r1)
    Ipred_atI = kI*np.clip(np.interp(tI,t_dense,x_dense)-np.interp(tI,t_dense,m_dense),0,None)
    Apred_atA = kA*np.interp(tA,t_dense,m_dense)
    R2_T,R2_I,R2_A = r2(Td,Tpred_atT), r2(Id,Ipred_atI), r2(Ad,Apred_atA)

    results[Tk]=dict(mu=mu,r1=r1,T0=T0,Tinf=Tinf,w=w,kI=kI,kA=kA,R2_T=R2_T,R2_I=R2_I,R2_A=R2_A,cost=cost)
    print(f"{Tk}K: R2_T={R2_T:.3f} R2_I={R2_I:.3f} R2_A={R2_A:.3f} | mu={mu:.3f} r1={r1:.4f} T0={T0:.3f} Tinf={Tinf:.3f} | w={w:.4g} kI={kI:.3f} kA={kA:.3f} cost={cost:.5f}")

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
plt.savefig('fit_result_overlay_N1_2.png', dpi=160, bbox_inches='tight')
print("\nsaved fit_result_overlay_N1_2.png")
