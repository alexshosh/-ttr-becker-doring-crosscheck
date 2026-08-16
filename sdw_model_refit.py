"""
Refit SDW's own model (T <-> I <-> A, plain reversible two-step chain, 4 rate
constants k1,km1,k2,km2) to the same digitized T/I/A data used for the mixture
model in ttr_crosscheck_v3.tex, so that RMSE can be compared head-to-head on
equal footing (same data, same metric). SDW's own paper reports rate constants
and van't Hoff free energies, not RMSE, so this comparison has to be built
independently -- it does not exist anywhere else in this project.

ODE (mass-conserving, T+I+A=1 for all t if it holds at t=0):
    dT/dt = -k1*T + km1*I
    dI/dt =  k1*T - km1*I - k2*I + km2*A
    dA/dt =  k2*I - km2*A
Initial condition: (T,I,A)(0) = (1,0,0), same convention as the mixture model.
"""
import csv
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
    T_pred = sol.sol(tT)[0]
    I_pred = sol.sol(tI)[1]
    A_pred = sol.sol(tA)[2]
    return T_pred, I_pred, A_pred

def residuals(params, tT, Td, tI, Id, tA, Ad):
    T_pred, I_pred, A_pred = model_curves(params, tT, tI, tA)
    return np.concatenate([T_pred - Td, I_pred - Id, A_pred - Ad])

def rmse(pred, data):
    return float(np.sqrt(np.mean((pred - data) ** 2)))

# our own model's RMSE, from Table 1 of ttr_crosscheck_v3.tex
ours = {
    277: dict(T=0.0078, I=0.0153, A=0.0213),
    298: dict(T=0.0144, I=0.0108, A=0.0182),
    310: dict(T=0.0150, I=0.0115, A=0.0154),
}

rng = np.random.default_rng(0)
print(f"{'T(K)':>5} {'k1':>8} {'km1':>8} {'k2':>8} {'km2':>8} "
      f"{'RMSE_T':>8} {'RMSE_I':>8} {'RMSE_A':>8} {'cost':>10}")

results = {}
for name, Tk in panels.items():
    d = load(name, Tk)
    tT, Td = d['T'][:, 0], d['T'][:, 1]
    tI, Id = d['I'][:, 0], d['I'][:, 1]
    tA, Ad = d['A'][:, 0], d['A'][:, 1]

    best = None
    for _ in range(40):
        x0 = 10 ** rng.uniform(-2, 0.5, size=4)  # 0.01 .. ~3
        try:
            sol = least_squares(residuals, x0, args=(tT, Td, tI, Id, tA, Ad),
                                 bounds=(1e-5, 20.0), method='trf', max_nfev=5000)
        except Exception:
            continue
        if best is None or sol.cost < best.cost:
            best = sol
    k1, km1, k2, km2 = best.x
    T_pred, I_pred, A_pred = model_curves(best.x, tT, tI, tA)
    rT, rI, rA = rmse(T_pred, Td), rmse(I_pred, Id), rmse(A_pred, Ad)
    results[Tk] = dict(k1=k1, km1=km1, k2=k2, km2=km2, RMSE_T=rT, RMSE_I=rI, RMSE_A=rA)
    print(f"{Tk:>5} {k1:>8.4f} {km1:>8.4f} {k2:>8.4f} {km2:>8.4f} "
          f"{rT:>8.4f} {rI:>8.4f} {rA:>8.4f} {best.cost:>10.6f}")

print("\n=== Head-to-head RMSE: ours vs SDW-model-refit-on-our-data ===")
wins_ours = 0
total = 0
for Tk in [277, 298, 310]:
    for sp in ['T', 'I', 'A']:
        o = ours[Tk][sp]
        s = results[Tk][f'RMSE_{sp}']
        pct = 100 * (s - o) / s  # how much better ours is, relative to SDW-refit's RMSE
        winner = 'ours' if o < s else 'SDW-refit'
        if o < s:
            wins_ours += 1
        total += 1
        print(f"{Tk}K {sp}: ours={o:.4f}  SDW-refit={s:.4f}  winner={winner}  "
              f"(ours better by {pct:.1f}%)" if o < s else
              f"{Tk}K {sp}: ours={o:.4f}  SDW-refit={s:.4f}  winner={winner}  "
              f"(SDW-refit better by {-pct:.1f}%)")
print(f"\nours wins {wins_ours} of {total}")
