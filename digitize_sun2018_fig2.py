import numpy as np
from PIL import Image
from scipy import ndimage
import csv

im = Image.open('fig02.jpg').convert('RGB')
a = np.array(im).astype(int)

t_ticks = np.array([0,10,20,30,40,50,60,70])
y_ticks = np.array([1,0.8,0.6,0.4,0.2,0])

# box col1 trimmed to ~t=66 hr to exclude the gold OD axis spine at the right edge
panels = {
    'B': dict(origin=(343,0),
              x_px=np.array([78.0,122.5,165.5,207.0,245.0,288.0,331.0,370.0]),
              y_px=np.array([25.5,71.0,114.5,156.5,198.5,242.5]),
              box=(10,238,85,357),
              T_start=298),
    'C': dict(origin=(0,277),
              x_px=np.array([44.5,92.0,134.5,173.5,214.5,256.5,297.5,339.0]),
              y_px=np.array([40.0,85.5,129.5,170.5,213.5,254.5]),
              box=(3,255,50,340),
              T_start=310),
    'D': dict(origin=(343,277),
              x_px=np.array([78.0,122.75,165.5,207.5,245.5,288.5,330.5,370.5]),
              y_px=np.array([40.0,85.5,129.5,170.5,213.5,254.5]),
              box=(3,255,83,357),
              T_start=277),
}

def keep_large_components(mask, min_frac=0.10):
    lbl, n = ndimage.label(mask, structure=np.ones((3,3)))
    if n==0:
        return mask
    sizes = ndimage.sum(mask, lbl, range(1,n+1))
    keep = sizes >= min_frac*sizes.max()
    out = np.zeros_like(mask)
    for i,k in enumerate(keep, start=1):
        if k:
            out |= (lbl==i)
    return out

def reject_outliers(bt, bv, window=7, thresh=0.15):
    bt,bv = np.array(bt), np.array(bv)
    keep = np.ones(len(bt), dtype=bool)
    for i in range(len(bt)):
        lo,hi = max(0,i-window), min(len(bt),i+window+1)
        neigh = np.delete(bv[lo:hi], i-lo)
        if len(neigh)==0: continue
        if abs(bv[i]-np.median(neigh)) > thresh:
            keep[i] = False
    return bt[keep], bv[keep]

results = {}
for name,p in panels.items():
    col_off,row_off = p['origin']
    xfit = np.polyfit(p['x_px'], t_ticks, 1)
    yfit = np.polyfit(p['y_px'], y_ticks, 1)
    r0,r1,c0,c1 = p['box']

    sub = a[row_off+r0:row_off+r1, col_off+c0:col_off+c1]
    R,G,Bc = sub[:,:,0].astype(int), sub[:,:,1].astype(int), sub[:,:,2].astype(int)

    blue = (Bc-R>35) & (Bc>150) & (G>100)
    red  = (R>140) & (G<90) & (Bc<90)
    green= (G-R>15) & (G-Bc>15) & (G>90) & (R<180)
    gold = (R>150) & (G>120) & (R-Bc>50) & (G-Bc>30) & ~blue & ~green

    series = {}
    for label,mask in [('T',blue),('I',red),('A',green),('OD',gold)]:
        mask = keep_large_components(mask, min_frac=0.10)
        rows,cols = np.where(mask)
        if len(rows)==0:
            series[label]=[]; continue
        col_local = cols + c0
        row_local = rows + r0
        t = xfit[0]*col_local + xfit[1]
        val = yfit[0]*row_local + yfit[1]
        order = np.argsort(t)
        t,val = t[order], val[order]
        binw = 0.5
        bins = np.round(t/binw)*binw
        uniq = np.unique(bins)
        bt,bv = [],[]
        for u in uniq:
            sel = bins==u
            bt.append(u); bv.append(np.median(val[sel]))
        bt,bv = reject_outliers(bt,bv, thresh=0.12 if label!='OD' else 0.2)
        series[label] = list(zip(bt.tolist(),bv.tolist()))
    results[name] = series

    with open(f'Sun2018_TTR_panel{name}_{p["T_start"]}K_digitized.csv','w',newline='') as f:
        w=csv.writer(f)
        w.writerow(['species','time_hr','value'])
        for label in ['T','I','A','OD']:
            for t,v in series[label]:
                w.writerow([label, round(t,2), round(v,4)])
    print(name, {k:len(v) for k,v in series.items()})

print("done")
