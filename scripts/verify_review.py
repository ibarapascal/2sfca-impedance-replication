"""
Review 定点验证: 对 pilot 计算的 6 个疑点做实测（不重算全量）。
V1 zone匹配丢失+50km外份额 | V2 Furness收敛 | V3 重心算法敏感性
V4 拟合空间敏感性 | V5 2SFCA实现 vs paper03参考实现 | V6 tie边界
"""
import numpy as np, pandas as pd, json, sys, os
sys.stdout.reconfigure(line_buffering=True)
import shapefile
from scipy.spatial import cKDTree
from scipy.optimize import curve_fit

BASE = os.environ.get("WORK_ROOT", "./work")
STORAGE = "${DATA_ROOT}/locanex/storage"

print("========== V1: zone 匹配丢失 + >50km 份额 ==========")
r = shapefile.Reader(f"{BASE}/ptdata/H30_kzone", encoding="cp932")
zcodes = set()
zrec = []
for sr in r.iterShapeRecords():
    pts = np.array(sr.shape.points)
    parts = list(sr.shape.parts) + [len(pts)]
    A = 0.0; cx_n = 0.0; cy_n = 0.0
    for pi in range(len(parts) - 1):
        ring = pts[parts[pi]:parts[pi+1]]
        x, y = ring[:, 0], ring[:, 1]
        cr = np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
        a = 0.5 * cr  # 符号保持: 洞会自然为负(如果绕向相反)
        A += a
        cx_n += np.sum((x + np.roll(x, -1)) * (x * np.roll(y, -1) - np.roll(x, -1) * y)) / 6
        cy_n += np.sum((y + np.roll(y, -1)) * (x * np.roll(y, -1) - np.roll(x, -1) * y)) / 6
    k = int(sr.record[0]); zcodes.add(k)
    cxs, cys = (cx_n / A, cy_n / A) if abs(A) > 1e-6 else (pts[:,0].mean(), pts[:,1].mean())
    zrec.append((k, pts[:,0].mean(), pts[:,1].mean(), cxs, cys, abs(A)))
zdf = pd.DataFrame(zrec, columns=["kzone","vx","vy","sx","sy","area"]).drop_duplicates("kzone").set_index("kzone")
shift = np.hypot(zdf.vx-zdf.sx, zdf.vy-zdf.sy)
print(f"zones {len(zdf)}; vertex-mean vs shoelace centroid shift m: p50={shift.median():.0f} p95={shift.quantile(.95):.0f} max={shift.max():.0f}")

od = pd.read_csv(f"{BASE}/ptdata/pt_od.csv", skiprows=4, header=None, encoding="cp932",
    names=["orig","dest","purpose","rail","bus","car","moto","bike","walk","other","unknown","total"])
od = od.dropna(subset=["orig","dest","purpose"])
def z(s):
    s=str(s).strip().lstrip(":").strip()
    return int(s) if s.isdigit() else -1
od["o"]=od.orig.map(z); od["d"]=od.dest.map(z)
od["total"]=pd.to_numeric(od.total,errors="coerce").fillna(0)
od["walkn"]=pd.to_numeric(od.walk,errors="coerce").fillna(0)
pri = od[od.purpose=="自宅－私事"]
numeric = pri[(pri.o>0)&(pri.d>0)]
matched = numeric[numeric.o.isin(zdf.index)&numeric.d.isin(zdf.index)]
print(f"私事 trips: numeric-zone rows total={numeric.total.sum():.0f}, matched={matched.total.sum():.0f}, "
      f"dropped={(numeric.total.sum()-matched.total.sum()):.0f} ({(1-matched.total.sum()/numeric.total.sum())*100:.2f}%)")
unmatched_codes = sorted(set(numeric.o[~numeric.o.isin(zdf.index)]) | set(numeric.d[~numeric.d.isin(zdf.index)]))
print(f"unmatched zone codes n={len(unmatched_codes)} sample={unmatched_codes[:10]}")
# >50km 份额
zid = {k:i for i,k in enumerate(zdf.index)}
cx, cy = zdf.sx.values, zdf.sy.values
D = np.hypot(cx[:,None]-cx[None,:], cy[:,None]-cy[None,:])
np.fill_diagonal(D, 0.5*np.sqrt(zdf.area.values))
oidx = matched.o.map(zid).values; didx = matched.d.map(zid).values
dvec = D[oidx, didx]
far = matched.total.values[dvec > 50000].sum()
print(f"私事 trips beyond 50km band: {far:.0f} ({far/matched.total.sum()*100:.3f}%)")

print("========== V2+V3+V4: Furness 收敛 + 重心/拟合空间敏感性 ==========")
BANDS = np.array([0,1,2,3,4,5,7.5,10,15,20,30,50])*1000.0
def invert(colvals, D):
    n=len(zdf); T=np.zeros((n,n))
    np.add.at(T,(oidx,didx),colvals)
    band_idx=np.digitize(D,BANDS)-1
    valid=(band_idx>=0)&(band_idx<len(BANDS)-1)
    nb=len(BANDS)-1
    O,Dd=T.sum(1),T.sum(0)
    obs=np.array([T[valid&(band_idx==b)].sum() for b in range(nb)])
    f=np.ones(nb); Ai=np.ones(n); Bj=np.ones(n)
    for it in range(40):
        F=np.where(valid,f[np.clip(band_idx,0,nb-1)],0.0)
        M=(Ai*O)[:,None]*(Bj*Dd)[None,:]*F
        rs=M.sum(1); Ai*=np.where(rs>0,O/np.where(rs>0,rs,1),1)
        M=(Ai*O)[:,None]*(Bj*Dd)[None,:]*F
        cs=M.sum(0); Bj*=np.where(cs>0,Dd/np.where(cs>0,cs,1),1)
        M=(Ai*O)[:,None]*(Bj*Dd)[None,:]*F
        mod=np.array([M[valid&(band_idx==b)].sum() for b in range(nb)])
        ratio=np.where(mod>0,obs/np.where(mod>0,mod,1),1)
        f*=ratio; f/=f[0] if f[0]>0 else 1
    return f,obs,ratio
walkv = matched.walkn.values
f_w, obs_w, last_ratio = invert(walkv, D)
print(f"Furness final band ratio obs/model (should ~1): {np.round(last_ratio,4)}")
mids=0.5*(BANDS[:-1]+BANDS[1:])
def fit_beta(f,obs,space):
    ok=(f>0)&(obs>0)&(mids>=3000)
    if space=="lin":
        p,_=curve_fit(lambda d,a,b:a*np.exp(-b*d),mids[ok],f[ok],p0=[1,3e-4],
                      bounds=([1e-6,1e-6],[1e3,1e-2]),sigma=1/np.sqrt(obs[ok]),maxfev=20000)
    else:
        p=np.polyfit(mids[ok],np.log(f[ok]),1,w=np.sqrt(obs[ok])); p=[np.exp(p[1]),-p[0]]
    return p[1]
# vertex-mean 版距离
Dv = np.hypot(zdf.vx.values[:,None]-zdf.vx.values[None,:], zdf.vy.values[:,None]-zdf.vy.values[None,:])
np.fill_diagonal(Dv, 0.5*np.sqrt(zdf.area.values))
f_wv, obs_wv, _ = invert(walkv, Dv)
print(f"walk beta_hat: shoelace+lin={fit_beta(f_w,obs_w,'lin'):.6f}  shoelace+log={fit_beta(f_w,obs_w,'log'):.6f}")
print(f"               vertexmean+lin={fit_beta(f_wv,obs_wv,'lin'):.6f}  vertexmean+log={fit_beta(f_wv,obs_wv,'log'):.6f}")

print("========== V5: 两遍实现 vs paper03 参考 e2sfca_loop (fukuoka, b003) ==========")
e2 = pd.read_parquet(f"{STORAGE}/structured/analysis/dim1_facility/e2sfca_kyushu.parquet",
                     columns=["mesh_code","lon","lat","pop_total"])
mesh = e2[(e2.lat>=33.0)&(e2.lat<=33.8)&(e2.lon>=130.0)&(e2.lon<=131.0)].reset_index(drop=True)
fac = pd.read_parquet(f"{STORAGE}/structured/analysis/dim1_facility/facilities_all.parquet",
                      columns=["category","lat","lon"])
fac = fac[(fac.category=="hospital")&(fac.lat>=33.0)&(fac.lat<=33.8)&(fac.lon>=130.0)&(fac.lon<=131.0)].reset_index(drop=True)
m_lon=111320.0*np.cos(np.radians(33.6)); M_LAT=111320.0
mesh_xy=np.column_stack([mesh.lon.values*m_lon,mesh.lat.values*M_LAT])
fac_xy=np.column_stack([fac.lon.values*m_lon,fac.lat.values*M_LAT])
pop=mesh.pop_total.values.astype(np.float64)
D0=5000.0; BETA=0.003
# 参考实现 (paper03 pipeline_v2.e2sfca_loop 逐设施循环)
fac_tree=cKDTree(fac_xy); mesh_tree=cKDTree(mesh_xy)
Rj=np.zeros(len(fac_xy))
for j in range(len(fac_xy)):
    idxs=mesh_tree.query_ball_point(fac_xy[j],D0)
    if not idxs: continue
    idxs=np.array(idxs); dd=np.linalg.norm(mesh_xy[idxs]-fac_xy[j],axis=1)
    den=np.sum(pop[idxs]*np.exp(-BETA*dd))
    if den>0: Rj[j]=1.0/den
A_ref=np.zeros(len(mesh_xy))
for i in range(len(mesh_xy)):
    fi=fac_tree.query_ball_point(mesh_xy[i],D0)
    if not fi: continue
    fi=np.array(fi); dd=np.linalg.norm(fac_xy[fi]-mesh_xy[i],axis=1)
    A_ref[i]=np.sum(Rj[fi]*np.exp(-BETA*dd))
# 我的实现 (chunk 两遍)
Rj2=np.zeros(len(fac_xy))
for c0 in range(0,len(fac_xy),500):
    for k,idxs in enumerate(mesh_tree.query_ball_point(fac_xy[c0:c0+500],D0)):
        if not idxs: continue
        idxs=np.asarray(idxs); dd=np.linalg.norm(mesh_xy[idxs]-fac_xy[c0+k],axis=1)
        den=np.sum(pop[idxs]*np.exp(-BETA*dd))
        if den>0: Rj2[c0+k]=1.0/den
A_mine=np.zeros(len(mesh_xy))
for c0 in range(0,len(fac_xy),500):
    for k,idxs in enumerate(mesh_tree.query_ball_point(fac_xy[c0:c0+500],D0)):
        if not idxs: continue
        idxs=np.asarray(idxs); dd=np.linalg.norm(mesh_xy[idxs]-fac_xy[c0+k],axis=1)
        A_mine[idxs]+=Rj2[c0+k]*np.exp(-BETA*dd)
print(f"max|A_ref-A_mine|={np.abs(A_ref-A_mine).max():.3e}, max rel={np.abs(A_ref-A_mine).max()/max(A_ref.max(),1e-12):.3e}")
print(f"Rj identical: {np.allclose(Rj,Rj2)}")

print("========== V6: 分位 tie 边界规模 (fukuoka euclid b003) ==========")
n=len(A_mine); order=np.argsort(A_mine,kind="stable")
ranks=np.empty(n,dtype=np.int64); ranks[order]=np.arange(n)
Q=(ranks*5//n)
for b in range(1,5):
    cut_val=np.sort(A_mine)[b*n//5]
    ties=(A_mine==cut_val).sum()
    print(f"quintile boundary {b}: value={cut_val:.6g} tied_count={ties}")
print(f"A==0 count: {(A_mine==0).sum()}")
print("done")
