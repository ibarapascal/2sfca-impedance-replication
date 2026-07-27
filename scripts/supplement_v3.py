"""
v3 补充计算批 — 消化两份对抗审的全部可计算项（审稿映射见各节注释）
数据: full2/{region}_mesh.parquet (8规格A+Q) + full2/pairs_{region}/ + inversion/ + ptdata/
输出: OUT/supplement_results.json + 逐节打印
预算: Part1 后处理 <5min | Part2 pairs 复用 ~30-40min | Part3 反演侧 ~10min。内存 <2GB。
"""
import numpy as np, pandas as pd, json, os, sys, glob, time
sys.stdout.reconfigure(line_buffering=True)

BASE = os.environ.get("WORK_ROOT", "./work")
OUT = {}
REGIONS = ["shutoken", "osaka", "fukuoka"]
NAMES6 = ["negexp_b001", "negexp_b003", "power_15", "gauss_s1500", "rect", "zone_e2sfca"]
SUBSET_B = ["negexp_b001", "gauss_s1500", "zone_e2sfca"]  # 行为邻近子集
BBOX = {"shutoken": ((34.9, 36.3), (138.9, 140.9)), "osaka": ((34.2, 35.0), (135.0, 135.8)),
        "fukuoka": ((33.0, 33.8), (130.0, 131.0))}
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}")

def quint(a):
    n = len(a); o = np.argsort(a, kind="stable")
    r = np.empty(n, dtype=np.int64); r[o] = np.arange(n)
    return (r * 5 // n).astype(np.int8), r / n * 100  # class, percentile

def flip_stats(Q):
    qmin, qmax = Q.min(axis=0), Q.max(axis=0)
    return float((qmax != qmin).mean() * 100), float(((qmax - qmin) >= 2).mean() * 100), qmin, qmax

# ============ Part 1: mesh parquet 后处理 ============
mesh_cache = {}
for reg in REGIONS:
    df = pd.read_parquet(f"{BASE}/full2/{reg}_mesh.parquet")
    mesh_cache[reg] = df
    pop = df.pop_total.values.astype(float)
    A6 = np.vstack([df[f"Anet_{n}"].values for n in NAMES6])
    QP = [quint(A6[s]) for s in range(6)]
    Q6 = np.vstack([q for q, _ in QP]); P6 = np.vstack([p for _, p in QP])
    r = {}

    # --- [methods P0-1 / position P0-2] 子集口径头条 ---
    idxB = [NAMES6.index(n) for n in SUBSET_B]
    anyB, ge2B, _, _ = flip_stats(Q6[idxB])
    any6, ge26, qmin6, qmax6 = flip_stats(Q6)
    r["anyflip6"] = any6; r["anyflip_subsetB"] = anyB; r["ge2_subsetB"] = ge2B
    # desert margin 两口径
    for tag, Q_ in [("6", Q6), ("B", Q6[idxB])]:
        bot = Q_ == 0; ba, bl = bot.any(0), bot.all(0)
        r[f"desert_union_{tag}"] = int(ba.sum()); r[f"desert_unstable_pct_{tag}"] = float((ba & ~bl).sum() / max(ba.sum(), 1) * 100)
        r[f"desert_margin_pop_{tag}"] = float(pop[ba & ~bl].sum())

    # --- [methods P0-4] percentile shift 分布 + 边界分解 ---
    shift6 = P6.max(0) - P6.min(0)
    r["pshift_median"] = float(np.median(shift6)); r["pshift_p90"] = float(np.percentile(shift6, 90))
    flipped = qmax6 != qmin6
    r["flip_with_shift_lt2pct"] = float((shift6[flipped] < 2).mean() * 100)  # 边界薄翻转占比
    r["flip_with_shift_ge10pct"] = float((shift6[flipped] >= 10).mean() * 100)

    # --- [position P0-1] 年龄分层（需 e2sfca 表 elder 字段 join）---
    try:
        e2f = {"shutoken": "kanto", "osaka": "kansai", "fukuoka": "kyushu"}[reg]
        e2 = pd.read_parquet(f"${DATA_ROOT}/locanex/storage/structured/analysis/dim1_facility/e2sfca_{e2f}.parquet",
                             columns=["mesh_code", "pop_65_over", "pop_total"])
        m = df[["mesh_code"]].merge(e2, on="mesh_code", how="left")  # left join 保序保长
        assert len(m) == len(df)
        p65 = m.pop_65_over.fillna(0).values; ptot = m.pop_total.fillna(0).values
        bot6 = Q6 == 0; ba, bl = bot6.any(0), bot6.all(0)
        groups = {"stable_desert_core": bl, "unstable_margin": ba & ~bl, "stable_nondesert": ~ba}
        r["age_strata"] = {g: {"n_cells": int(mask.sum()),
                               "elder_share_pct": float(p65[mask].sum() / max(ptot[mask].sum(), 1) * 100)}
                           for g, mask in groups.items()}
    except Exception as e:
        r["age_strata"] = {"error": str(e)}

    # --- [position P0-4] 绝对阈值荒漠（HPSA 型 1:3500）---
    for thr_name, thr in [("hpsa3500", 1/3500.0), ("r2500", 1/2500.0)]:
        M6 = A6 < thr
        u, i = M6.any(0), M6.all(0)
        r[f"absthr_{thr_name}"] = {"union_cells": int(u.sum()),
            "unstable_pct": float((u & ~i).sum() / max(u.sum(), 1) * 100),
            "margin_pop": float(pop[u & ~i].sum()),
            "per_spec_desert_pct": {NAMES6[s]: float(M6[s].mean() * 100) for s in range(6)}}

    # --- [methods P1-7] 边界 buffer 稳健性（剔除距 bbox 边 5km 内 cell）---
    (bla, blo) = BBOX[reg]
    dlat = 5000 / 111320.0; dlon = 5000 / (111320.0 * np.cos(np.radians((bla[0]+bla[1])/2)))
    inner = ((df.lat >= bla[0]+dlat) & (df.lat <= bla[1]-dlat) &
             (df.lon >= blo[0]+dlon) & (df.lon <= blo[1]-dlon)).values
    Q6i = np.vstack([quint(A6[s][inner])[0] for s in range(6)])
    anyI, _, _, _ = flip_stats(Q6i)
    r["anyflip6_inner"] = anyI; r["inner_cells_pct"] = float(inner.mean() * 100)

    # --- [position P1-6] 500m 聚合（compound vs overlap 检验素材）---
    mc = df.mesh_code.astype(str)
    if mc.str.len().iloc[0] >= 9:
        c500 = mc.str[:9]
        agg = {}
        for s, n in enumerate(NAMES6):
            agg[n] = pd.Series(A6[s] * pop).groupby(c500.values).sum() / pd.Series(pop).groupby(c500.values).sum().replace(0, np.nan)
        A500 = np.vstack([agg[n].fillna(0).values for n in NAMES6])
        Q500 = np.vstack([quint(A500[s])[0] for s in range(6)])
        any500, _, _, _ = flip_stats(Q500)
        r["anyflip6_500m"] = any500; r["n_500m_cells"] = int(A500.shape[1])
    OUT[reg] = r
    log(f"{reg} part1 done: any6={any6:.1f} subsetB={anyB:.1f} shift_med={r['pshift_median']:.1f}pct")

# --- pairwise 中位数/IQR [methods P0-1, position P1-4] ---
for reg in REGIONS:
    rj = json.load(open(f"{BASE}/full2/{reg}_results.json"))
    pf = rj["network"]["pairwise_flip5_pct"]
    v = [pf[f"{a}|{b}"] for i, a in enumerate(NAMES6) for b in NAMES6[i+1:]]
    OUT[reg]["pairwise15_median"] = float(np.median(v)); OUT[reg]["pairwise15_iqr"] = [float(np.percentile(v, 25)), float(np.percentile(v, 75))]

# --- [位置 P1-9] 人口增减 join（尝试 2021_T001162=2015 人口?）---
try:
    old = pd.read_parquet("${DATA_ROOT}/locanex/storage/structured/snapshots/estat/2021_T001162.parquet")
    log(f"T001162 cols: {list(old.columns)[:6]} rows={len(old)}")
    OUT["popchange_note"] = f"T001162 cols {list(old.columns)[:6]}"
except Exception as e:
    OUT["popchange_note"] = f"unavailable: {e}"

# ============ Part 2: pairs 复用（微扰 floor / clamp / 3km / 容量代理）============
def phase_b_specs(reg, spec_funcs, d_max=None, fac_weight=None):
    """通用 phase-B: spec_funcs={name:fn}; d_max 截断; fac_weight 替代 fac_cnt(按 ufac 序)"""
    pdir = f"{BASE}/full2/pairs_{reg}"
    meta = np.load(f"{pdir}/meta.npz")
    ufac, fac_cnt = meta["ufac"], meta["fac_cnt"].astype(float)
    w_fac = fac_weight if fac_weight is not None else fac_cnt
    df = mesh_cache[reg]; pop = df.pop_total.values.astype(float); n_mesh = len(df)
    chunks = sorted(glob.glob(f"{pdir}/c*.npz"))
    names = list(spec_funcs)
    denom = {n: np.zeros(len(ufac)) for n in names}
    for cf in chunks:
        z = np.load(cf); src, mi, d = z["src"].astype(np.int64), z["mesh"].astype(np.int64), z["d"].astype(float)
        if d_max: keep = d <= d_max; src, mi, d = src[keep], mi[keep], d[keep]
        for n in names:
            denom[n] += np.bincount(src, weights=pop[mi] * spec_funcs[n](d) * 1.0, minlength=len(ufac))
    Rj = {n: np.where(denom[n] > 0, 1.0/np.where(denom[n] > 0, denom[n], 1), 0) for n in names}
    A = {n: np.zeros(n_mesh) for n in names}
    for cf in chunks:
        z = np.load(cf); src, mi, d = z["src"].astype(np.int64), z["mesh"].astype(np.int64), z["d"].astype(float)
        if d_max: keep = d <= d_max; src, mi, d = src[keep], mi[keep], d[keep]
        for n in names:
            A[n] += np.bincount(mi, weights=w_fac[src] * Rj[n][src] * spec_funcs[n](d), minlength=n_mesh)
    return A

SPEC_FN = {"negexp_b001": lambda d: np.exp(-0.001*d), "negexp_b003": lambda d: np.exp(-0.003*d),
           "power_15": lambda d: np.clip(d,125,None)**-1.5, "gauss_s1500": lambda d: np.exp(-d**2/(2*1500**2)),
           "rect": lambda d: np.ones_like(d), "zone_e2sfca": lambda d: np.where(d<=1000,1.0,np.where(d<=3000,0.68,np.where(d<=5000,0.22,0.0)))}

# --- [methods P0-4] 微扰 floor + [methods P1-2] clamp 敏感性（fukuoka 代表,省时）---
reg = "fukuoka"
pert = phase_b_specs(reg, {"b0011": lambda d: np.exp(-0.0011*d), "gauss1400": lambda d: np.exp(-d**2/(2*1400**2)),
                            "power250": lambda d: np.clip(d,250,None)**-1.5, "power500": lambda d: np.clip(d,500,None)**-1.5})
df = mesh_cache[reg]
base = {"b0011": "negexp_b001", "gauss1400": "gauss_s1500", "power250": "power_15", "power500": "power_15"}
OUT["perturbation_floor_fukuoka"] = {}
for pname, bname in base.items():
    Qp, _ = quint(pert[pname]); Qb, _ = quint(df[f"Anet_{bname}"].values)
    OUT["perturbation_floor_fukuoka"][f"{pname}_vs_{bname}"] = float((Qp != Qb).mean() * 100)
log(f"perturbation floor: {OUT['perturbation_floor_fukuoka']}")

# --- [methods P1-2] 3km catchment 敏感性（fukuoka）---
A3 = phase_b_specs(reg, SPEC_FN, d_max=3000.0)
Q3 = np.vstack([quint(A3[n])[0] for n in NAMES6])
any3, _, _, _ = flip_stats(Q3)
OUT["anyflip6_3km_fukuoka"] = any3
log(f"3km catchment fukuoka anyflip: {any3:.1f}")

# --- [methods P1-1 / position P0-3] 容量代理（fukuoka: 名称含病院→10 else 1）---
try:
    fac = pd.read_parquet("${DATA_ROOT}/locanex/storage/structured/analysis/dim1_facility/facilities_all.parquet",
                          columns=["category","lat","lon","name"])
    (bla, blo) = BBOX[reg]
    fac = fac[(fac.category=="hospital")&(fac.lat>=bla[0])&(fac.lat<=bla[1])&(fac.lon>=blo[0])&(fac.lon<=blo[1])].reset_index(drop=True)
    cap = np.where(fac.name.astype(str).str.contains("病院"), 10.0, 1.0)
    OUT["capacity_proxy_fukuoka_hospital_share"] = float((cap==10).mean()*100)
    # 重snap到 kyushu 子图节点(仅节点,无需边)
    nodes = pd.read_parquet("${DATA_ROOT}/locanex/storage/structured/osm/walk_nodes_kyushu.parquet")
    keep = ((nodes.lat>=bla[0]-0.06)&(nodes.lat<=bla[1]+0.06)&(nodes.lon>=blo[0]-0.06)&(nodes.lon<=blo[1]+0.06)).values
    nodes = nodes[keep].reset_index(drop=True)
    from scipy.spatial import cKDTree
    tree = cKDTree(np.column_stack([nodes.lon.values, nodes.lat.values]))
    _, fnode = tree.query(np.column_stack([fac.lon.values, fac.lat.values]))
    meta = np.load(f"{BASE}/full2/pairs_{reg}/meta.npz"); ufac = meta["ufac"]
    pos = {u: i for i, u in enumerate(ufac)}
    capw = np.zeros(len(ufac))
    matched = 0
    for fn_, c in zip(fnode, cap):
        if fn_ in pos: capw[pos[fn_]] += c; matched += 1
    OUT["capacity_snap_match_pct"] = float(matched/len(fac)*100)
    Acap = phase_b_specs(reg, SPEC_FN, fac_weight=capw)
    Qc = np.vstack([quint(Acap[n])[0] for n in NAMES6])
    anyC, _, _, _ = flip_stats(Qc)
    OUT["anyflip6_capacity_fukuoka"] = anyC
    # 与 baseline 同规格分类偏移
    OUT["capacity_vs_unit_flip_per_spec"] = {n: float((quint(Acap[n])[0] != quint(df[f"Anet_{n}"].values)[0]).mean()*100) for n in NAMES6}
    log(f"capacity proxy: anyflip={anyC:.1f}, snap match={OUT['capacity_snap_match_pct']:.1f}%")
except Exception as e:
    OUT["capacity_proxy_error"] = str(e); log(f"capacity proxy failed: {e}")

# ============ Part 3: 反演侧 ============
import shapefile
from scipy.optimize import curve_fit
PT = f"{BASE}/ptdata"
r = shapefile.Reader(f"{PT}/H30_kzone", encoding="cp932")
zrec = []
for sr in r.iterShapeRecords():
    pts = np.array(sr.shape.points); parts = list(sr.shape.parts)+[len(pts)]
    Aa=0; cxn=0; cyn=0
    for pi in range(len(parts)-1):
        ring=pts[parts[pi]:parts[pi+1]]; x,y=ring[:,0],ring[:,1]
        cr=np.dot(x,np.roll(y,-1))-np.dot(y,np.roll(x,-1)); a=0.5*cr; Aa+=a
        cxn+=np.sum((x+np.roll(x,-1))*(x*np.roll(y,-1)-np.roll(x,-1)*y))/6
        cyn+=np.sum((y+np.roll(y,-1))*(x*np.roll(y,-1)-np.roll(x,-1)*y))/6
    cx,cy=(cxn/Aa,cyn/Aa) if abs(Aa)>1e-6 else (pts[:,0].mean(),pts[:,1].mean())
    zrec.append((int(sr.record[0]),cx,cy,abs(Aa)))
zdf = pd.DataFrame(zrec, columns=["kzone","cx","cy","area"]).drop_duplicates("kzone").set_index("kzone")
od = pd.read_csv(f"{PT}/pt_od.csv", skiprows=4, header=None, encoding="cp932",
    names=["orig","dest","purpose","rail","bus","car","moto","bike","walk","other","unknown","total"]).dropna(subset=["orig","dest","purpose"])
def zc(s):
    s=str(s).strip().lstrip(":").strip(); return int(s) if s.isdigit() else -1
od["o"]=od.orig.map(zc); od["d"]=od.dest.map(zc)
for c in ["walk","total"]: od[c]=pd.to_numeric(od[c],errors="coerce").fillna(0)
od = od[(od.o>0)&(od.d>0)&od.o.isin(zdf.index)&od.d.isin(zdf.index)]
zid = {k:i for i,k in enumerate(zdf.index)}
n_z = len(zdf); cx,cy,ar = zdf.cx.values, zdf.cy.values, zdf.area.values
D = np.hypot(cx[:,None]-cx[None,:], cy[:,None]-cy[None,:]); np.fill_diagonal(D, 0.5*np.sqrt(ar))
BANDS = np.array([0,1,2,3,4,5,7.5,10,15,20,30,50])*1000.0
band_idx = np.digitize(D,BANDS)-1; valid=(band_idx>=0)&(band_idx<len(BANDS)-1); nb=len(BANDS)-1
mids = 0.5*(BANDS[:-1]+BANDS[1:])

def invert_T(T, n_iter=40, bi=None, va=None):
    bi = band_idx if bi is None else bi; va = valid if va is None else va
    bflat = np.where(va, bi, nb).ravel()  # 无效对归入第 nb 桶
    O,Dd = T.sum(1),T.sum(0)
    obs = np.bincount(bflat, weights=T.ravel(), minlength=nb+1)[:nb]
    f=np.ones(nb); Ai=np.ones(n_z); Bj=np.ones(n_z)
    for _ in range(n_iter):
        F=np.where(va,f[np.clip(bi,0,nb-1)],0.0)
        M=(Ai*O)[:,None]*(Bj*Dd)[None,:]*F
        rs=M.sum(1); Ai*=np.where(rs>0,O/np.where(rs>0,rs,1),1)
        M=(Ai*O)[:,None]*(Bj*Dd)[None,:]*F
        cs=M.sum(0); Bj*=np.where(cs>0,Dd/np.where(cs>0,cs,1),1)
        M=(Ai*O)[:,None]*(Bj*Dd)[None,:]*F
        mod=np.bincount(bflat, weights=M.ravel(), minlength=nb+1)[:nb]
        f*=np.where(mod>0,obs/np.where(mod>0,mod,1),1); f/= f[0] if f[0]>0 else 1
    return f, obs

def build_T(sub, col):
    T=np.zeros((n_z,n_z)); np.add.at(T,(sub.o.map(zid).values,sub.d.map(zid).values),sub[col].values); return T

def tail_stats(f, obs):
    ok=(f>0)&(obs>0)&(mids>=3000)
    beta_lin=np.nan; beta_log=np.nan
    try:
        p,_=curve_fit(lambda d,a,b:a*np.exp(-b*d),mids[ok],f[ok],p0=[1,3e-4],bounds=([1e-6,1e-6],[1e3,1e-2]),sigma=1/np.sqrt(obs[ok]),maxfev=20000)
        beta_lin=p[1]
    except Exception: pass
    try:
        pl=np.polyfit(mids[ok],np.log(f[ok]),1,w=np.sqrt(obs[ok])); beta_log=-pl[0]
    except Exception: pass
    # 无族 tail log-slope: 3-10km 端点带
    i1 = 3  # band [3,4) mid 3.5
    i2 = 7  # band [10,15) mid 12.5 → 用 [7.5,10) mid 8.75 更稳: index 6
    i2 = 6
    slope = np.nan
    if f[i1]>0 and f[i2]>0: slope = -(np.log(f[i2])-np.log(f[i1]))/(mids[i2]-mids[i1])
    return beta_lin, beta_log, slope

pri = od[od.purpose=="自宅－私事"]; com = od[od.purpose=="自宅－勤務"]
Tw = build_T(pri,"walk"); Tt = build_T(pri,"total"); Tc = build_T(com,"total"); Tcw = build_T(com,"walk")
fw,obsw = invert_T(Tw); ft,obst = invert_T(Tt); fc,obsc = invert_T(Tc); fcw,obscw = invert_T(Tcw)
bl,bg,sl = tail_stats(fw,obsw)
OUT["tail_walk"] = {"beta_lin": bl, "beta_log": bg, "modelfree_slope_3p5_8p75km": sl}
_,_,sl_t = tail_stats(ft,obst); _,_,sl_c = tail_stats(fc,obsc); _,_,sl_cw = tail_stats(fcw,obscw)
OUT["tail_slopes_modelfree"] = {"personal_allmode": sl_t, "commute_allmode": sl_c,
                                 "personal_walk": sl, "commute_walk": sl_cw,
                                 "purpose_ratio_allmode": sl_t/sl_c if sl_c else None,
                                 "purpose_ratio_walk": sl/sl_cw if sl_cw else None}
log(f"tail slopes: walk {sl:.5f} personal_all {sl_t:.6f} commute_all {sl_c:.6f}")

# --- [methods P0-5] bootstrap β̂（zone cluster, 200 reps）---
rng = np.random.default_rng(20260726)
bls, bgs, sls = [], [], []
for rep in range(200):
    idx = rng.integers(0, n_z, n_z)
    Tb = Tw[np.ix_(idx, idx)]
    Db = D[np.ix_(idx, idx)]
    bi = np.digitize(Db,BANDS)-1; va=(bi>=0)&(bi<nb)
    f, obs = invert_T(Tb, n_iter=25, bi=bi, va=va)
    ok=(f>0)&(obs>0)&(mids>=3000)
    if ok.sum()>=3:
        try:
            pl=np.polyfit(mids[ok],np.log(f[ok]),1,w=np.sqrt(obs[ok])); bgs.append(-pl[0])
        except Exception: pass
        if f[3]>0 and f[6]>0: sls.append(-(np.log(f[6])-np.log(f[3]))/(mids[6]-mids[3]))
    if rep%50==0: log(f"bootstrap rep {rep}")
OUT["bootstrap_walk"] = {"beta_log_ci95": [float(np.percentile(bgs,2.5)), float(np.percentile(bgs,97.5))],
                          "slope_ci95": [float(np.percentile(sls,2.5)), float(np.percentile(sls,97.5))],
                          "n_valid": len(bgs)}
log(f"bootstrap: slope CI {OUT['bootstrap_walk']['slope_ci95']}")

# --- [methods P1-9] 目的地设施密度分层反演 ---
try:
    fac_all = pd.read_parquet("${DATA_ROOT}/locanex/storage/structured/analysis/dim1_facility/facilities_all.parquet",
                              columns=["category","lat","lon"])
    fh = fac_all[fac_all.category=="hospital"]
    # PT zone 为 JGD2011 zone9 平面坐标; 设施经纬度需转换。近似: 用 pyproj 若有,否则跳过
    try:
        from pyproj import Transformer
        tr = Transformer.from_crs("EPSG:4612","EPSG:6677",always_xy=True)  # zone9
        fx, fy = tr.transform(fh.lon.values, fh.lat.values)
        from scipy.spatial import cKDTree as KDT
        zt = KDT(np.column_stack([cx,cy]))
        _, zn = zt.query(np.column_stack([fx,fy]))
        cnt = np.bincount(zn, minlength=n_z)
        dens = cnt / (ar/1e6)
        hi = dens >= np.median(dens)
        res_strat = {}
        for tag, mask in [("high_density_dest", hi), ("low_density_dest", ~hi)]:
            Tm = Tw.copy(); Tm[:, ~mask] = 0
            fm, om = invert_T(Tm)
            _,_,s_ = tail_stats(fm, om)
            res_strat[tag] = s_
        OUT["endogeneity_strata_slope"] = res_strat
        log(f"endogeneity strata: {res_strat}")
    except ImportError:
        OUT["endogeneity_strata_slope"] = "pyproj unavailable, skipped"
except Exception as e:
    OUT["endogeneity_strata_slope"] = f"error {e}"

# --- [methods P0-4] null flip floor: 给定 Spearman ρ 的期望五分位分歧 ---
def null_flip(rho, n=500000):
    z1 = rng.standard_normal(n); z2 = rho*z1 + np.sqrt(1-rho**2)*rng.standard_normal(n)
    q1,_ = quint(z1); q2,_ = quint(z2)
    return float((q1!=q2).mean()*100)
OUT["null_flip_by_rho"] = {str(rh): null_flip(rh) for rh in [0.98, 0.9, 0.8, 0.6, 0.4]}
log(f"null flips: {OUT['null_flip_by_rho']}")

# --- walk 份额 [position P1-1] ---
OUT["walk_share_personal_pct"] = float(pri.walk.sum()/max(pri.total.sum(),1)*100)

with open(f"{BASE}/supplement_results.json","w") as f:
    json.dump(OUT, f, indent=2, ensure_ascii=False, default=float)
log("ALL DONE")
print(json.dumps({k:v for k,v in OUT.items() if k in ["perturbation_floor_fukuoka","null_flip_by_rho","tail_slopes_modelfree"]}, indent=2, default=float))
