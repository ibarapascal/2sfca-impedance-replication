"""
xx-impedance-standard Pilot: 阻抗函数选择的翻转率量化（首都圏・医疗设施）

6 种阻抗规格各跑一遍 2SFCA（Euclidean 距离 pilot 版，路网版留 upgrade），
量化 mesh 级可达性分位数分类在规格间的翻转率。

规格（共用 d_max=5km catchment）:
  E1 negexp β=0.001 | E2 negexp β=0.003 | P  power d^-1.5 (clamp 125m)
  G  gaussian σ=1500 | R  rectangular    | Z  E2SFCA 三区权重 (1.0/0.68/0.22)

复杂度: ~95k 设施 × 5km 邻域 mesh，分块两遍，内存 <1GB。
输出: pilot_results.json + mesh_accessibility.parquet + flip_map.png
"""
import numpy as np, pandas as pd, json, time, sys, os

sys.stdout.reconfigure(line_buffering=True)
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

STORAGE = "${DATA_ROOT}/locanex/storage"
OUT = os.environ.get("WORK_ROOT", "./work")
os.makedirs(OUT, exist_ok=True)

BB_LAT, BB_LON = (34.9, 36.3), (138.9, 140.9)
D_MAX = 5000.0
M_LAT = 111320.0
CHUNK = 500

# === 阻抗规格 ===
def w_negexp_slow(d): return np.exp(-0.001 * d)
def w_negexp_fast(d): return np.exp(-0.003 * d)
def w_power(d):       return np.clip(d, 125.0, None) ** -1.5
def w_gauss(d):       return np.exp(-d**2 / (2 * 1500.0**2))
def w_rect(d):        return np.ones_like(d)
def w_zone(d):
    w = np.zeros_like(d)
    w[d <= 1000] = 1.0
    w[(d > 1000) & (d <= 3000)] = 0.68
    w[(d > 3000) & (d <= 5000)] = 0.22
    return w

SPECS = [
    ("negexp_b001", w_negexp_slow), ("negexp_b003", w_negexp_fast),
    ("power_15", w_power), ("gauss_s1500", w_gauss),
    ("rect", w_rect), ("zone_e2sfca", w_zone),
]

t0 = time.time()
def log(msg): print(f"[{time.time()-t0:7.1f}s] {msg}")

# === 数据加载 ===
log("loading mesh...")
e2 = pd.read_parquet(f"{STORAGE}/structured/analysis/dim1_facility/e2sfca_kanto.parquet",
                     columns=["mesh_code", "lon", "lat", "pop_total"])
mesh = e2[(e2.lat >= BB_LAT[0]) & (e2.lat <= BB_LAT[1]) &
          (e2.lon >= BB_LON[0]) & (e2.lon <= BB_LON[1])].reset_index(drop=True)
log(f"mesh: {len(mesh)}, pop: {mesh.pop_total.sum():.0f}")

log("loading facilities...")
fac = pd.read_parquet(f"{STORAGE}/structured/analysis/dim1_facility/facilities_all.parquet")
fac = fac[(fac.category == "hospital") &
          (fac.lat >= BB_LAT[0]) & (fac.lat <= BB_LAT[1]) &
          (fac.lon >= BB_LON[0]) & (fac.lon <= BB_LON[1])].reset_index(drop=True)
log(f"hospital facilities: {len(fac)}")

ref_lat = 35.7
m_lon = M_LAT * np.cos(np.radians(ref_lat))
mesh_xy = np.column_stack([mesh.lon.values * m_lon, mesh.lat.values * M_LAT])
fac_xy = np.column_stack([fac.lon.values * m_lon, fac.lat.values * M_LAT])
pop = mesh.pop_total.values.astype(np.float64)

mesh_tree = cKDTree(mesh_xy)
n_fac, n_mesh, n_spec = len(fac_xy), len(mesh_xy), len(SPECS)

# === Pass 1: 各设施 Rj = 1 / Σ_i P_i·w(d_ij)（6 规格同时）===
log("pass 1: Rj denominators...")
Rj = np.zeros((n_spec, n_fac))
total_pairs = 0
for c0 in range(0, n_fac, CHUNK):
    idx_lists = mesh_tree.query_ball_point(fac_xy[c0:c0+CHUNK], D_MAX)
    for k, idxs in enumerate(idx_lists):
        if not idxs: continue
        idxs = np.asarray(idxs)
        d = np.linalg.norm(mesh_xy[idxs] - fac_xy[c0+k], axis=1)
        total_pairs += len(d)
        p = pop[idxs]
        for s, (_, wf) in enumerate(SPECS):
            denom = np.sum(p * wf(d))
            if denom > 0: Rj[s, c0+k] = 1.0 / denom
    if (c0 // CHUNK) % 20 == 0:
        log(f"  pass1 {c0}/{n_fac} pairs={total_pairs/1e6:.0f}M")
log(f"pass 1 done, total pairs {total_pairs/1e6:.0f}M")

# === Pass 2: A_i = Σ_j Rj·w(d_ij) 散射回 mesh ===
log("pass 2: accessibility scatter...")
A = np.zeros((n_spec, n_mesh))
for c0 in range(0, n_fac, CHUNK):
    idx_lists = mesh_tree.query_ball_point(fac_xy[c0:c0+CHUNK], D_MAX)
    for k, idxs in enumerate(idx_lists):
        if not idxs: continue
        idxs = np.asarray(idxs)
        d = np.linalg.norm(mesh_xy[idxs] - fac_xy[c0+k], axis=1)
        for s, (_, wf) in enumerate(SPECS):
            A[s, idxs] += Rj[s, c0+k] * wf(d)
    if (c0 // CHUNK) % 20 == 0:
        log(f"  pass2 {c0}/{n_fac}")
log("pass 2 done")

# === 分位数分类与翻转率 ===
log("computing flip rates...")
names = [n for n, _ in SPECS]
Q = np.zeros((n_spec, n_mesh), dtype=np.int8)
for s in range(n_spec):
    # 等频五分位（rank-based，处理 ties 用序号）
    order = np.argsort(A[s], kind="stable")
    ranks = np.empty(n_mesh, dtype=np.int64); ranks[order] = np.arange(n_mesh)
    Q[s] = (ranks * 5 // n_mesh).astype(np.int8)

zero_counts = {names[s]: int((A[s] == 0).sum()) for s in range(n_spec)}

# 主指标：任意两规格间分位数类不一致的 mesh 比例
qmin, qmax = Q.min(axis=0), Q.max(axis=0)
any_flip = qmax != qmin
range2 = (qmax - qmin) >= 2
res = {
    "n_mesh": int(n_mesh), "n_fac": int(n_fac), "total_pop": float(pop.sum()),
    "d_max_m": D_MAX, "specs": names, "zero_access_counts": zero_counts,
    "flip_any_pct": float(any_flip.mean() * 100),
    "flip_any_popweighted_pct": float(pop[any_flip].sum() / pop.sum() * 100),
    "flip_range_ge2_pct": float(range2.mean() * 100),
    "flip_range_ge2_popweighted_pct": float(pop[range2].sum() / pop.sum() * 100),
}

# pairwise 翻转矩阵 + Spearman
pw_flip = np.zeros((n_spec, n_spec)); pw_rho = np.ones((n_spec, n_spec))
for a in range(n_spec):
    for b in range(a+1, n_spec):
        f = float((Q[a] != Q[b]).mean() * 100)
        pw_flip[a, b] = pw_flip[b, a] = f
        rho = spearmanr(A[a], A[b]).statistic
        pw_rho[a, b] = pw_rho[b, a] = rho
res["pairwise_flip_pct"] = {f"{names[a]}|{names[b]}": round(pw_flip[a, b], 2)
                            for a in range(n_spec) for b in range(a+1, n_spec)}
res["pairwise_spearman"] = {f"{names[a]}|{names[b]}": round(float(pw_rho[a, b]), 4)
                            for a in range(n_spec) for b in range(a+1, n_spec)}

# 政策敏感指标：最差 20%（可达性荒漠）成员资格翻转
bottom = Q == 0
bot_any = bottom.any(axis=0); bot_all = bottom.all(axis=0)
res["bottom20_unstable_pct_of_union"] = float((bot_any & ~bot_all).sum() / max(bot_any.sum(), 1) * 100)
res["bottom20_union_minus_core_mesh"] = int((bot_any & ~bot_all).sum())

with open(f"{OUT}/pilot_results.json", "w") as f:
    json.dump(res, f, indent=2, ensure_ascii=False)

df = mesh[["mesh_code", "lon", "lat", "pop_total"]].copy()
for s, n in enumerate(names):
    df[f"A_{n}"] = A[s]; df[f"Q_{n}"] = Q[s]
df["q_range"] = qmax - qmin
df.to_parquet(f"{OUT}/mesh_accessibility.parquet", index=False)

# === 翻转热力图（Fig.1 草稿）===
log("rendering flip map...")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 8), dpi=130)
sc = ax.scatter(df.lon, df.lat, c=df.q_range, s=0.3, cmap="YlOrRd",
                vmin=0, vmax=4, linewidths=0)
plt.colorbar(sc, label="quintile class range across 6 impedance specs")
ax.set_title(f"Impedance-choice flip map (hospital 2SFCA, shutoken)\n"
             f"any-flip {res['flip_any_pct']:.1f}% | range>=2 {res['flip_range_ge2_pct']:.1f}%")
ax.set_aspect(1 / np.cos(np.radians(ref_lat)))
plt.tight_layout(); plt.savefig(f"{OUT}/flip_map.png"); plt.close()

log("done")
print(json.dumps(res, indent=2, ensure_ascii=False))
