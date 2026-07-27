"""
xx-impedance-standard ②: 東京都市圏 PT 調査 OD → 显示性距离衰减曲线反演 (Fig.2)

数据: 表d-1 目的種類別代表交通手段別OD表 (H30/2018, e-Stat 公开 CSV)
      615 計画基本ゾーン × 目的(自宅－私事/自宅－勤務) × 手段(計/自動車/鉄道)
方法: 双约束重力模型 + 分段常数衰减函数 Furness 校准
      f_b 迭代: T*_ij = A_i O_i B_j D_j f(d_ij), f_b *= T_obs_b / T_model_b
      拟合族: negexp / power / gaussian / lognormal(对照), 拟合按带内流量加权
距离: zone 重心 Euclidean (JGD2011 zone9 米坐标), 带内距离 d_ii=0.5√area
复杂度: 615×615 矩阵, 秒级。
输出: decay_inversion.json + fig2_draft.png (empirical vs 6 手选规格)
"""
import numpy as np, pandas as pd, json, sys, os
import shapefile

sys.stdout.reconfigure(line_buffering=True)
from scipy.optimize import curve_fit

BASE = os.environ.get("WORK_ROOT", "./work")
PT = f"{BASE}/ptdata"
OUT = f"{BASE}/inversion"
os.makedirs(OUT, exist_ok=True)

BANDS = np.array([0, 1, 2, 3, 4, 5, 7.5, 10, 15, 20, 30, 50]) * 1000.0  # m
N_ITER = 40

# === zone 重心/面积 ===
r = shapefile.Reader(f"{PT}/H30_kzone", encoding="cp932")
zrec = []
for sr in r.iterShapeRecords():
    pts = np.array(sr.shape.points)
    # 多边形重心近似: 顶点均值(带内误差对 5-50km 带划分无实质影响); 面积用 shoelace
    parts = list(sr.shape.parts) + [len(pts)]
    area = 0.0
    for pi in range(len(parts) - 1):
        ring = pts[parts[pi]:parts[pi+1]]
        x, y = ring[:, 0], ring[:, 1]
        area += 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    zrec.append((int(sr.record[0]), pts[:, 0].mean(), pts[:, 1].mean(), area))
zones = pd.DataFrame(zrec, columns=["kzone", "cx", "cy", "area"]).drop_duplicates("kzone").set_index("kzone")
print(f"zones: {len(zones)}")

# === OD 读取 ===
od = pd.read_csv(f"{PT}/pt_od.csv", skiprows=4, header=None, encoding="cp932",
                 names=["orig", "dest", "purpose", "rail", "bus", "car", "moto",
                        "bike", "walk", "other", "unknown", "total"])
od = od.dropna(subset=["orig", "dest", "purpose"])
def zcode(s):
    s = str(s).strip().lstrip(":").strip()
    return int(s) if s.isdigit() else -1
od["o"] = od.orig.map(zcode); od["d"] = od.dest.map(zcode)
od = od[(od.o > 0) & (od.d > 0) & od.o.isin(zones.index) & od.d.isin(zones.index)]
for c in ["rail", "car", "walk", "total"]:
    od[c] = pd.to_numeric(od[c], errors="coerce").fillna(0)
print(f"od rows (zone-matched): {len(od)}, purposes: {sorted(od.purpose.unique())}")

zid = {k: i for i, k in enumerate(zones.index)}
n = len(zones)
cx, cy, ar = zones.cx.values, zones.cy.values, zones.area.values
D = np.hypot(cx[:, None] - cx[None, :], cy[:, None] - cy[None, :])
np.fill_diagonal(D, 0.5 * np.sqrt(ar))
band_idx = np.digitize(D, BANDS) - 1  # -1 => <0 不存在; 最大带外 = len(BANDS)-1
valid = (band_idx >= 0) & (band_idx < len(BANDS) - 1)
nb = len(BANDS) - 1

def invert(purpose, col):
    sub = od[od.purpose == purpose]
    T = np.zeros((n, n))
    np.add.at(T, (sub.o.map(zid).values, sub.d.map(zid).values), sub[col].values)
    O, Dd = T.sum(1), T.sum(0)
    obs_b = np.array([T[valid & (band_idx == b)].sum() for b in range(nb)])
    f = np.ones(nb)
    Ai, Bj = np.ones(n), np.ones(n)
    for it in range(N_ITER):
        F = np.where(valid, f[np.clip(band_idx, 0, nb - 1)], 0.0)
        M = (Ai * O)[:, None] * (Bj * Dd)[None, :] * F
        # Furness 行列平衡
        rs = M.sum(1); Ai *= np.where(rs > 0, O / np.where(rs > 0, rs, 1), 1)
        M = (Ai * O)[:, None] * (Bj * Dd)[None, :] * F
        cs = M.sum(0); Bj *= np.where(cs > 0, Dd / np.where(cs > 0, cs, 1), 1)
        M = (Ai * O)[:, None] * (Bj * Dd)[None, :] * F
        mod_b = np.array([M[valid & (band_idx == b)].sum() for b in range(nb)])
        ratio = np.where(mod_b > 0, obs_b / np.where(mod_b > 0, mod_b, 1), 1)
        f *= ratio
        f /= f[0] if f[0] > 0 else 1
    mids = 0.5 * (BANDS[:-1] + BANDS[1:])
    w = obs_b  # 带内观测流量为拟合权重
    fits = {}
    def fit(name, func, p0, bounds, mask, tag):
        try:
            p, _ = curve_fit(func, mids[mask], f[mask], p0=p0, bounds=bounds,
                             sigma=1/np.sqrt(w[mask]), maxfev=20000)
            pred = func(mids[mask], *p)
            lr = np.log(f[mask]); lp = np.log(np.clip(pred, 1e-12, None))
            ss = 1 - np.sum(w[mask] * (lr - lp)**2) / np.sum(w[mask] * (lr - lr.mean())**2)
            fits[f"{name}{tag}"] = {"params": [float(x) for x in p], "logR2_weighted": round(float(ss), 4)}
        except Exception as e:
            fits[f"{name}{tag}"] = {"error": str(e)}
    ok = (f > 0) & (w > 0)
    tail = ok & (mids >= 3000)  # 内々トリップ非公表 → <3km 帯は左截断, 単調尾部のみ
    for mask, tag in [(ok, ""), (tail, "_tail3km")]:
        fit("negexp", lambda d, a, b: a * np.exp(-b * d), [1, 3e-4],
            ([1e-6, 1e-6], [1e3, 1e-2]), mask, tag)
        fit("power", lambda d, a, al: a * (d / 1000.0) ** (-al), [1, 1.5],
            ([1e-6, 0.1], [1e4, 6]), mask, tag)
        fit("gauss", lambda d, a, s: a * np.exp(-d**2 / (2 * s**2)), [1, 5000],
            ([1e-6, 100], [1e3, 1e5]), mask, tag)
        fit("lognorm", lambda d, a, mu, s: a * np.exp(-(np.log(d) - mu)**2 / (2 * s**2)),
            [1, 8, 1], ([1e-6, 4, 0.05], [1e3, 12, 4]), mask, tag)
    return {"purpose": purpose, "mode_col": col, "total_trips": float(T.sum()),
            "note": "published OD excludes most intra-zone trips (diag share<1%); bands <3km left-truncated",
            "intra_share": float(np.trace(T) / max(T.sum(), 1)),
            "band_km": [[b0/1000, b1/1000] for b0, b1 in zip(BANDS[:-1], BANDS[1:])],
            "band_mid_m": mids.tolist(), "obs_trips": obs_b.tolist(),
            "f_empirical": f.tolist(), "fits": fits}

results = {}
for purpose in ["自宅－私事", "自宅－勤務"]:
    for col in ["total", "car", "rail", "walk"]:
        key = f"{purpose}_{col}"
        results[key] = invert(purpose, col)
        best = max((v for v in results[key]["fits"].items() if "logR2_weighted" in v[1]),
                   key=lambda kv: kv[1]["logR2_weighted"])
        print(key, "intra_share:", round(results[key]["intra_share"], 3),
              "best fit:", best[0], best[1])

with open(f"{OUT}/decay_inversion.json", "w") as fo:
    json.dump(results, fo, indent=2, ensure_ascii=False)

# === Fig.2 草稿: empirical (私事 total) vs 6 手选规格 ===
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
main = results["自宅－私事_total"]
mids = np.array(main["band_mid_m"]); f = np.array(main["f_empirical"])
dgrid = np.linspace(100, 20000, 400)
specs = {
    "negexp β=0.001": np.exp(-0.001 * dgrid), "negexp β=0.003": np.exp(-0.003 * dgrid),
    "power d^-1.5": np.clip(dgrid, 125, None)**-1.5 / (125.0**-1.5),
    "gauss σ=1500": np.exp(-dgrid**2 / (2 * 1500**2)),
    "rect 5km": (dgrid <= 5000).astype(float),
}
fig, ax = plt.subplots(figsize=(9, 6), dpi=130)
for lbl, y in specs.items():
    ax.plot(dgrid / 1000, y, lw=1.2, alpha=0.7, label=lbl)
zx = np.array([0, 1000, 1000, 3000, 3000, 5000, 5000]) / 1000
zy = [1, 1, 0.68, 0.68, 0.22, 0.22, 0]
ax.plot(zx, zy, lw=1.2, alpha=0.7, label="zone 1/.68/.22", drawstyle="steps-post")
ax.plot(mids / 1000, f, "ko-", ms=5, lw=2, label="empirical all-mode (PT home-personal)")
walk = results["自宅－私事_walk"]
fw = np.array(walk["f_empirical"])
okw = fw > 0
ax.plot(mids[okw] / 1000, fw[okw], "s-", color="dimgray", ms=5, lw=2,
        label="empirical walk-only (PT home-personal)")
bw = walk["fits"].get("negexp_tail3km", {}).get("params")
if bw: ax.plot(dgrid / 1000, np.exp(-bw[1] * dgrid), "--", color="dimgray", lw=1.5,
               label=f"walk negexp tail fit β={bw[1]:.5f}/m")
ax.set_yscale("log"); ax.set_ylim(1e-4, 1.5); ax.set_xlim(0, 20)
ax.set_xlabel("distance (km)"); ax.set_ylabel("relative weight f(d), log scale")
ax.set_title("Empirical revealed decay (Tokyo PT 2018, home-personal trips) vs hand-picked impedance specs")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_draft.png"); plt.close()
print("done ->", OUT)
