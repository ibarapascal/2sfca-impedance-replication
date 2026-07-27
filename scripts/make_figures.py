"""
Paper 14 review 用图集生成（v1 阶段: 供用户 review, 非投稿最终版）
fig1: 三都市圏 network flip map 拼版 | fig2: 反演曲线+截断阴影+协议带
fig3: closure 排行(两协议×三区域) | fig4: pairwise flip 热图(Tokyo)
"""
import numpy as np, pandas as pd, json, os, sys
sys.stdout.reconfigure(line_buffering=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.environ.get("WORK_ROOT", "./work")
OUT = f"{BASE}/figures"
os.makedirs(OUT, exist_ok=True)

REGIONS = [("shutoken", "Tokyo MA", 35.7), ("osaka", "Osaka MA", 34.7), ("fukuoka", "Fukuoka MA", 33.6)]
NAMES = ["negexp_b001", "negexp_b003", "power_15", "gauss_s1500", "rect", "zone_e2sfca"]
LBL = {"negexp_b001": "exp β=0.001", "negexp_b003": "exp β=0.003", "power_15": "power d⁻¹·⁵",
       "gauss_s1500": "Gaussian σ=1.5km", "rect": "rectangular", "zone_e2sfca": "zonal 1/.68/.22"}

# === Fig.1: 三区域 flip map ===
fig, axes = plt.subplots(1, 3, figsize=(16.5, 6), dpi=150,
                         gridspec_kw={"width_ratios": [1.35, 1, 1]})
res = {}
for ax, (reg, title, ref_lat) in zip(axes, REGIONS):
    df = pd.read_parquet(f"{BASE}/full2/{reg}_mesh.parquet",
                         columns=["lon", "lat", "q_range_net"])
    r = json.load(open(f"{BASE}/full2/{reg}_results.json")); res[reg] = r
    sc = ax.scatter(df.lon, df.lat, c=df.q_range_net, s=0.25, cmap="YlOrRd",
                    vmin=0, vmax=4, linewidths=0, rasterized=True)
    ax.set_aspect(1 / np.cos(np.radians(ref_lat)))
    n = r["network"]
    ax.set_title(f"{title}\nany-flip {n['c5_flip_any_pct']:.1f}% | ≥2 classes {n['c5_flip_ge2_pct']:.1f}%",
                 fontsize=11)
    ax.tick_params(labelsize=7)
    # 简易比例尺 (20km)
    lat0 = df.lat.min() + 0.03
    lon0 = df.lon.min() + 0.05
    dlon = 20000 / (111320 * np.cos(np.radians(ref_lat)))
    ax.plot([lon0, lon0 + dlon], [lat0, lat0], "k-", lw=2)
    ax.text(lon0 + dlon / 2, lat0 + 0.015, "20 km", ha="center", fontsize=8)
cb = fig.colorbar(sc, ax=axes, shrink=0.85, ticks=[0, 1, 2, 3, 4], pad=0.012)
cb.set_label("quintile class range across 6 impedance specifications", fontsize=10)
fig.suptitle("Fig. 1  Impedance-choice classification instability, network 2SFCA to medical facilities (125 m cells)",
             fontsize=13, y=0.99)
plt.savefig(f"{OUT}/fig1_flip_maps.png", bbox_inches="tight"); plt.close()
print("fig1 done")

# === Fig.2: 反演曲线 ===
inv = json.load(open(f"{BASE}/inversion/decay_inversion.json"))
main = inv["自宅－私事_total"]; walk = inv["自宅－私事_walk"]
mids = np.array(main["band_mid_m"])
dgrid = np.linspace(100, 20000, 400)
fig, ax = plt.subplots(figsize=(9.5, 6.5), dpi=150)
ax.axvspan(0, 3, color="0.88", zorder=0)
ax.text(1.5, 2.3e-4, "near field unidentified\n(intrazonal flows\nnot published)", ha="center", fontsize=8, color="0.35")
specs = {"exp β=0.001": (np.exp(-0.001 * dgrid), "tab:blue"),
         "exp β=0.003": (np.exp(-0.003 * dgrid), "tab:orange"),
         "power d⁻¹·⁵": (np.clip(dgrid, 125, None) ** -1.5 / 125.0 ** -1.5, "tab:green"),
         "Gaussian σ=1.5km": (np.exp(-dgrid ** 2 / (2 * 1500 ** 2)), "tab:red"),
         "rectangular 5km": ((dgrid <= 5000).astype(float), "tab:purple")}
for lbl, (y, c) in specs.items():
    ax.plot(dgrid / 1000, y, lw=1.2, alpha=0.75, color=c, label=lbl)
zx = np.array([0, 1000, 1000, 3000, 3000, 5000, 5000]) / 1000
ax.plot(zx, [1, 1, 0.68, 0.68, 0.22, 0.22, 0], lw=1.2, alpha=0.75, color="tab:brown",
        drawstyle="steps-post", label="zonal 1/.68/.22")
f_all = np.array(main["f_empirical"])
ax.plot(mids / 1000, f_all, "ko-", ms=5, lw=2, label="observed all-mode (home-based personal)")
fw = np.array(walk["f_empirical"]); okw = fw > 0
ax.plot(mids[okw] / 1000, fw[okw], "s-", color="dimgray", ms=5, lw=2, label="observed walk-only")
ax.fill_between(dgrid / 1000, np.exp(-0.0006 * dgrid), np.exp(-0.0010 * dgrid),
                color="steelblue", alpha=0.22, label="walk protocol band β∈[0.0006, 0.0010]/m")
ax.set_yscale("log"); ax.set_ylim(1e-4, 2.2); ax.set_xlim(0, 20)
ax.set_xlabel("network-equivalent distance (km)"); ax.set_ylabel("relative weight f(d), log scale")
ax.set_title("Fig. 2  Revealed distance decay (Tokyo PT survey 2018) vs hand-picked impedance specifications")
ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_decay.png"); plt.close()
print("fig2 done")

# === Fig.3: closure 排行 ===
patch = json.load(open(f"{BASE}/full2/beta0006_patch.json"))["flip5_vs_existing_pct"]
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
order = ["gauss_s1500", "zone_e2sfca", "negexp_b001", "power_15", "negexp_b003", "rect"]
x = np.arange(len(order)); w = 0.36
hi = {n: [res[r]["network"]["pairwise_flip5_pct"].get(f"{n}|emp_negexp_walk",
          res[r]["network"]["pairwise_flip5_pct"].get(f"emp_negexp_walk|{n}")) for r, _, _ in REGIONS] for n in order}
lo = {n: [patch[r][n] for r, _, _ in REGIONS] for n in order}
for dx, data, lbl, c in [(-w/2, hi, "vs β=0.0010 reference (linear-loss fit)", "steelblue"),
                          (w/2, lo, "vs β=0.0006 reference (log-loss fit)", "indianred")]:
    means = [np.mean(data[n]) for n in order]
    mins = [np.min(data[n]) for n in order]; maxs = [np.max(data[n]) for n in order]
    ax.bar(x + dx, means, w, color=c, alpha=0.85, label=lbl,
           yerr=[np.array(means) - mins, np.array(maxs) - np.array(means)], capsize=3)
band_w = 26.0
ax.axhline(band_w, color="0.3", ls="--", lw=1)
ax.text(len(order) - 0.45, band_w + 1, "band internal width (25–26%)", fontsize=8, ha="right", color="0.3")
ax.set_xticks(x); ax.set_xticklabels([LBL[n] for n in order], fontsize=9)
ax.set_ylabel("quintile flip vs empirical reference (%)")
ax.set_title("Fig. 3  Distance of conventional specifications from the empirical walk-decay band\n(bars = 3-region mean, whiskers = range)")
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(f"{OUT}/fig3_closure.png"); plt.close()
print("fig3 done")

# === Fig.4: pairwise 热图 (Tokyo network, 6 手选) ===
pf = res["shutoken"]["network"]["pairwise_flip5_pct"]
sp = res["shutoken"]["network"]["pairwise_spearman"]
M = np.zeros((6, 6)); S = np.ones((6, 6))
for i, a in enumerate(NAMES):
    for j, b in enumerate(NAMES):
        if i < j:
            M[i, j] = M[j, i] = pf[f"{a}|{b}"]; S[i, j] = S[j, i] = sp[f"{a}|{b}"]
fig, ax = plt.subplots(figsize=(7.6, 6.4), dpi=150)
im = ax.imshow(M, cmap="YlOrRd", vmin=0, vmax=70)
for i in range(6):
    for j in range(6):
        if i != j:
            ax.text(j, i, f"{M[i,j]:.0f}%\nρ{S[i,j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] > 45 else "black")
        else:
            ax.text(j, i, "—", ha="center", va="center", color="0.6")
ax.set_xticks(range(6)); ax.set_xticklabels([LBL[n] for n in NAMES], rotation=30, ha="right", fontsize=8)
ax.set_yticks(range(6)); ax.set_yticklabels([LBL[n] for n in NAMES], fontsize=8)
plt.colorbar(im, label="quintile flip share (%)")
ax.set_title("Fig. 4  Pairwise quintile disagreement and Spearman ρ\n(Tokyo MA, network 2SFCA)")
plt.tight_layout(); plt.savefig(f"{OUT}/fig4_pairwise.png"); plt.close()
print("fig4 done")
