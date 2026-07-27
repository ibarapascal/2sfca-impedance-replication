"""
Review P1-1 补测: log 协议 β=0.0006 规格 phase-B-only 计算（复用 full2 留存距离对）
输出: 新规格 vs 既有 8 规格的五分位 pairwise 翻转 → full2/beta0006_patch.json
"""
import numpy as np, pandas as pd, json, sys, os, glob
sys.stdout.reconfigure(line_buffering=True)

OUT = os.path.join(os.environ.get("WORK_ROOT","./work"),"full2")
BETA_LOG = 0.0006  # verify_review.py: walk 尾部 log 空间加权拟合 β̂≈0.00059→取 0.0006

res_all = {}
for region in ["fukuoka", "osaka", "shutoken"]:
    pairs_dir = f"{OUT}/pairs_{region}"
    meta = np.load(f"{pairs_dir}/meta.npz")
    ufac, fac_cnt = meta["ufac"], meta["fac_cnt"].astype(np.float64)
    mesh = pd.read_parquet(f"{OUT}/{region}_mesh.parquet")
    pop = mesh.pop_total.values.astype(np.float64)
    n_mesh = len(mesh)
    chunks = sorted(glob.glob(f"{pairs_dir}/c*.npz"))
    denom = np.zeros(len(ufac))
    for cf in chunks:
        z = np.load(cf)
        src, mi, d = z["src"].astype(np.int64), z["mesh"].astype(np.int64), z["d"].astype(np.float64)
        denom += np.bincount(src, weights=pop[mi] * np.exp(-BETA_LOG * d), minlength=len(ufac))
    Rj = np.where(denom > 0, 1.0 / np.where(denom > 0, denom, 1), 0.0)
    A = np.zeros(n_mesh)
    for cf in chunks:
        z = np.load(cf)
        src, mi, d = z["src"].astype(np.int64), z["mesh"].astype(np.int64), z["d"].astype(np.float64)
        A += np.bincount(mi, weights=fac_cnt[src] * Rj[src] * np.exp(-BETA_LOG * d), minlength=n_mesh)
    order = np.argsort(A, kind="stable")
    ranks = np.empty(n_mesh, dtype=np.int64); ranks[order] = np.arange(n_mesh)
    Qn = (ranks * 5 // n_mesh).astype(np.int8)
    row = {}
    for c in [c for c in mesh.columns if c.startswith("Qnet_")]:
        row[c.replace("Qnet_", "")] = round(float((mesh[c].values != Qn).mean() * 100), 2)
    res_all[region] = row
    print(region, row)

with open(f"{OUT}/beta0006_patch.json", "w") as f:
    json.dump({"beta": BETA_LOG, "flip5_vs_existing_pct": res_all}, f, indent=2, ensure_ascii=False)
print("done")
