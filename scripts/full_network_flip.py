"""
xx-impedance-standard 正式计算: 真实路网 2SFCA × 6 阻抗规格 × 3 都市圏翻转率

设计（内存/时间约束: pc-a 32GB 其中 OrbStack 占 ~20GB, 严禁 OOM）:
  Phase A (per region): OSM walk graph bbox 裁子图 → 设施 snap 到 unique node →
    分块 Dijkstra(limit=5km) → mesh 列即抽即弃 → (fac_node, mesh, dist) 对存盘 npz
  Phase B: 存盘距离对两遍复用 → 6 规格 Rj → A → 翻转指标
    （Dijkstra 只跑一次, 6 规格零额外图计算）
  Euclidean 对照: 同规格 KDTree 版（pilot 逻辑）, 全三区域
  指标: 3/5/10 分位翻转率(mesh/人口加权) + pairwise flip/Spearman +
    bottom-20% 荒漠稳定性 + euclid vs network 距离度量翻转

内存: Dijkstra chunk 32×子图节点×8B（kanto 子图 ~10M → ~2.5GB 瞬时, 即释放）;
     其余累加器 <500MB。CHUNK 可按 benchmark 调。
用法:
  python3 full_network_flip.py benchmark   # 只跑 shutoken 3 chunk 估时
  python3 full_network_flip.py shutoken|osaka|fukuoka|all
"""
import numpy as np, pandas as pd, json, time, sys, os, gc, shutil

sys.stdout.reconfigure(line_buffering=True)
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.stats import spearmanr

STORAGE = "${DATA_ROOT}/locanex/storage"
OUT = os.path.join(os.environ.get("WORK_ROOT","./work"),"full")
os.makedirs(OUT, exist_ok=True)

D_MAX = 5000.0
M_LAT = 111320.0
CHUNK = 32
BUF = 0.06  # bbox 裁图缓冲 ~6km（>5km catchment, 防边界截断）

REGIONS = {
    "shutoken": {"graph": "kanto",  "bbox_lat": (34.9, 36.3), "bbox_lon": (138.9, 140.9), "ref_lat": 35.7},
    "osaka":    {"graph": "kansai", "bbox_lat": (34.2, 35.0), "bbox_lon": (135.0, 135.8), "ref_lat": 34.7},
    "fukuoka":  {"graph": "kyushu", "bbox_lat": (33.0, 33.8), "bbox_lon": (130.0, 131.0), "ref_lat": 33.6},
}
E2_FILE = {"shutoken": "kanto", "osaka": "kansai", "fukuoka": "kyushu"}

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

SPECS = [("negexp_b001", w_negexp_slow), ("negexp_b003", w_negexp_fast),
         ("power_15", w_power), ("gauss_s1500", w_gauss),
         ("rect", w_rect), ("zone_e2sfca", w_zone)]
NAMES = [n for n, _ in SPECS]
NS = len(SPECS)

t0 = time.time()
def log(msg): print(f"[{time.time()-t0:8.1f}s] {msg}")

# === 数据加载 ===

def load_region_data(region):
    cfg = REGIONS[region]
    bla, blo = cfg["bbox_lat"], cfg["bbox_lon"]
    e2 = pd.read_parquet(f"{STORAGE}/structured/analysis/dim1_facility/e2sfca_{E2_FILE[region]}.parquet",
                         columns=["mesh_code", "lon", "lat", "pop_total"])
    mesh = e2[(e2.lat >= bla[0]) & (e2.lat <= bla[1]) &
              (e2.lon >= blo[0]) & (e2.lon <= blo[1])].reset_index(drop=True)
    fac = pd.read_parquet(f"{STORAGE}/structured/analysis/dim1_facility/facilities_all.parquet",
                          columns=["category", "lat", "lon"])
    fac = fac[(fac.category == "hospital") &
              (fac.lat >= bla[0]) & (fac.lat <= bla[1]) &
              (fac.lon >= blo[0]) & (fac.lon <= blo[1])].reset_index(drop=True)
    log(f"{region}: mesh={len(mesh)} pop={mesh.pop_total.sum():.0f} fac={len(fac)}")
    return mesh, fac

def load_subgraph(region):
    cfg = REGIONS[region]
    g = cfg["graph"]
    bla = (cfg["bbox_lat"][0] - BUF, cfg["bbox_lat"][1] + BUF)
    blo = (cfg["bbox_lon"][0] - BUF, cfg["bbox_lon"][1] + BUF)
    nodes = pd.read_parquet(f"{STORAGE}/structured/osm/walk_nodes_{g}.parquet")
    keep = ((nodes.lat >= bla[0]) & (nodes.lat <= bla[1]) &
            (nodes.lon >= blo[0]) & (nodes.lon <= blo[1])).values
    nodes = nodes[keep].reset_index(drop=True)
    n = len(nodes)
    log(f"  subgraph nodes: {n} (graph {g})")
    id2idx = pd.Series(np.arange(n), index=nodes.node_id.values)
    edges = pd.read_parquet(f"{STORAGE}/structured/osm/walk_edges_{g}.parquet")
    src = id2idx.reindex(edges.from_node.values).values
    dst = id2idx.reindex(edges.to_node.values).values
    ok = ~(np.isnan(src) | np.isnan(dst))
    src = src[ok].astype(np.int64); dst = dst[ok].astype(np.int64)
    wgt = edges.length_m.values[ok].astype(np.float64)
    del edges; gc.collect()
    graph = csr_matrix((np.concatenate([wgt, wgt]),
                        (np.concatenate([src, dst]), np.concatenate([dst, src]))),
                       shape=(n, n))
    tree = cKDTree(np.column_stack([nodes.lon.values, nodes.lat.values]))
    log(f"  subgraph edges: {ok.sum()}")
    return graph, tree, n

# === Phase A: Dijkstra 距离对存盘 ===

def phase_a(region, mesh, fac, benchmark=False):
    pairs_dir = f"{OUT}/pairs_{region}"
    os.makedirs(pairs_dir, exist_ok=True)
    graph, tree, n_nodes = load_subgraph(region)
    snap_d_fac, fac_node = tree.query(np.column_stack([fac.lon.values, fac.lat.values]))
    snap_d_mesh, mesh_node = tree.query(np.column_stack([mesh.lon.values, mesh.lat.values]))
    log(f"  snap dist p50/p95 (deg): fac {np.percentile(snap_d_fac,50):.5f}/{np.percentile(snap_d_fac,95):.5f} "
        f"mesh {np.percentile(snap_d_mesh,50):.5f}/{np.percentile(snap_d_mesh,95):.5f}")
    ufac, fac_cnt = np.unique(fac_node, return_counts=True)
    n_ufac = len(ufac)
    log(f"  unique fac nodes: {n_ufac} (from {len(fac)})")
    np.savez(f"{pairs_dir}/meta.npz", ufac=ufac, fac_cnt=fac_cnt, mesh_node=mesh_node)

    n_chunks = (n_ufac + CHUNK - 1) // CHUNK
    lim = 3 if benchmark else n_chunks
    tA = time.time()
    total_pairs = 0
    for ci in range(min(n_chunks, lim)):
        c0 = ci * CHUNK
        idx = ufac[c0:c0+CHUNK]
        dm = dijkstra(graph, directed=False, indices=idx, limit=D_MAX)
        dmesh = dm[:, mesh_node].astype(np.float32)  # (chunk, n_mesh)
        del dm; gc.collect()
        ii, jj = np.where(dmesh <= D_MAX)
        np.savez(f"{pairs_dir}/c{ci:05d}.npz",
                 src=(ii + c0).astype(np.uint32), mesh=jj.astype(np.uint32),
                 d=dmesh[ii, jj])
        total_pairs += len(ii)
        del dmesh
        if ci % 25 == 0 or benchmark:
            el = time.time() - tA
            per = el / (ci + 1)
            log(f"  dijkstra chunk {ci+1}/{n_chunks} pairs={total_pairs/1e6:.1f}M "
                f"({per:.1f}s/chunk, ETA {(n_chunks-ci-1)*per/3600:.1f}h)")
    del graph, tree; gc.collect()
    if benchmark:
        per = (time.time() - tA) / lim
        log(f"BENCHMARK: {per:.1f}s/chunk × {n_chunks} chunks = {n_chunks*per/3600:.2f}h for {region}")
        return None
    log(f"  phase A done: {total_pairs/1e6:.1f}M pairs")
    return pairs_dir, n_chunks

# === Phase B: 距离对 → 6 规格 A ===

def phase_b(region, mesh, pairs_dir, n_chunks):
    meta = np.load(f"{pairs_dir}/meta.npz")
    ufac, fac_cnt = meta["ufac"], meta["fac_cnt"].astype(np.float64)
    pop = mesh.pop_total.values.astype(np.float64)
    n_mesh = len(mesh)
    denom = np.zeros((NS, len(ufac)))
    for ci in range(n_chunks):
        z = np.load(f"{pairs_dir}/c{ci:05d}.npz")
        src, mi, d = z["src"].astype(np.int64), z["mesh"].astype(np.int64), z["d"].astype(np.float64)
        p = pop[mi]
        for s, (_, wf) in enumerate(SPECS):
            denom[s] += np.bincount(src, weights=p * wf(d), minlength=len(ufac))
    Rj = np.where(denom > 0, 1.0 / np.where(denom > 0, denom, 1), 0.0)
    A = np.zeros((NS, n_mesh))
    for ci in range(n_chunks):
        z = np.load(f"{pairs_dir}/c{ci:05d}.npz")
        src, mi, d = z["src"].astype(np.int64), z["mesh"].astype(np.int64), z["d"].astype(np.float64)
        for s, (_, wf) in enumerate(SPECS):
            A[s] += np.bincount(mi, weights=fac_cnt[src] * Rj[s, src] * wf(d), minlength=n_mesh)
    log(f"  phase B done")
    return A

# === Euclidean 对照（pilot 逻辑） ===

def euclid_A(region, mesh, fac):
    ref_lat = REGIONS[region]["ref_lat"]
    m_lon = M_LAT * np.cos(np.radians(ref_lat))
    mesh_xy = np.column_stack([mesh.lon.values * m_lon, mesh.lat.values * M_LAT])
    fac_xy = np.column_stack([fac.lon.values * m_lon, fac.lat.values * M_LAT])
    pop = mesh.pop_total.values.astype(np.float64)
    mtree = cKDTree(mesh_xy)
    n_fac, n_mesh = len(fac_xy), len(mesh_xy)
    Rj = np.zeros((NS, n_fac))
    for c0 in range(0, n_fac, 500):
        for k, idxs in enumerate(mtree.query_ball_point(fac_xy[c0:c0+500], D_MAX)):
            if not idxs: continue
            idxs = np.asarray(idxs)
            d = np.linalg.norm(mesh_xy[idxs] - fac_xy[c0+k], axis=1)
            for s, (_, wf) in enumerate(SPECS):
                dn = np.sum(pop[idxs] * wf(d))
                if dn > 0: Rj[s, c0+k] = 1.0 / dn
    A = np.zeros((NS, n_mesh))
    for c0 in range(0, n_fac, 500):
        for k, idxs in enumerate(mtree.query_ball_point(fac_xy[c0:c0+500], D_MAX)):
            if not idxs: continue
            idxs = np.asarray(idxs)
            d = np.linalg.norm(mesh_xy[idxs] - fac_xy[c0+k], axis=1)
            for s, (_, wf) in enumerate(SPECS):
                A[s, idxs] += Rj[s, c0+k] * wf(d)
    log(f"  euclid done")
    return A

# === 指标 ===

def classify(A, n_class):
    n = A.shape[1]
    Q = np.zeros((NS, n), dtype=np.int8)
    for s in range(NS):
        order = np.argsort(A[s], kind="stable")
        ranks = np.empty(n, dtype=np.int64); ranks[order] = np.arange(n)
        Q[s] = (ranks * n_class // n).astype(np.int8)
    return Q

def flip_metrics(A, pop):
    res = {}
    for nc in (3, 5, 10):
        Q = classify(A, nc)
        qmin, qmax = Q.min(axis=0), Q.max(axis=0)
        anyf, r2 = qmax != qmin, (qmax - qmin) >= 2
        res[f"c{nc}_flip_any_pct"] = float(anyf.mean() * 100)
        res[f"c{nc}_flip_any_pop_pct"] = float(pop[anyf].sum() / pop.sum() * 100)
        res[f"c{nc}_flip_ge2_pct"] = float(r2.mean() * 100)
        res[f"c{nc}_flip_ge2_pop_pct"] = float(pop[r2].sum() / pop.sum() * 100)
        if nc == 5:
            Q5, qr5 = Q, qmax - qmin
            bottom = Q == 0
            ba, bl = bottom.any(axis=0), bottom.all(axis=0)
            res["bottom20_union_mesh"] = int(ba.sum())
            res["bottom20_unstable_pct_of_union"] = float((ba & ~bl).sum() / max(ba.sum(), 1) * 100)
            res["bottom20_unstable_pop"] = float(pop[ba & ~bl].sum())
    res["pairwise_flip5_pct"] = {f"{NAMES[a]}|{NAMES[b]}": round(float((Q5[a] != Q5[b]).mean() * 100), 2)
                                 for a in range(NS) for b in range(a+1, NS)}
    res["pairwise_spearman"] = {f"{NAMES[a]}|{NAMES[b]}": round(float(spearmanr(A[a], A[b]).statistic), 4)
                                for a in range(NS) for b in range(a+1, NS)}
    res["zero_access_counts"] = {NAMES[s]: int((A[s] == 0).sum()) for s in range(NS)}
    return res, Q5, qr5

def run_region(region, benchmark=False):
    log(f"=== {region} ===")
    mesh, fac = load_region_data(region)
    if benchmark:
        phase_a(region, mesh, fac, benchmark=True)
        return
    pop = mesh.pop_total.values.astype(np.float64)
    pairs_dir, n_chunks = phase_a(region, mesh, fac)
    A_net = phase_b(region, mesh, pairs_dir, n_chunks)
    A_eu = euclid_A(region, mesh, fac)
    res = {"region": region, "n_mesh": len(mesh), "n_fac": len(fac),
           "total_pop": float(pop.sum()), "d_max_m": D_MAX, "specs": NAMES}
    net, Q5n, qr5n = flip_metrics(A_net, pop)
    eu, Q5e, _ = flip_metrics(A_eu, pop)
    res["network"], res["euclid"] = net, eu
    # 距离度量本身的翻转（同规格 euclid vs network 五分位）
    res["euclid_vs_network_flip5_pct"] = {
        NAMES[s]: round(float((Q5n[s] != Q5e[s]).mean() * 100), 2) for s in range(NS)}
    with open(f"{OUT}/{region}_results.json", "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    df = mesh[["mesh_code", "lon", "lat", "pop_total"]].copy()
    for s, n in enumerate(NAMES):
        df[f"Anet_{n}"] = A_net[s]; df[f"Qnet_{n}"] = Q5n[s]
        df[f"Aeu_{n}"] = A_eu[s]
    df["q_range_net"] = qr5n
    df.to_parquet(f"{OUT}/{region}_mesh.parquet", index=False)
    # flip map
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 8), dpi=130)
    sc = ax.scatter(df.lon, df.lat, c=df.q_range_net, s=0.3, cmap="YlOrRd", vmin=0, vmax=4, linewidths=0)
    plt.colorbar(sc, label="quintile class range across 6 impedance specs (network)")
    ax.set_title(f"{region} network flip map | any {net['c5_flip_any_pct']:.1f}% ge2 {net['c5_flip_ge2_pct']:.1f}%")
    ax.set_aspect(1 / np.cos(np.radians(REGIONS[region]["ref_lat"])))
    plt.tight_layout(); plt.savefig(f"{OUT}/{region}_flip_map.png"); plt.close()
    shutil.rmtree(pairs_dir)  # 距离对用完即删（磁盘控制）
    log(f"=== {region} done ===")
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}, ensure_ascii=False))

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "benchmark"
    if arg == "benchmark":
        run_region("shutoken", benchmark=True)
    elif arg == "all":
        for r in ["fukuoka", "osaka", "shutoken"]:  # 小→大, 早出结果早发现问题
            run_region(r)
        log("ALL DONE")
    else:
        run_region(arg)
