# -*- coding: utf-8 -*-
"""EPT-Dock 计划1：反筛富集基准
设计：全部合格肽-蛋白复合物的受体 → detect_pockets(kmax=6) → 口袋数据库
      （+ 球蛋白诱饵的口袋）；每条肽为查询，在整个口袋库里找自己真实结合口袋。
度量：true pocket 排名、EF@1/5/10、MRR。消融：纯几何 vs 几何+类型化学项。
用法：python3 -I screen_bench.py   （结果 → results_screen_bench.json）
"""
import os
import sys
import json
import glob
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import entrobill_proto as eb  # noqa: E402
import entrobill_v1 as v1  # noqa: E402
import benchmark as bm  # noqa: E402

CALDIR = os.path.abspath(os.path.join(HERE, "..", "data", "calib"))
OUT = os.path.join(HERE, "results_screen_bench.json")
W_T = 0.5          # 类型化学项权重（Å² 当量：0~80 → 0~40）
EPS = 100.0
KMAX_POCKETS = 6
MAX_POCKET_PTS = 140


def receptor_surfaces(residues):
    xyz, resl = [], []
    for i, (seq, nm, at) in enumerate(residues):
        for an, (p, e) in at.items():
            xyz.append(p)
            resl.append(i)
    xyz = np.array(xyz)
    pts = eb.chain_field_points(residues)
    keep = eb.surface_filter(pts, xyz, np.array(resl))
    return [pts[k] for k in keep]


def typed_ot(A_pts, B_pts, w_t=W_T, eps=EPS, iters=200):
    """类型感知 Sinkhorn 距离：C = d² + w_t·type_cost；返回去偏 Sinkhorn 散度"""
    A = np.array([p["x"] for p in A_pts])
    B = np.array([p["x"] for p in B_pts])
    ta = [p["t"] for p in A_pts]
    tb = [p["t"] for p in B_pts]

    def core(Ca, Cb, extra):
        K = np.exp(-Cb / eps) if extra is None else np.exp(-(Cb + extra) / eps)
        n, m = Ca.shape
        a = np.full(n, 1.0 / n)
        b = np.full(m, 1.0 / m)
        v = np.ones(m)
        for _ in range(iters):
            u = a / np.maximum(K @ v, 1e-300)
            v = b / np.maximum(K.T @ u, 1e-300)
        P = u[:, None] * K * v[None, :]
        return float((P * Cb).sum() if extra is None else (P * (Cb + extra)).sum())

    C_ab = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    C_aa = ((A[:, None, :] - A[None, :, :]) ** 2).sum(-1)
    C_bb = ((B[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    T = np.array([[eb.type_cost(x, y) for y in tb] for x in ta]) * w_t
    d_ab = core(C_ab, C_ab, T)
    d_aa = core(C_aa, C_aa, None)
    T_bb = np.array([[eb.type_cost(x, y) for y in tb] for x in tb]) * w_t
    d_bb = core(C_bb, C_bb, T_bb)
    return d_ab - 0.5 * d_aa - 0.5 * d_bb


def geo_ot(A_pts, B_pts, eps=EPS, iters=200):
    A = np.array([p["x"] for p in A_pts])
    B = np.array([p["x"] for p in B_pts])
    return v1.sinkhorn_div(A, B, eps=eps, iters=iters)


def main():
    t_all = time.perf_counter()
    # ---- 收集复合物 ----
    complexes = []
    decoy_proteins = []
    for p in sorted(glob.glob(os.path.join(CALDIR, "*.pdb"))):
        pid = os.path.basename(p)[:-4]
        try:
            ch = eb.parse_pdb(p)
            pc = bm.pick_complex(ch)
            if pc is not None:
                complexes.append((pid, pc))
            else:
                best = max(ch.values(), key=lambda r: len(r)) if ch else None
                if best is not None and len(best) >= 40:
                    decoy_proteins.append((pid, best))
        except Exception:
            pass
    print(f"复合物 {len(complexes)} 个（查询肽） | 诱饵蛋白 {len(decoy_proteins)} 个")

    # ---- 建口袋数据库 ----
    db = []          # (name, pocket_pts, owner)
    queries = []     # (pid, pep_residues, pep_pts, true_key)
    t0 = time.perf_counter()
    for pid, (prot, pep) in complexes:
        try:
            pk_pts = receptor_surfaces(prot)
            if len(pk_pts) < 40:
                continue
            pk_atoms = np.array([p for (seq, nm, at) in prot for an, (p, e) in at.items()])
            pk = v1.detect_pockets(pk_pts, pk_atoms, kmax=KMAX_POCKETS, rad=14.0)
            X0 = np.array([x for (seq, nm, at) in pep for an, (x, e) in at.items()])
            d_nat = np.sqrt(((X0[:, None, :] - pk_atoms[None, :, :]) ** 2).sum(-1))
            site = pk_atoms[np.where((d_nat < 4.5).any(0))[0]].mean(0)
            true_key, true_d = None, 1e9
            for ci, pts in pk:
                if len(pts) < 12:
                    continue
                key = f"{pid}#p{ci[0]}-{len(pk)}"
                d = float(np.linalg.norm(np.mean([p["x"] for p in pts], 0) - site))
                if len(pts) > MAX_POCKET_PTS:
                    idx = np.random.default_rng(1).choice(len(pts), MAX_POCKET_PTS, replace=False)
                    pts = [pts[i] for i in idx]
                db.append((key, pts, pid))
                if d < true_d:
                    true_d, true_key = d, key
            if true_key is None or len(pep) < 5:
                continue
            pep_pts = eb.chain_field_points(pep)
            if len(pep_pts) < 8:
                continue
            queries.append((pid, f"{''.join(eb.__dict__.get('THREE2ONE', {}).get(r[1], '?') for r in pep)}", pep_pts, true_key))
        except Exception:
            continue
    for pid, prot in decoy_proteins:
        try:
            pk_pts = receptor_surfaces(prot)
            pk_atoms = np.array([p for (seq, nm, at) in prot for an, (p, e) in at.items()])
            for ci, pts in v1.detect_pockets(pk_pts, pk_atoms, kmax=KMAX_POCKETS, rad=14.0):
                if len(pts) < 12:
                    continue
                if len(pts) > MAX_POCKET_PTS:
                    idx = np.random.default_rng(1).choice(len(pts), MAX_POCKET_PTS, replace=False)
                    pts = [pts[i] for i in idx]
                db.append((f"{pid}#decoy", pts, pid))
        except Exception:
            pass
    print(f"口袋库 {len(db)} 个（其中诱饵蛋白口袋 {sum(1 for k,_,o in db if '#decoy' in k)}）"
          f" | 查询 {len(queries)} 条 | 建 库 {time.perf_counter()-t0:.0f}s")

    # ---- 筛选矩阵（变体扫描）----
    res = {}
    variants = [
        ("geo_e100", lambda q, p: geo_ot(q, p, eps=100.0)),
        ("geo_e50", lambda q, p: geo_ot(q, p, eps=50.0)),
        ("chem_e100_wt1", lambda q, p: typed_ot(q, p, w_t=1.0, eps=100.0)),
        ("chem_e50_wt1", lambda q, p: typed_ot(q, p, w_t=1.0, eps=50.0)),
    ]
    for variant, fn in variants:
        t0 = time.perf_counter()
        ranks = []
        rows = []
        for qi, (pid, seq, qpts, true_key) in enumerate(queries):
            scores = [(fn(qpts, pts), key, owner) for key, pts, owner in db]
            scores.sort()
            rank = [i for i, (s, k, o) in enumerate(scores) if k == true_key][0] + 1
            ranks.append(rank)
            rows.append({"query": pid, "seq": seq, "rank": rank,
                         "db": len(db), "top1": scores[0][1]})
        ranks = np.array(ranks)
        n = len(ranks)
        stats = {"n_queries": n, "db": len(db),
                 "EF1": float((ranks == 1).mean()), "EF5": float((ranks <= 5).mean()),
                 "EF10": float((ranks <= 10).mean()),
                 "MRR": float((1.0 / ranks).mean()),
                 "median_rank": float(np.median(ranks)),
                 "mean_pct": float((ranks / len(db)).mean())}
        res[variant] = {"stats": stats, "rows": rows}
        print(f"[{variant}] EF@1={stats['EF1']:.2f} EF@5={stats['EF5']:.2f} EF@10={stats['EF10']:.2f} "
              f"MRR={stats['MRR']:.3f} 中位排名={stats['median_rank']:.0f}/{len(db)} "
              f"({time.perf_counter()-t0:.0f}s)")

    json.dump({"meta": {"w_t": W_T, "eps": EPS, "kmax": KMAX_POCKETS},
               "results": res}, open(OUT, "w"), ensure_ascii=False, indent=1)
    print(f"\n总耗时 {time.perf_counter()-t_all:.0f}s → {OUT}")


if __name__ == "__main__":
    main()
