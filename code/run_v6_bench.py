# -*- coding: utf-8 -*-
"""EPT-Dock v6 全基准：多异构种子池 + 双准则选择
v6 探针结论：异构描述子多种子把种子覆盖 15%→24%（弱）；GW/凹穴/锚点投票均未突破。
本实验：8 个异构种子全进池（不做单选）→ 每个种子 6 抖动+EM → 池合并 →
        场点分数 top-4 → 原子精修 → 双准则报告（原子能量选 / 场点分数选 / oracle）。
运行：python3 -I run_v6_bench.py   （增量写入 results_v6_bench.json）
"""
import os
import sys
import json
import time
import glob

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import entrobill_proto as eb  # noqa: E402
import entrobill_v1 as v1  # noqa: E402
import entrobill_v2 as v2  # noqa: E402
import benchmark as bm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CALDIR = os.path.abspath(os.path.join(HERE, "..", "data", "calib"))
OUT = os.path.join(HERE, "results_v6_bench.json")

VARIANTS = [
    dict(name="base",      bins=((0, 6), (6, 12)),         alpha=8.0, eps=2.0, excl_adj=True,  sub=1.0),
    dict(name="shell5-10", bins=((0, 5), (5, 10)),         alpha=8.0, eps=2.0, excl_adj=True,  sub=1.0),
    dict(name="shell8-16", bins=((0, 8), (8, 16)),         alpha=8.0, eps=2.0, excl_adj=True,  sub=1.0),
    dict(name="3shell",    bins=((0, 4), (4, 8), (8, 12)), alpha=8.0, eps=2.0, excl_adj=True,  sub=1.0),
    dict(name="strongChem", bins=((0, 6), (6, 12)),         alpha=3.0, eps=2.0, excl_adj=True,  sub=1.0),
    dict(name="sharp",     bins=((0, 6), (6, 12)),         alpha=8.0, eps=0.8, excl_adj=True,  sub=1.0),
    dict(name="noAdjExcl", bins=((0, 6), (6, 12)),         alpha=8.0, eps=2.0, excl_adj=False, sub=1.0),
    dict(name="sub70",     bins=((0, 6), (6, 12)),         alpha=8.0, eps=2.0, excl_adj=True,  sub=0.7),
]


def rot_axis(a, th):
    a = a / np.linalg.norm(a)
    c, s = np.cos(th), np.sin(th)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) * c + s * K + (1 - c) * np.outer(a, a)


def run_one(pid, path, rng):
    chains = eb.parse_pdb(path)
    pc = bm.pick_complex(chains)
    if pc is None:
        return None
    prot, pep = pc
    n_res = len(pep)
    rec = v2.receptor_atoms(prot)
    ff = v2.AtomFF(rec)
    pk_atoms, Rresof = rec[0], rec[3]
    allpts = v1.chain_field_points(prot)
    keep = eb.surface_filter(allpts, pk_atoms, np.array(Rresof))
    pk_pts = [allpts[k] for k in keep]
    if len(pk_pts) < 40:
        return None
    pep_res = v1.mutable_residues(pep)
    M = v1.MatcherV1(v1.chain_field_points(pep_res), pk_pts, n_res, pk_atoms=pk_atoms)
    relaxer = v1.TorsionRelaxer(pep_res)
    X0 = relaxer.X().copy()
    ca0, _ = relaxer.ca_coords()
    if len(ca0) < 5:
        return None

    def rmsd():
        ca, _ = relaxer.ca_coords()
        return float(np.sqrt(((ca - ca0) ** 2).sum(1).mean()))

    def contacts(X, cut=4.5):
        d = np.sqrt(((X[:, None, :] - pk_atoms[None, :, :]) ** 2).sum(-1))
        m = d < cut
        return set(zip(relaxer.res_of[np.where(m)[0]], Rresof[np.where(m)[1]]))

    C0 = contacts(X0)
    if not C0:
        return None
    t0 = time.perf_counter()

    # 1) K 个异构种子（全部保留）
    seeds = []
    for vi, v in enumerate(VARIANTS):
        relaxer.set_X(X0)
        try:
            v1.seed_pose(M, relaxer, pk_pts, eps_seed=v["eps"], alpha=v["alpha"], n_rounds=3,
                         bins=v["bins"], excl_adj=v["excl_adj"], sub=v["sub"], rng=rng)
        except Exception:
            continue
        Xs = relaxer.X().copy()
        # 去重（平移距离 > 5 Å 才算新种子）
        if all(np.linalg.norm(Xs.mean(0) - x.mean(0)) > 5.0 for x, _, _ in seeds):
            _, Xp, U1 = v1._pts_arrays(relaxer)
            sc = M.score(Xp, U1, Xs)
            seeds.append((Xs, sc, rmsd()))
    n_seed10 = sum(1 for _, _, r in seeds if r < 10)

    # 2) 每个种子 6 抖动 + EM（池合并）
    pool = []
    for Xs, sc0, r0 in seeds:
        com = Xs.mean(0)
        relaxer.set_X(Xs)
        pool.append({"score": sc0, "X": Xs.copy()})
        for k in range(6):
            relaxer.set_X(Xs)
            R = rot_axis(rng.normal(size=3), np.radians(rng.uniform(8, 25)))
            relaxer.rigid_all(R, com - R @ com)
            relaxer.rigid_all(np.eye(3), rng.normal(0, 1.5, 3))
            Pn, wres, h = v1.dock_flex(M, relaxer, anneal=False, flex=False)
            pool.append({"score": h[-1], "X": relaxer.X().copy()})
    t_em = time.perf_counter() - t0

    # 3) 场点分数 top-4 → 原子精修（各 1 次 + 1 次盆地跳跃）
    t1 = time.perf_counter()
    refined = []
    for c in sorted(pool, key=lambda r: r["score"])[:4]:
        relaxer.set_X(c["X"])
        for jit in range(2):
            if jit:
                com2 = relaxer.X().mean(0)
                R = rot_axis(rng.normal(size=3), np.radians(rng.uniform(0, 5)))
                relaxer.rigid_all(R, com2 - R @ com2)
                relaxer.rigid_all(np.eye(3), rng.normal(0, 0.8, 3))
            xr2 = relaxer.X().copy()
            try:
                sub = ff.near(relaxer.X(), 12.0)
                v2.minimize_atomic(sub, relaxer, n_steps=200, xref=xr2)
            except Exception:
                break
            E = sum(ff.energy_forces(*v2.build_atoms(relaxer))[0].values())
            X = relaxer.X().copy()
            d = np.sqrt(((X[:, None, :] - pk_atoms[None, :, :]) ** 2).sum(-1))
            refined.append({"E": float(E), "rmsd": rmsd(),
                            "contacts": len(contacts(X)) / max(len(C0), 1),
                            "clash": int((d < 2.7).sum()),
                            "fscore": c["score"]})
        relaxer.set_X(c["X"])
    t_ref = time.perf_counter() - t1
    if not refined:
        return None
    pick_E = min(refined, key=lambda r: r["E"])
    pick_F = min(refined, key=lambda r: r["fscore"])
    best = min(refined, key=lambda r: r["rmsd"])
    return {"pdb": pid, "n_rec": len(prot), "n_pep": len(pep),
            "n_seeds": len(seeds), "seed_best": round(min(r for _, _, r in seeds), 2),
            "n_seed10": n_seed10,
            "pickE_rmsd": round(pick_E["rmsd"], 2), "pickE_contacts": round(pick_E["contacts"], 3),
            "pickF_rmsd": round(pick_F["rmsd"], 2), "pickF_contacts": round(pick_F["contacts"], 3),
            "oracle_rmsd": round(best["rmsd"], 2), "oracle_contacts": round(best["contacts"], 3),
            "pool_best": round(min(rmsd_of(p["X"], relaxer, ca0) for p in pool), 2),
            "t_em": round(t_em, 1), "t_ref": round(t_ref, 1)}


def rmsd_of(X, relaxer, ca0):
    idx = [k for k, (ri, an) in enumerate(relaxer.idx) if an == "CA"]
    if len(idx) < 5:
        return float("nan")
    return float(np.sqrt(((X[idx] - ca0) ** 2).sum(1).mean()))


def main():
    pids = []
    for p in sorted(glob.glob(os.path.join(CALDIR, "*.pdb"))):
        pid = os.path.basename(p)[:-4]
        try:
            if bm.pick_complex(eb.parse_pdb(p)) is not None:
                pids.append(pid)
        except Exception:
            pass
    done = {}
    if os.path.exists(OUT):
        try:
            done = {r["pdb"]: r for r in json.load(open(OUT))["cases"]}
        except Exception:
            done = {}
    print(f"v6 全基准：{len(pids)} 个复合物（已完成 {len(done)}），K={len(VARIANTS)} 种子 × 6 抖动")
    cases = list(done.values())
    for i, pid in enumerate(pids):
        if pid in done:
            continue
        rng = np.random.default_rng(2026 + i)
        try:
            r = run_one(pid, os.path.join(CALDIR, pid + ".pdb"), rng)
        except Exception as e:
            print(f"  {pid}: 失败 {type(e).__name__}: {e}")
            r = None
        if r:
            cases.append(r)
            json.dump({"cases": cases}, open(OUT, "w"), ensure_ascii=False, indent=1)
            print(f"  {pid}: 种子{r['n_seeds']}个(最好{r['seed_best']}Å, <10Å {r['n_seed10']}) "
                  f"池最好{r['pool_best']} → E选 {r['pickE_rmsd']} / F选 {r['pickF_rmsd']} / "
                  f"oracle {r['oracle_rmsd']} Å [{r['t_em']+r['t_ref']:.0f}s]")
        else:
            print(f"  {pid}: 无结果")

    if not cases:
        return
    print("\n" + "=" * 76)
    e_r = np.array([c["pickE_rmsd"] for c in cases])
    f_r = np.array([c["pickF_rmsd"] for c in cases])
    o_r = np.array([c["oracle_rmsd"] for c in cases])
    p_r = np.array([c["pool_best"] for c in cases])
    s_r = np.array([c["seed_best"] for c in cases])
    n = len(cases)
    for name, arr in (("按原子能量选", e_r), ("按场点分数选", f_r), ("oracle(候选内最好)", o_r),
                      ("EM池最好", p_r), ("种子最好", s_r)):
        print(f"  {name:16s}: 中位 {np.median(arr):6.2f} Å  <2.5Å {int((arr<2.5).sum())}/{n}  "
              f"<5Å {int((arr<5).sum())}/{n}  <10Å {int((arr<10).sum())}/{n}")
    e_c = np.array([c["pickE_contacts"] for c in cases])
    print(f"  E选接触恢复: 中位 {100*np.median(e_c):.0f}%")
    json.dump({"cases": cases,
               "summary": {"n": n, "hit25_E": int((e_r < 2.5).sum()), "hit5_E": int((e_r < 5).sum()),
                           "hit25_F": int((f_r < 2.5).sum()), "hit5_F": int((f_r < 5).sum()),
                           "hit5_oracle": int((o_r < 5).sum()), "median_E": float(np.median(e_r))}},
              open(OUT, "w"), ensure_ascii=False, indent=1)
    print(f"\n结果已写入 {OUT}")


if __name__ == "__main__":
    main()
