# -*- coding: utf-8 -*-
"""EPT-Dock v7：已知口袋评估模式（方法的诚实使用场景）
引导式协议：用户提供口袋（= 受体上以结合位点为中心的场点子集，模拟"已知/预测口袋"）→
  凹穴内描述子种子 → 抖动 ×12 → 锐化 EM → top-K 原子精修 → 双准则选。
对比盲式（v5）逐例给出增益。
用法：python3 -I run_v7.py [--limit N] [--report PID]   （增量写入 results_v7_guided.json）
"""
import os
import sys
import json
import time
import glob
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import entrobill_proto as eb  # noqa: E402
import entrobill_v1 as v1  # noqa: E402
import entrobill_v2 as v2  # noqa: E402
import benchmark as bm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CALDIR = os.path.abspath(os.path.join(HERE, "..", "data", "calib"))
OUT = os.path.join(HERE, "results_v7_guided.json")


def rot_axis(a, th):
    a = a / np.linalg.norm(a)
    c, s = np.cos(th), np.sin(th)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) * c + s * K + (1 - c) * np.outer(a, a)


def guided_run(pid, path, rng, n_jitter=12, n_top=3, verbose=True):
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

    # ---- 引导：口袋 = 真实位点为中心的场点子集（模拟"用户提供已知口袋"）
    #      半径随肽回转半径缩放——12 Å 固定球会截断长肽，导致 Procrustes 压缩错配 ----
    d_nat = np.sqrt(((X0[:, None, :] - pk_atoms[None, :, :]) ** 2).sum(-1))
    site_atoms = np.where((d_nat < 4.5).any(0))[0]
    site_cen = pk_atoms[site_atoms].mean(0)
    rg = float(np.sqrt(((X0 - X0.mean(0)) ** 2).sum(1).mean()))
    prad = max(10.0, rg + 6.0)
    pocket_pts = [p for p in pk_pts if np.linalg.norm(p["x"] - site_cen) < prad]
    if len(pocket_pts) < 20:
        return None
    t0 = time.perf_counter()
    relaxer.set_X(X0)
    try:
        v1.seed_pose(M, relaxer, pocket_pts, eps_seed=2.0, alpha=8.0, n_rounds=3)
    except Exception:
        return None
    seedX = relaxer.X().copy()
    r_seed = rmsd()

    # ---- 抖动 + EM ----
    com = seedX.mean(0)
    pool = [{"score": None, "X": seedX.copy()}]
    _, Xp, U1 = v1._pts_arrays(relaxer)
    pool[0]["score"] = M.score(Xp, U1, seedX)
    for k in range(n_jitter):
        relaxer.set_X(seedX)
        R = rot_axis(rng.normal(size=3), np.radians(rng.uniform(6, 18)))
        relaxer.rigid_all(R, com - R @ com)
        relaxer.rigid_all(np.eye(3), rng.normal(0, 1.2, 3))
        Pn, wres, h = v1.dock_flex(M, relaxer, anneal=False, flex=False)
        pool.append({"score": h[-1], "X": relaxer.X().copy()})
    em_best = min(rmsd_of(c["X"], relaxer, ca0) for c in pool)
    t_em = time.perf_counter() - t0

    # ---- top-K 原子精修 ----
    t1 = time.perf_counter()
    refined = []
    for c in sorted([p for p in pool if p["score"] is not None], key=lambda r: r["score"])[:n_top]:
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
            Ee, _ = ff.energy_forces(*v2.build_atoms(relaxer))
            X = relaxer.X().copy()
            d = np.sqrt(((X[:, None, :] - pk_atoms[None, :, :]) ** 2).sum(-1))
            refined.append({"E": float(sum(Ee.values())), "Eparts": {k: float(v) for k, v in Ee.items()},
                            "rmsd": rmsd(), "contacts": len(contacts(X) & C0) / max(len(C0), 1),
                            "n_contacts": len(contacts(X)), "clash": int((d < 2.7).sum()),
                            "fscore": c["score"]})
        relaxer.set_X(c["X"])
    t_ref = time.perf_counter() - t1
    if not refined:
        return None
    pickE = min(refined, key=lambda r: r["E"])
    pickF = min(refined, key=lambda r: r["fscore"])
    best = min(refined, key=lambda r: r["rmsd"])
    out = {"pdb": pid, "n_rec": len(prot), "n_pep": len(pep), "n_pocket_pts": len(pocket_pts),
           "seed_rmsd": round(r_seed, 2), "em_best": round(em_best, 2),
           "pickE_rmsd": round(pickE["rmsd"], 2), "pickE_contacts": round(pickE["contacts"], 3),
           "pickE_clash": pickE["clash"], "pickE_E": round(pickE["E"], 1),
           "pickF_rmsd": round(pickF["rmsd"], 2),
           "oracle_rmsd": round(best["rmsd"], 2), "oracle_contacts": round(best["contacts"], 3),
           "t_em": round(t_em, 1), "t_ref": round(t_ref, 1)}
    if verbose:
        print(f"  {pid}: 引导种子 {out['seed_rmsd']:5.2f} → EM {out['em_best']:5.2f} → "
              f"E选 {out['pickE_rmsd']:5.2f} Å 接触 {100*pickE['contacts']:3.0f}%  "
              f"(盲式 v5 对照见报告) [{t_em+t_ref:.0f}s]")
    out["_refined"] = refined
    out["_site_cen"] = site_cen.tolist()
    return out


def rmsd_of(X, relaxer, ca0):
    idx = [k for k, (ri, an) in enumerate(relaxer.idx) if an == "CA"]
    if len(idx) < 5:
        return float("nan")
    return float(np.sqrt(((X[idx] - ca0) ** 2).sum(1).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jitter", type=int, default=12)
    args = ap.parse_args()
    pids = []
    for p in sorted(glob.glob(os.path.join(CALDIR, "*.pdb"))):
        pid = os.path.basename(p)[:-4]
        try:
            if bm.pick_complex(eb.parse_pdb(p)) is not None:
                pids.append(pid)
        except Exception:
            pass
    if args.limit:
        pids = pids[:args.limit]
    done = {}
    if os.path.exists(OUT):
        try:
            done = {r["pdb"]: r for r in json.load(open(OUT))["cases"]}
        except Exception:
            done = {}
    print(f"v7 引导式评估：{len(pids)} 个复合物（已完成 {len(done)}）")
    cases = list(done.values())
    for i, pid in enumerate(pids):
        if pid in done:
            continue
        rng = np.random.default_rng(7000 + i)
        try:
            r = guided_run(pid, os.path.join(CALDIR, pid + ".pdb"), rng, n_jitter=args.jitter)
        except Exception as e:
            print(f"  {pid}: 失败 {type(e).__name__}: {e}")
            r = None
        if r:
            store = {k: v for k, v in r.items() if not k.startswith("_")}
            cases.append(store)
            json.dump({"cases": cases}, open(OUT, "w"), ensure_ascii=False, indent=1)
        else:
            print(f"  {pid}: 无结果")

    if not cases:
        return
    print("\n" + "=" * 76)
    gE = np.array([c["pickE_rmsd"] for c in cases])
    gF = np.array([c["pickF_rmsd"] for c in cases])
    gO = np.array([c["oracle_rmsd"] for c in cases])
    gS = np.array([c["seed_rmsd"] for c in cases])
    gC = np.array([c["pickE_contacts"] for c in cases])
    n = len(cases)
    print(f"v7 引导式（已知口袋，{n} 例）：")
    for name, arr in (("种子", gS), ("E选", gE), ("F选", gF), ("oracle", gO)):
        print(f"  {name:6s}: 中位 {np.median(arr):6.2f} Å  <2.5Å {int((arr<2.5).sum())}/{n}  "
              f"<5Å {int((arr<5).sum())}/{n}  <10Å {int((arr<10).sum())}/{n}")
    print(f"  E选天然接触恢复: 中位 {100*np.median(gC):3.0f}%")
    # 与盲式 v5 对照
    try:
        b = {r["pdb"]: r for r in json.load(open(os.path.join(HERE, "results_bench.json")))["cases"]}
        pairs = [(c["pdb"], c["pickE_rmsd"], b[c["pdb"]]["pick_rmsd"]) for c in cases if c["pdb"] in b]
        if pairs:
            win = sum(1 for _, g, bl in pairs if g < bl - 0.3)
            lose = sum(1 for _, g, bl in pairs if g > bl + 0.3)
            mg = np.array([g for _, g, bl in pairs])
            mb = np.array([bl for _, g, bl in pairs])
            print(f"  vs 盲式 v5：引导更好 {win} 例 / 更差 {lose} 例 / 相当 {len(pairs)-win-lose} 例；"
                  f"中位 {np.median(mg):.2f} vs {np.median(mb):.2f} Å")
    except Exception:
        pass
    json.dump({"cases": cases,
               "summary": {"n": n, "hit25": int((gE < 2.5).sum()), "hit5": int((gE < 5).sum()),
                           "median": float(np.median(gE)), "median_contacts": float(np.median(gC))}},
              open(OUT, "w"), ensure_ascii=False, indent=1)
    print(f"结果已写入 {OUT}")


if __name__ == "__main__":
    main()
