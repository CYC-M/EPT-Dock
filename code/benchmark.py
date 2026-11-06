# -*- coding: utf-8 -*-
"""EPT-Dock v5：基准集批量评估
协议：盲式对接（不给口袋信息）——肽用晶体构象，随机位姿；
      描述子种子 → 抖动 ×N → 锐化 EM → top-K 原子精修 → 按原子能量选。
指标：受体坐标系 Cα-RMSD（不做叠加）、天然接触恢复率（<4.5 Å 残基对）、碰撞数、各阶段耗时。
用法：python3 -I benchmark.py [--limit N] [--jitter 16]
结果增量写入 results_bench.json（可中断续跑）。
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

HERE = os.path.dirname(os.path.abspath(__file__))
CALDIR = os.path.abspath(os.path.join(HERE, "..", "data", "calib"))
OUT = os.path.join(HERE, "results_bench.json")

STD_AA = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
          "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"}


def rot_axis(a, th):
    a = a / np.linalg.norm(a)
    c, s = np.cos(th), np.sin(th)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) * c + s * K + (1 - c) * np.outer(a, a)


def pick_complex(chains):
    cands = {}
    for c, rs in chains.items():
        if not rs:
            continue
        n_std = sum(1 for (_, nm, _) in rs if nm in STD_AA)
        if n_std / len(rs) >= 0.9 and len(rs) >= 5:
            cands[c] = rs
    if len(cands) < 2:
        return None
    rec_c = max(cands, key=lambda c: len(cands[c]))
    if len(cands[rec_c]) < 60:
        return None
    pep_c = [c for c in cands if c != rec_c and 5 <= len(cands[c]) <= 35]
    if not pep_c:
        return None
    return cands[rec_c], cands[pep_c[0]]


def run_one(pid, path, n_jitter=16, n_top=3, verbose=True):
    chains = eb.parse_pdb(path)
    pc = pick_complex(chains)
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

    t = {}
    rng = np.random.default_rng(abs(hash(pid)) % 10000)

    # 1) 确定性种子（盲）
    t0 = time.perf_counter()
    relaxer.set_X(X0)
    try:
        v1.seed_pose(M, relaxer, pk_pts, eps_seed=2.0, alpha=8.0, n_rounds=3)
    except Exception:
        return None
    seedX = relaxer.X().copy()
    r_seed = rmsd()
    t["seed"] = time.perf_counter() - t0

    # 2) 抖动 + 锐化 EM
    t0 = time.perf_counter()
    com = seedX.mean(0)
    pool = []
    for k in range(n_jitter):
        relaxer.set_X(seedX)
        R = rot_axis(rng.normal(size=3), np.radians(rng.uniform(8, 25)))
        relaxer.rigid_all(R, com - R @ com)
        relaxer.rigid_all(np.eye(3), rng.normal(0, 1.5, 3))
        Pn, wres, h = v1.dock_flex(M, relaxer, anneal=False, flex=False)
        pool.append({"score": h[-1], "X": relaxer.X().copy()})
    t["em"] = time.perf_counter() - t0
    em_best = min(rmsd_of(c["X"], relaxer, ca0) for c in pool)

    # 3) top-K 原子精修
    t0 = time.perf_counter()
    refined = []
    for c in sorted(pool, key=lambda r: r["score"])[:n_top]:
        relaxer.set_X(c["X"])
        xr = relaxer.X().copy()
        Ee = None
        for jit in range(2):
            if jit:
                com2 = relaxer.X().mean(0)
                R = rot_axis(rng.normal(size=3), np.radians(rng.uniform(0, 5)))
                relaxer.rigid_all(R, com2 - R @ com2)
                relaxer.rigid_all(np.eye(3), rng.normal(0, 0.8, 3))
            xr2 = relaxer.X().copy()
            sub = ff.near(relaxer.X(), 12.0)
            try:
                v2.minimize_atomic(sub, relaxer, n_steps=200, xref=xr2)
            except Exception:
                break
            E = sum(ff.energy_forces(*v2.build_atoms(relaxer))[0].values())
            X = relaxer.X().copy()
            C = contacts(X)
            d = np.sqrt(((X[:, None, :] - pk_atoms[None, :, :]) ** 2).sum(-1))
            refined.append({"E": float(E), "rmsd": rmsd_of(X, relaxer, ca0),
                            "contacts": len(C & C0) / max(len(C0), 1),
                            "clash": int((d < 2.7).sum())})
        relaxer.set_X(c["X"])
    t["refine"] = time.perf_counter() - t0

    if not refined:
        return None
    pick = min(refined, key=lambda r: r["E"])
    best_r = min(r["rmsd"] for r in refined)
    out = {"pdb": pid, "n_rec": len(prot), "n_pep": len(pep), "n_native_pairs": len(C0),
           "seed_rmsd": round(r_seed, 2), "em_best": round(em_best, 2),
           "pick_rmsd": round(pick["rmsd"], 2), "pick_contacts": round(pick["contacts"], 3),
           "pick_clash": pick["clash"], "pick_E": round(pick["E"], 1),
           "oracle_rmsd": round(best_r, 2),
           "t_seed": round(t["seed"], 1), "t_em": round(t["em"], 1),
           "t_refine": round(t["refine"], 1)}
    if verbose:
        print(f"  {pid}: {len(prot)}R+{n_res}P  种子 {out['seed_rmsd']:5.2f} → EM {out['em_best']:5.2f} "
              f"→ 选中 {out['pick_rmsd']:5.2f} Å  接触 {100*pick['contacts']:3.0f}%  碰撞 {pick['clash']:3d} "
              f"[{out['t_seed']+out['t_em']+out['t_refine']:.0f}s]")
    return out


def rmsd_of(X, relaxer, ca0):
    """给定肽全原子坐标 X，取 Cα 计算受体坐标系 RMSD"""
    idx = [k for k, (ri, an) in enumerate(relaxer.idx) if an == "CA"]
    if len(idx) < 5:
        return float("nan")
    return float(np.sqrt(((X[idx] - ca0) ** 2).sum(1).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jitter", type=int, default=16)
    ap.add_argument("--top", type=int, default=3)
    args = ap.parse_args()

    pids = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(CALDIR, "*.pdb")))
    # 只保留真正的肽-蛋白复合物（自动筛选，避免评估非复合物结构）
    qualified = []
    for pid in pids:
        try:
            ch = eb.parse_pdb(os.path.join(CALDIR, pid + ".pdb"))
            if pick_complex(ch) is not None:
                qualified.append(pid)
        except Exception:
            pass
    pids = qualified
    if args.limit:
        pids = pids[:args.limit]
    done = {}
    if os.path.exists(OUT):
        try:
            done = {r["pdb"]: r for r in json.load(open(OUT))["cases"]}
        except Exception:
            done = {}
    print(f"基准集：{len(pids)} 个 PDB 候选（已完成 {len(done)}）")
    cases = list(done.values())
    t_start = time.perf_counter()
    for i, pid in enumerate(pids):
        if pid in done:
            continue
        path = os.path.join(CALDIR, pid + ".pdb")
        try:
            r = run_one(pid, path, n_jitter=args.jitter, n_top=args.top)
        except Exception as e:
            print(f"  {pid}: 失败（{type(e).__name__}: {e}）")
            r = None
        if r:
            cases.append(r)
            json.dump({"cases": cases, "updated": time.strftime("%Y-%m-%d %H:%M")},
                      open(OUT, "w"), ensure_ascii=False, indent=1)
        if (i + 1) % 5 == 0:
            print(f"  --- {i+1}/{len(pids)} 累计 {time.perf_counter()-t_start:.0f}s ---")

    if not cases:
        print("无可评估复合物")
        return
    rms = np.array([c["pick_rmsd"] for c in cases])
    con = np.array([c["pick_contacts"] for c in cases])
    orc = np.array([c["oracle_rmsd"] for c in cases])
    tm = np.array([c["t_seed"] + c["t_em"] + c["t_refine"] for c in cases])
    n = len(cases)
    print("\n" + "=" * 76)
    print(f"基准结果（{n} 个复合物，盲式对接协议）")
    print(f"  选中位姿 RMSD：中位 {np.median(rms):6.2f} Å  均值 {rms.mean():6.2f} Å")
    print(f"  成功率   <2.5 Å: {int((rms<2.5).sum())}/{n}    <5 Å: {int((rms<5).sum())}/{n}    "
          f"<10 Å: {int((rms<10).sum())}/{n}")
    print(f"  oracle（候选内最好）：中位 {np.median(orc):6.2f} Å   <2.5 Å: {int((orc<2.5).sum())}/{n}   "
          f"<5 Å: {int((orc<5).sum())}/{n}")
    print(f"  接触恢复：中位 {100*np.median(con):3.0f}%   平均 {100*con.mean():3.0f}%")
    print(f"  耗时/复合物：中位 {np.median(tm):5.1f} s（种子 {np.median([c['t_seed'] for c in cases]):.1f}s / "
          f"EM {np.median([c['t_em'] for c in cases]):.1f}s / 精修 {np.median([c['t_refine'] for c in cases]):.1f}s）")
    json.dump({"cases": cases, "updated": time.strftime("%Y-%m-%d %H:%M"),
               "summary": {"n": n, "median_rmsd": float(np.median(rms)),
                           "hit25": int((rms < 2.5).sum()), "hit5": int((rms < 5).sum()),
                           "hit10": int((rms < 10).sum()),
                           "oracle_hit5": int((orc < 5).sum()),
                           "median_contacts": float(np.median(con)),
                           "median_sec": float(np.median(tm))}},
              open(OUT, "w"), ensure_ascii=False, indent=1)
    print(f"结果已写入 {OUT}")


if __name__ == "__main__":
    main()
