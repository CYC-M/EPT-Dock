# -*- coding: utf-8 -*-
"""EPT-Dock v4b：EM ⇄ 原子层混合迭代（打通 5.4 Å → <3 Å）
v4 诊断：原子层自身吸引盆 ~4 Å、保持晶体 1.32 Å；但 EM 停在其外（5.4–6.5 Å）。
本实验：抖动 → EM(锐化, 分两段) ⇄ 原子极小化，让两层交替收敛。
运行：python3 -I run_v4b.py
"""
import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import entrobill_proto as eb  # noqa: E402
import entrobill_v1 as v1  # noqa: E402
import entrobill_v2 as v2  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PDB = os.path.abspath(os.path.join(HERE, "..", "data", "2BBM.pdb"))
OUT = os.path.join(HERE, "results_v4b.json")


def rot_axis(axis, theta):
    axis = axis / np.linalg.norm(axis)
    c, s = np.cos(theta), np.sin(theta)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) * c + s * K + (1 - c) * np.outer(axis, axis)


def main():
    chains = eb.parse_pdb(PDB)
    order = sorted(chains, key=lambda c: len(chains[c]), reverse=True)
    prot, pep = chains[order[0]], chains[order[1]]
    n_res = len(pep)
    rec = v2.receptor_atoms(prot)
    ff_full = v2.AtomFF(rec)
    pk_atoms = rec[0]
    pk_all = v1.chain_field_points(prot)
    keep = eb.surface_filter(pk_all, pk_atoms, np.array(rec[3]))
    pk_pts = [pk_all[k] for k in keep]
    pep_res = v1.mutable_residues(pep)
    pep_pts0 = v1.chain_field_points(pep_res)
    M = v1.MatcherV1(pep_pts0, pk_pts, n_res, pk_atoms=pk_atoms)
    relaxer = v1.TorsionRelaxer(pep_res)
    X0 = relaxer.X()
    ca0, _ = relaxer.ca_coords()

    def rmsd():
        ca, _ = relaxer.ca_coords()
        return float(np.sqrt(((ca - ca0) ** 2).sum(1).mean()))

    def atom_min(n_steps=160):
        P = v2.build_atoms(relaxer)
        sub = ff_full.near(P[0], 12.0)
        xref = relaxer.X().copy()
        v2.minimize_atomic(sub, relaxer, n_steps=n_steps, xref=xref)
        E = ff_full.energy_forces(*v2.build_atoms(relaxer))[0]
        return sum(E.values())

    results = {}
    print("=" * 76)
    print("v4b：抖动 → 混合迭代（EM 15 轮 ⇄ 原子 160 步）×3 → 按原子能量选")
    pipe = []
    for rseed in (33, 77, 101):
        rng = np.random.default_rng(rseed)
        relaxer.set_X(X0)
        v1.seed_pose(M, relaxer, pk_pts, eps_seed=2.0, alpha=8.0, n_rounds=3)
        seedX = relaxer.X()
        # 抖动池
        com = seedX.mean(0)
        pool = []
        for k in range(16):
            relaxer.set_X(seedX)
            R = rot_axis(rng.normal(size=3), np.radians(rng.uniform(10, 25)))
            relaxer.rigid_all(R, com - R @ com)
            relaxer.rigid_all(np.eye(3), rng.normal(0, 1.5, 3))
            Pn, wres, h = v1.dock_flex(M, relaxer, anneal=False, flex=False)
            pool.append({"score": h[-1], "X": relaxer.X().copy(), "rmsd0": rmsd()})
        cand = sorted(pool, key=lambda r: r["score"])[:4]
        out = []
        for c in cand:
            relaxer.set_X(c["X"])
            trace = [("EM", rmsd())]
            for rnd in range(3):
                # EM 段（关退火，短）
                v1.P1.n_em = 15
                v1.dock_flex(M, relaxer, anneal=False, flex=False)
                trace.append((f"EM{rnd}", rmsd()))
                # 原子段
                E = atom_min(160)
                trace.append((f"at{rnd}", rmsd()))
            out.append({"trace": [(t, round(r, 2)) for t, r in trace],
                        "rmsd": rmsd(), "E": E})
        v1.P1.n_em = 40
        pick = min(out, key=lambda r: r["E"])
        best = min(r["rmsd"] for r in out)
        pipe.append({"rseed": rseed, "final": round(pick["rmsd"], 2),
                     "best": round(best, 2), "E": round(pick["E"], 1),
                     "trace": pick["trace"]})
        print(f"  rseed={rseed}: 选中 {pick['rmsd']:5.2f} Å（候选最好 {best:5.2f}）")
        print(f"     轨迹: " + " → ".join(f"{t}:{r:.1f}" for t, r in pick["trace"]))
    results["pipe"] = pipe
    fin = [p["final"] for p in pipe]
    bst = [p["best"] for p in pipe]
    print(f"\n按原子能量自选: <3Å {sum(1 for f in fin if f<3)}/3  <5Å {sum(1 for f in fin if f<5)}/3  均值 {np.mean(fin):.2f}")
    print(f"oracle 最好: <3Å {sum(1 for f in bst if f<3)}/3  <5Å {sum(1 for f in bst if f<5)}/3  均值 {np.mean(bst):.2f}")
    json.dump(results, open(OUT, "w"), ensure_ascii=False, indent=1)
    print(f"结果已写入 {OUT}")


if __name__ == "__main__":
    main()
