# -*- coding: utf-8 -*-
"""EPT-Dock v4 实验：原子级局部极小化层
V1  晶体姿态在原子 FF 下的漂移（场点层为 ~2 Å；期望 <1 Å）
V2  吸引盆宽度：晶体扰动 1–4 Å → 原子极小化 → 恢复率
V3  全管道：种子 → 嵌套抖动 → 锐化 EM（~6 Å）→ 原子极小化（多随机种子）
运行：python3 -I run_v4.py
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
OUT = os.path.join(HERE, "results_v4.json")


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
    ff = v2.AtomFF(rec)
    pep_res = v1.mutable_residues(pep)
    relaxer = v1.TorsionRelaxer(pep_res)
    X0 = relaxer.X()
    ca0, _ = relaxer.ca_coords()

    # 场点层（用于 V3 前段）
    pk_atoms = rec[0]
    pk_all = v1.chain_field_points(prot)
    keep = eb.surface_filter(pk_all, pk_atoms, np.array(rec[3]))
    pk_pts = [pk_all[k] for k in keep]
    pep_pts0 = v1.chain_field_points(pep_res)
    M = v1.MatcherV1(pep_pts0, pk_pts, n_res, pk_atoms=pk_atoms)

    def rmsd():
        ca, _ = relaxer.ca_coords()
        return float(np.sqrt(((ca - ca0) ** 2).sum(1).mean()))

    results = {"pdb": "2BBM"}
    print("=" * 76)
    print("V1 晶体姿态在原子力场下的极小化漂移")
    relaxer.set_X(X0)
    E0, _ = ff.energy_forces(*v2.build_atoms(relaxer))
    e_start = sum(E0.values())
    print(f"  晶体姿态 E = {e_start:.1f} kcal/mol  ({ {k: round(v,1) for k,v in E0.items()} })")
    hist = v2.minimize_atomic(ff, relaxer, n_steps=150, verbose=False)
    d_crystal = rmsd()
    print(f"  极小化后：E = {hist[-1]:.1f}  ΔE = {hist[-1]-e_start:+.1f}  "
          f"位姿漂移 = {d_crystal:.2f} Å   （场点层基线 ~2.0 Å）")
    results["V1"] = {"E_start": e_start, "E_end": hist[-1], "drift": round(d_crystal, 2)}

    print("\nV2 吸引盆宽度：晶体扰动 → 原子极小化")
    rng = np.random.default_rng(5)
    v2r = []
    for pert in (1.0, 2.0, 3.0, 4.0, 6.0):
        recs = []
        for k in range(3):
            relaxer.set_X(X0)
            R = rot_axis(rng.normal(size=3), np.radians(pert * 4.0))
            cen = X0.mean(0)
            relaxer.rigid_all(R, cen - R @ cen)
            d = rng.normal(size=3)
            d /= np.linalg.norm(d)
            relaxer.rigid_all(np.eye(3), d * pert)
            r0 = rmsd()
            v2.minimize_atomic(ff, relaxer, n_steps=150)
            recs.append((r0, rmsd()))
        recs = np.array(recs)
        print(f"  起始 {recs[:,0].mean():4.2f} Å → 恢复 {recs[:,1].mean():4.2f} Å "
              f"（3 次；最好 {recs[:,1].min():.2f}）")
        v2r.append({"pert": pert, "start": round(float(recs[:, 0].mean()), 2),
                    "end": round(float(recs[:, 1].mean()), 2),
                    "best": round(float(recs[:, 1].min()), 2)})
    results["V2"] = v2r

    print("\n" + "=" * 76)
    print("V3 全管道：种子 → 嵌套抖动 → 锐化 EM → (top-K 原子极小化 + 盆地跳跃)")
    pipe = []
    for rseed in (33, 77, 101):
        rng = np.random.default_rng(rseed)
        relaxer.set_X(X0)
        v1.seed_pose(M, relaxer, pk_pts, eps_seed=2.0, alpha=8.0, n_rounds=3)
        r_seed = rmsd()

        def stage(base_X, n, ang, tr):
            com = base_X.mean(0)
            out = []
            for k in range(n):
                relaxer.set_X(base_X)
                R = rot_axis(rng.normal(size=3), np.radians(rng.uniform(*ang)))
                relaxer.rigid_all(R, com - R @ com)
                relaxer.rigid_all(np.eye(3), rng.normal(0, tr, 3))
                Pn, wres, hist_e = v1.dock_flex(M, relaxer, anneal=False, flex=False)
                out.append({"rmsd": rmsd(), "score": hist_e[-1], "X": relaxer.X().copy()})
            return out

        s1 = stage(relaxer.X(), 24, (10, 25), 1.5)
        p1 = min(s1, key=lambda r: r["score"])
        s2 = stage(p1["X"], 12, (5, 12), 1.0)
        pool = s1 + s2
        em_best = min(r["rmsd"] for r in pool)

        # top-K 按场点分数 → 原子极小化 + 盆地跳跃
        top = sorted(pool, key=lambda r: r["score"])[:5]
        refined = []
        for cand in top:
            for jit_ang, jit_tr in ((0.0, 0.0), (5.0, 0.8), (5.0, 0.8)):
                relaxer.set_X(cand["X"])
                if jit_ang > 0:
                    com = relaxer.X().mean(0)
                    R = rot_axis(rng.normal(size=3), np.radians(rng.uniform(0, jit_ang)))
                    relaxer.rigid_all(R, com - R @ com)
                    relaxer.rigid_all(np.eye(3), rng.normal(0, jit_tr, 3))
                xref = relaxer.X().copy()
                v2.minimize_atomic(ff, relaxer, n_steps=200, xref=xref)
                Eend = ff.energy_forces(*v2.build_atoms(relaxer))[0]
                refined.append({"E": sum(Eend.values()), "rmsd": rmsd()})
        pick = min(refined, key=lambda r: r["E"])   # 按原子能量自选
        sortd = sorted(refined, key=lambda r: r["E"])
        pipe.append({"rseed": rseed, "seed": round(r_seed, 2), "em_best": round(em_best, 2),
                     "em_pick": None, "final": round(pick["rmsd"], 2),
                     "final_E": round(pick["E"], 1),
                     "refine_best": round(min(r["rmsd"] for r in refined), 2),
                     "top3_by_E": [round(r["rmsd"], 2) for r in sortd[:3]]})
        print(f"  rseed={rseed}: 种子 {r_seed:5.2f} → EM 抖动最好 {em_best:5.2f} "
              f"→ 原子精修后按能量选 {pick['rmsd']:5.2f} Å（该轮最好 {pipe[-1]['refine_best']:.2f}）")
    results["V3"] = pipe
    finals = [p["final"] for p in pipe]
    bests = [p["refine_best"] for p in pipe]
    print(f"\n按原子能量自选: <2.5Å {sum(1 for f in finals if f<2.5)}/3   "
          f"<5Å {sum(1 for f in finals if f<5)}/3   均值 {np.mean(finals):.2f} Å")
    print(f"候选内最好（oracle）: <2.5Å {sum(1 for f in bests if f<2.5)}/3   "
          f"<5Å {sum(1 for f in bests if f<5)}/3   均值 {np.mean(bests):.2f} Å")

    json.dump(results, open(OUT, "w"), ensure_ascii=False, indent=1)
    print(f"结果已写入 {OUT}")


if __name__ == "__main__":
    main()
