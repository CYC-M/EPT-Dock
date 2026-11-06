# -*- coding: utf-8 -*-
"""EPT-Dock v3 实验：打通 8→5 Å 最后一公里
基线（v2）：盲式确定性种子 8.01 Å（一次免采样）；随机 24 取向最好 4.26 Å。
策略（全部从确定性种子出发）：
  A  种子出发的退火 EM（ε_max = 50/150/400）——全局计划重定域
  B  种子为中心的取向抖动 ×12（±10–25°，±1.5 Å 平移）→ 锐化 EM
  C  抖动 + 温和退火（ε_max=150）×6
  D  长种子-EM（8 轮 ICP 式）
  E  score-RMSD 秩相关（管线自选能力）
运行：python3 -I run_v3.py
"""
import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import entrobill_proto as eb  # noqa: E402
import entrobill_v1 as v1  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PDB = os.path.abspath(os.path.join(HERE, "..", "data", "2BBM.pdb"))
OUT = os.path.join(HERE, "results_v3.json")


def rot_axis(axis, theta):
    axis = axis / np.linalg.norm(axis)
    c, s = np.cos(theta), np.sin(theta)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) * c + s * K + (1 - c) * np.outer(axis, axis)


def main():
    rng = np.random.default_rng(2026)
    chains = eb.parse_pdb(PDB)
    order = sorted(chains, key=lambda c: len(chains[c]), reverse=True)
    prot, pep = chains[order[0]], chains[order[1]]
    n_res = len(pep)
    xyz, resl = [], []
    for i, (seq, nm, at) in enumerate(prot):
        for an, (p, e) in at.items():
            xyz.append(p)
            resl.append(i)
    pk_atoms = np.array(xyz)
    pk_all = v1.chain_field_points(prot)
    keep = eb.surface_filter(pk_all, pk_atoms, np.array(resl))
    pk_pts = [pk_all[k] for k in keep]
    pep_res = v1.mutable_residues(pep)
    pep_pts0 = v1.chain_field_points(pep_res)
    M = v1.MatcherV1(pep_pts0, pk_pts, n_res, pk_atoms=pk_atoms)
    relaxer = v1.TorsionRelaxer(pep_res)
    X0 = relaxer.X()
    ca0, _ = relaxer.ca_coords()

    def clashes():
        Xa = relaxer.X()
        d = np.linalg.norm(Xa[:, None, :] - pk_atoms[None, :, :], axis=2)
        return int((d < 2.7).sum())

    def rmsd():
        ca, _ = relaxer.ca_coords()
        return float(np.sqrt(((ca - ca0) ** 2).sum(1).mean()))

    # ---------- 确定性种子（v2 最优变体） ----------
    relaxer.set_X(X0)
    v1.seed_pose(M, relaxer, pk_pts, eps_seed=2.0, alpha=8.0, n_rounds=3)
    seed_X = relaxer.X()
    r_seed = rmsd()
    print(f"种子 RMSD = {r_seed:.2f} Å（基线 v2: 8.79）")
    recs = []

    # ---------- A：种子出发的退火 ----------
    print("\nA. 种子出发退火 EM（刚体）")
    for emax in (50.0, 150.0, 400.0):
        v1.P1.eps_max = emax
        relaxer.set_X(seed_X)
        Pn, wres, hist = v1.dock_flex(M, relaxer, anneal=True, flex=False)
        recs.append({"exp": f"A-anneal{int(emax)}", "rmsd": round(rmsd(), 2),
                     "score": round(hist[-1], 1), "clash": clashes()})
        print(f"  ε_max={int(emax):3d}: RMSD={recs[-1]['rmsd']:6.2f} Å  score={hist[-1]:9.1f}  clash={recs[-1]['clash']}")
    v1.P1.eps_max = 400.0

    # ---------- D：长种子-EM ----------
    relaxer.set_X(X0)
    v1.seed_pose(M, relaxer, pk_pts, eps_seed=2.0, alpha=8.0, n_rounds=8)
    recs.append({"exp": "D-seedEM8", "rmsd": round(rmsd(), 2), "score": None, "clash": clashes()})
    print(f"\nD. 长种子-EM（8 轮）: 种子 RMSD={recs[-1]['rmsd']:.2f} Å")
    seed_X8 = relaxer.X()

    # ---------- B：取向抖动 ×12 → 锐化 EM ----------
    print("\nB. 种子取向抖动 ×12 → 锐化 EM（刚体）")
    jb = []
    com = seed_X.mean(0)
    for k in range(12):
        relaxer.set_X(seed_X)
        R = rot_axis(rng.normal(size=3), np.radians(rng.uniform(10, 25)))
        relaxer.rigid_all(R, com - R @ com)
        relaxer.rigid_all(np.eye(3), rng.normal(0, 1.5, 3))
        Pn, wres, hist = v1.dock_flex(M, relaxer, anneal=False, flex=False)
        jb.append({"exp": f"B-jitter{k}", "rmsd": round(rmsd(), 2),
                   "score": round(hist[-1], 1), "clash": clashes()})
    recs.extend(jb)
    print("   " + " ".join(f"{r['rmsd']:.1f}" for r in sorted(jb, key=lambda x: x['rmsd'])))

    # ---------- C：抖动 + 温和退火 ×6 ----------
    print("\nC. 抖动 + 温和退火（ε_max=150）×6")
    v1.P1.eps_max = 150.0
    jc = []
    for k in range(6):
        relaxer.set_X(seed_X)
        R = rot_axis(rng.normal(size=3), np.radians(rng.uniform(10, 25)))
        relaxer.rigid_all(R, com - R @ com)
        relaxer.rigid_all(np.eye(3), rng.normal(0, 1.5, 3))
        Pn, wres, hist = v1.dock_flex(M, relaxer, anneal=True, flex=False)
        jc.append({"exp": f"C-jitAnneal{k}", "rmsd": round(rmsd(), 2),
                   "score": round(hist[-1], 1), "clash": clashes()})
    v1.P1.eps_max = 400.0
    recs.extend(jc)
    print("   " + " ".join(f"{r['rmsd']:.1f}" for r in sorted(jc, key=lambda x: x['rmsd'])))

    # ---------- E：score-RMSD 秩相关 ----------
    valid = [r for r in recs if r["score"] is not None]
    results = {"seed_rmsd": round(r_seed, 2), "recs": recs}
    sc = np.array([r["score"] for r in valid])
    rm = np.array([r["rmsd"] for r in valid])
    rank_s = np.argsort(np.argsort(sc))
    rank_r = np.argsort(np.argsort(rm))
    rho = float(np.corrcoef(rank_s, rank_r)[0, 1])
    print(f"\nE. score-RMSD Spearman 秩相关 = {rho:.2f}（n={len(valid)}；越负越好）")
    results["spearman_score_rmsd"] = rho

    best = min(recs, key=lambda r: r["rmsd"])
    n5 = sum(1 for r in recs if r["rmsd"] < 5)
    n25 = sum(1 for r in recs if r["rmsd"] < 2.5)
    print(f"\n总计 {len(recs)} 次从种子出发：<5Å {n5} 次，<2.5Å {n25} 次，最好 {best['rmsd']:.2f} Å（{best['exp']}）")
    results["best"] = best
    results["hits_5A"] = n5
    results["hits_2.5A"] = n25
    json.dump(results, open(OUT, "w"), ensure_ascii=False, indent=1)
    print(f"结果已写入 {OUT}")


if __name__ == "__main__":
    main()
