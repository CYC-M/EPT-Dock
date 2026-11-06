# -*- coding: utf-8 -*-
"""EPT-Dock v3b：嵌套两阶段抖动 + 分数选择 + 最优姿态 PDB 导出
v3 结论：抖动×12 最好 5.42 Å；score-RMSD 秩相关 +0.65（分数可自选）。
本实验：阶段1（24 抖动 ±10–25°）按分数选优 → 阶段2（12 抖动 ±5–12°）→ 最终姿态。
运行：python3 -I run_v3b.py
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
OUT = os.path.join(HERE, "results_v3b.json")
BEST_PDB = os.path.join(HERE, "docked_best.pdb")


def rot_axis(axis, theta):
    axis = axis / np.linalg.norm(axis)
    c, s = np.cos(theta), np.sin(theta)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) * c + s * K + (1 - c) * np.outer(axis, axis)


def write_complex_pdb(path, receptor_lines, relaxer):
    """受体原始 ATOM 行 + 变换后肽 ATOM 行"""
    idx = 0
    with open(path, "w") as fh:
        for line in receptor_lines:
            fh.write(line)
        for (seq, nm, at) in relaxer.residues:
            for an, (xyz, el) in at.items():
                idx += 1
                fh.write("ATOM  %5d %-4s %3s B%4d    %8.3f%8.3f%8.3f  1.00  0.00          %2s\n"
                         % (idx, an.ljust(3), nm, seq % 10000, xyz[0], xyz[1], xyz[2], el.upper()[:2]))


def main():
    rng = np.random.default_rng(33)
    chains = eb.parse_pdb(PDB)
    order = sorted(chains, key=lambda c: len(chains[c]), reverse=True)
    prot, pep = chains[order[0]], chains[order[1]]
    n_res = len(pep)
    receptor_lines = [ln for ln in open(PDB) if ln.startswith("ATOM") and ln[21] == order[0]]
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

    relaxer.set_X(X0)
    v1.seed_pose(M, relaxer, pk_pts, eps_seed=2.0, alpha=8.0, n_rounds=3)
    seed_X = relaxer.X()
    print(f"种子 RMSD = {rmsd():.2f} Å")

    def jitter_stage(base_X, n, ang_rng, trans_sig, tag):
        com = base_X.mean(0)
        out = []
        for k in range(n):
            relaxer.set_X(base_X)
            R = rot_axis(rng.normal(size=3), np.radians(rng.uniform(*ang_rng)))
            relaxer.rigid_all(R, com - R @ com)
            relaxer.rigid_all(np.eye(3), rng.normal(0, trans_sig, 3))
            Pn, wres, hist = v1.dock_flex(M, relaxer, anneal=False, flex=False)
            out.append({"exp": f"{tag}{k}", "rmsd": round(rmsd(), 2),
                        "score": round(hist[-1], 1), "clash": clashes(),
                        "X": relaxer.X().copy()})
        return out

    print("\n阶段 1：24 抖动 ±10–25° / ±1.5 Å → 锐化 EM")
    s1 = jitter_stage(seed_X, 24, (10, 25), 1.5, "S1-")
    ok1 = [r for r in s1 if r["clash"] <= 400]
    pick1 = min(ok1 or s1, key=lambda r: r["score"])
    best1 = min(s1, key=lambda r: r["rmsd"])
    print(f"  按分数选: RMSD={pick1['rmsd']:.2f} (score={pick1['score']:.0f}) | "
          f"实际最好: {best1['rmsd']:.2f} | <5Å: {sum(r['rmsd'] < 5 for r in s1)}/24")
    print("   " + " ".join(f"{r['rmsd']:.1f}" for r in sorted(s1, key=lambda x: x["rmsd"])))

    print("\n阶段 2：围绕分数最优位姿，12 抖动 ±5–12° / ±1.0 Å")
    s2 = jitter_stage(pick1["X"], 12, (5, 12), 1.0, "S2-")
    ok2 = [r for r in s2 if r["clash"] <= 400]
    pick2 = min(ok2 or s2, key=lambda r: r["score"])
    best2 = min(s2, key=lambda r: r["rmsd"])
    print(f"  按分数选: RMSD={pick2['rmsd']:.2f} (score={pick2['score']:.0f}) | "
          f"实际最好: {best2['rmsd']:.2f} | <5Å: {sum(r['rmsd'] < 5 for r in s2)}/12")
    print("   " + " ".join(f"{r['rmsd']:.1f}" for r in sorted(s2, key=lambda x: x["rmsd"])))

    allr = s1 + s2
    best = min(allr, key=lambda r: r["rmsd"])
    print(f"\n总 36 次：最好 {best['rmsd']:.2f} Å（{best['exp']}）；"
          f"分数选择链 S1({pick1['rmsd']:.2f}) → S2({pick2['rmsd']:.2f})")

    relaxer.set_X(pick2["X"])
    write_complex_pdb(BEST_PDB, receptor_lines, relaxer)
    results = {"stage1": [{k: v for k, v in r.items() if k != "X"} for r in s1],
               "stage2": [{k: v for k, v in r.items() if k != "X"} for r in s2],
               "picked1": pick1["rmsd"], "picked2": pick2["rmsd"], "best": best["rmsd"],
               "best_exp": best["exp"], "pdb": BEST_PDB}
    json.dump(results, open(OUT, "w"), ensure_ascii=False, indent=1)
    print(f"最优姿态复合物已写出: {BEST_PDB}")
    print(f"结果已写入 {OUT}")


if __name__ == "__main__":
    main()
