# -*- coding: utf-8 -*-
"""EPT-Dock v2 实验：OT 计划 Procrustes 确定性姿态种子
基线（v1）：盲测 0/12 < 5 Å；真区域 + 24 随机取向最好 4.26 Å（1/24）。
本实验：描述子匹配计划 → 软对应 Procrustes 种子（免采样）→ Sinkhorn-EM 细化。
  E2a 盲式种子（对全 CaM 表面匹配，不给区域信息）
  E2b 区域种子（真口袋区域，漏斗模式）
运行：python3 -I run_v2.py
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
OUT = os.path.join(HERE, "results_v2.json")


def main():
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
    Xp0 = np.array([p["x"] for p in pep_pts0])
    ca0 = np.array([pep[i][2]["CA"][0] for i in range(n_res) if "CA" in pep[i][2]])
    cen_p = Xp0.mean(0)
    true_reg_pts = [p for p in pk_pts if np.linalg.norm(p["x"] - cen_p) < 15.0]

    M = v1.MatcherV1(pep_pts0, pk_pts, n_res, pk_atoms=pk_atoms)
    relaxer = v1.TorsionRelaxer(pep_res)
    X0 = relaxer.X()

    def clashes():
        Xa = relaxer.X()
        d = np.linalg.norm(Xa[:, None, :] - pk_atoms[None, :, :], axis=2)
        ri = relaxer.res_of
        d2 = np.linalg.norm(Xa[:, None, :] - Xa[None, :, :], axis=2)
        self_m = (d2 < 2.7) & (np.abs(ri[:, None] - ri[None, :]) >= 2)
        return int((d < 2.7).sum()) + int(self_m.sum()) // 2

    def rmsd():
        ca, _ = relaxer.ca_coords()
        return float(np.sqrt(((ca - ca0) ** 2).sum(1).mean()))

    variants = [
        ("blind  e2.0 a5", None, dict(eps_seed=2.0, alpha=5.0)),
        ("blind  e2.0 a8", None, dict(eps_seed=2.0, alpha=8.0)),
        ("blind  e4.0 a5", None, dict(eps_seed=4.0, alpha=5.0)),
        ("region e2.0 a5", true_reg_pts, dict(eps_seed=2.0, alpha=5.0)),
    ]
    results = {"baseline": {"blind_24random_best": 4.26, "blind_12restart_hits": "0/12"}}
    recs = []
    print("=" * 76)
    print("E2：OT 计划 Procrustes 确定性种子 → Sinkhorn-EM（每个变体 1 个种子，免采样）")
    for name, reg_pts, kw in variants:
        for flex, k_teth in ((False, 0.0), (True, 1.0), (True, 4.0)):
            relaxer.set_X(X0)
            v1.seed_pose(M, relaxer, reg_pts if reg_pts is not None else pk_pts, verbose=False, **kw)
            r_seed = rmsd()
            xref = relaxer.X()  # 系绳参考 = 种子构象（晶体内部坐标的刚体映像）
            Pn, wres, hist = v1.dock_flex(M, relaxer, anneal=False, flex=flex,
                                          xref=xref if flex else None, k_teth=k_teth)
            rec = {"variant": name, "flex": flex, "rmsd_seed": round(r_seed, 2),
                   "rmsd": round(rmsd(), 2), "score": round(hist[-1], 1), "clash": clashes()}
            recs.append(rec)
            print(f"  [{name} flex={int(flex)}] 种子RMSD={r_seed:6.2f} → 终RMSD={rec['rmsd']:6.2f} Å"
                  f"  score={hist[-1]:8.1f}  clash={rec['clash']}")
    best = min(recs, key=lambda r: r["rmsd"])
    print(f"\n  确定性种子最好: {best['rmsd']:.2f} Å（{best['variant']}, flex={best['flex']}）"
          f"  —— 基线: 24 随机取向最好 4.26 Å / 盲测 0/12")
    results["E2"] = recs
    results["E2_best"] = best

    json.dump(results, open(OUT, "w"), ensure_ascii=False, indent=1)
    print(f"结果已写入 {OUT}")


if __name__ == "__main__":
    main()
