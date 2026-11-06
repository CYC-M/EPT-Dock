# -*- coding: utf-8 -*-
"""EPT-Dock v1 实验（标定参数 + 柔性模块）
数据：2BBM（CaM 148 残基 + M13 肽 26 残基）
实验：
  A  晶体姿态：标定参数下的 EM 漂移与可解释对应
  B  盲测刚体恢复（12 重启， anneal）
  B1 盲测 + 扭角弛豫（柔性模块开）
  B0 打乱构象（±25° 扭角）+ 随机位姿 → 全柔性恢复
  C  熵账单（标定版 coil 统计）：晶体 vs 扰动
运行：python3 -I run_v1.py
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
OUT = os.path.join(HERE, "results_v1.json")
THREE2ONE = eb.__dict__.get("THREE2ONE", {})


def pose_rmsd(relaxer, ca0):
    ca, _ = relaxer.ca_coords()
    return float(np.sqrt(((ca - ca0) ** 2).sum(1).mean()))


def main():
    rng = np.random.default_rng(42)
    chains = eb.parse_pdb(PDB)
    order = sorted(chains, key=lambda c: len(chains[c]), reverse=True)
    prot, pep = chains[order[0]], chains[order[1]]
    n_res = len(pep)
    print("=" * 72)
    print(f"2BBM：CaM {len(prot)} 残基 | M13 {n_res} 残基 | 熵模型标定: {v1.CALIB_ENTROPY}")
    print("MORSE d*/σ（标定）:", {f"{a[:2]}|{b[:2]}": (round(d, 2), round(s, 2))
                                    for (a, b), (d, s, w) in v1.P1.MORSE.items()})
    print(f"vdw_d = {v1.P1.vdw_d:.2f}")

    # 受体
    pk_all = v1.chain_field_points(prot)
    xyz, resl = [], []
    for i, (seq, nm, at) in enumerate(prot):
        for an, (p, e) in at.items():
            xyz.append(p)
            resl.append(i)
    pk_atoms = np.array(xyz)
    keep = eb.surface_filter(pk_all, pk_atoms, np.array(resl))
    pk_pts = [pk_all[k] for k in keep]

    # 肽（可变坐标）
    pep_res = v1.mutable_residues(pep)
    pep_pts0 = v1.chain_field_points(pep_res)
    M = v1.MatcherV1(pep_pts0, pk_pts, n_res, pk_atoms=pk_atoms)
    relaxer = v1.TorsionRelaxer(pep_res)
    X0 = relaxer.X()
    ca0, _ = relaxer.ca_coords()

    def clashes():
        Xa = relaxer.X()
        d = np.linalg.norm(Xa[:, None, :] - pk_atoms[None, :, :], axis=2)
        n_rec = int((d < 2.7).sum())
        ri = relaxer.res_of
        d2 = np.linalg.norm(Xa[:, None, :] - Xa[None, :, :], axis=2)
        self_m = (d2 < 2.7) & (np.abs(ri[:, None] - ri[None, :]) >= 2)
        return n_rec + int(self_m.sum()) // 2

    results = {"calibrated": v1.CALIB_ENTROPY,
               "morse": {f"{a}|{b}": [round(d, 3), round(s, 3)] for (a, b), (d, s, w) in v1.P1.MORSE.items()},
               "vdw_d": v1.P1.vdw_d}
    print(f"场点：受体表面 {len(pk_pts)} | 肽 {len(pep_pts0)}；晶体碰撞基准 = {clashes()}")

    # ---------- A：晶体姿态 ----------
    print("\n" + "=" * 72)
    print("实验 A：晶体姿态（anneal=off, rigid）")
    relaxer.set_X(X0)
    drift = []
    v1.P1.n_em = 5
    Pn, wres, hist = v1.dock_flex(M, relaxer, anneal=False, flex=False,
                                  track=lambda it, rl: drift.append(pose_rmsd(rl, ca0)))
    v1.P1.n_em = 40
    print(f"  EM 漂移（0→4 迭代）: {' → '.join(f'{r:.2f}' for r in drift)} Å   最终 score={hist[-1]:.1f}")
    pk_seq = [p["seq"] for p in pk_pts]
    pk_nm = [p["rname"] for p in pk_pts]
    rows = sorted([(float(wres[r]), r, pep[r][1], int(pk_seq[int(np.argmax(Pn[r]))]),
                    pk_nm[int(np.argmax(Pn[r]))]) for r in range(n_res)], reverse=True)
    print("  top-6 对应: " + " | ".join(f"{rn}{pep[r][0]}→{pn}{ps}" for w, r, rn, ps, pn in rows[:6]))
    results["A"] = {"drift": drift, "score": hist[-1],
                    "assign": [{"pep": f"{rn}{pep[r][0]}", "pocket": f"{pn}{ps}", "mass": w}
                               for w, r, rn, ps, pn in rows[:12]]}

    # ---------- 盲测系列 ----------
    def blind(n_start, flex, scramble=None, label=""):
        recs = []
        for k in range(n_start):
            relaxer.set_X(X0)
            internal = 0.0
            if scramble:
                relaxer.torsion_scramble(rng, scramble)
                ca_s, _ = relaxer.ca_coords()
                internal = eb.kabsch_rmsd(ca_s, ca0)
            R, t = eb.rand_se3(rng, ca0.mean(0))
            relaxer.rigid_all(R, t)
            Pn, wres, hist = v1.dock_flex(M, relaxer, anneal=True, flex=flex)
            recs.append({"k": k, "internal": round(internal, 2),
                         "rmsd": round(pose_rmsd(relaxer, ca0), 2),
                         "score": round(hist[-1], 1), "clash": clashes()})
        ok = [r for r in recs if r["clash"] <= 12]
        best = min(ok or recs, key=lambda r: r["rmsd"])
        bscore = min(ok or recs, key=lambda r: r["score"])
        print(f"  [{label}] 最好 RMSD={best['rmsd']:.2f} Å (clash={best['clash']}) | "
              f"最低分者: score={bscore['score']:.1f}→RMSD={bscore['rmsd']:.2f} | "
              f"<5Å: {sum(r['rmsd'] < 5 for r in recs)}/{n_start}  <2.5Å: {sum(r['rmsd'] < 2.5 for r in recs)}/{n_start}")
        return recs

    print("\n" + "=" * 72)
    print("实验 B：盲测刚体恢复（12 重启）")
    results["B"] = blind(12, flex=False, label="rigid")
    print("实验 B1：盲测 + 扭角弛豫（柔性）")
    results["B1"] = blind(12, flex=True, label="flex")
    print("实验 B0：打乱构象(±25°) + 随机位姿 → 全柔性恢复")
    results["B0"] = blind(12, flex=True, scramble=25.0, label="flex+scramble")

    # ---------- C：熵账单（标定版） ----------
    print("\n" + "=" * 72)
    print("实验 C：构象熵账单（标定 coil 统计）")
    phi, psi = eb.backbone_torsions(pep)
    classes = [eb.torsion_class(r[1]) for r in pep]
    states = [eb.classify_torsion(a, b) for a, b in zip(phi, psi)]
    pert = [(None if a is None else (a + rng.normal(0, 45) + 180) % 360 - 180,
             None if b is None else (b + rng.normal(0, 45) + 180) % 360 - 180) for a, b in zip(phi, psi)]
    states2 = [eb.classify_torsion(a, b) for a, b in pert]
    b1, j1, _ = v1.entropy_bills_v1(states, classes)
    b2, j2, _ = v1.entropy_bills_v1(states2, classes)
    print(f"  主链账单：晶体 = {b1.sum():.2f} kcal/mol | 扰动(±45°) = {b2.sum():.2f} kcal/mol")
    print(f"  联合 log p：晶体 = {j1:.2f} | 扰动 = {j2:.2f} nats")
    hi = sorted([(float(b1[i]), f"{pep[i][1]}{pep[i][0]}", states[i]) for i in range(n_res)
                 if states[i]], reverse=True)[:5]
    print("  账单最高残基: " + " | ".join(f"{rn}({st}) {b:.2f}" for b, rn, st in hi))
    results["C"] = {"crystal": float(b1.sum()), "perturbed": float(b2.sum()),
                    "jlp_crystal": j1, "jlp_perturbed": j2}

    json.dump(results, open(OUT, "w"), ensure_ascii=False, indent=1)
    print(f"\n结果已写入 {OUT}")


if __name__ == "__main__":
    main()
