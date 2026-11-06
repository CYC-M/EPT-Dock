# -*- coding: utf-8 -*-
"""EPT-Dock 消融：熵系绳（=构象熵先验的搜索内体现）对柔性对接的作用
对照：flex 无系绳（k=0，纯力场）/ 弱系绳（k=1）/ 强系绳（k=4）/ 刚体（无柔性）
指标：位姿 RMSD、内部形变（对起始构象的 RMSD）、自碰撞
用法：python3 -I ablation_tether.py
"""
import os
import sys
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import entrobill_proto as eb  # noqa: E402
import entrobill_v1 as v1  # noqa: E402
import entrobill_v2 as v2  # noqa: E402
import benchmark as bm  # noqa: E402

OUT = os.path.join(HERE, "results_ablation_tether.json")


def rot_axis(a, th):
    a = a / np.linalg.norm(a)
    c, s = np.cos(th), np.sin(th)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) * c + s * K + (1 - c) * np.outer(a, a)


def run_one(pid, n_restart=2):
    path = os.path.abspath(os.path.join(HERE, "..", "data", "calib", pid + ".pdb"))
    if not os.path.exists(path):
        return None
    pc = bm.pick_complex(eb.parse_pdb(path))
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

    def pose_rmsd():
        ca, _ = relaxer.ca_coords()
        return float(np.sqrt(((ca - ca0) ** 2).sum(1).mean()))

    def internal_rmsd():
        """对起始构象叠加后的内部形变"""
        ca, _ = relaxer.ca_coords()
        return float(eb.kabsch_rmsd(ca, ca0))

    def self_clash():
        Xa = relaxer.X()
        ri = relaxer.res_of
        d = np.sqrt(((Xa[:, None, :] - Xa[None, :, :]) ** 2).sum(-1))
        m = (d < 2.7) & (np.abs(ri[:, None] - ri[None, :]) >= 2)
        return int(m.sum()) // 2

    X0c = ca0.copy()
    d_nat = np.sqrt(((X0[:, None, :] - pk_atoms[None, :, :]) ** 2).sum(-1))
    site = pk_atoms[np.where((d_nat < 4.5).any(0))[0]].mean(0)
    rg = float(np.sqrt(((X0 - X0.mean(0)) ** 2).sum(1).mean()))
    pocket = [p for p in pk_pts if np.linalg.norm(p["x"] - site) < max(10.0, rg + 6.0)]

    variants = [("rigid", False, 0.0), ("flex_k0", True, 0.0),
                ("flex_k1", True, 1.0), ("flex_k4", True, 4.0)]
    out = {"pdb": pid, "n_pep": n_res, "runs": {}}
    for name, flex, k in variants:
        best = None
        for rs in range(n_restart):
            rng = np.random.default_rng(3100 + rs)
            relaxer.set_X(X0)
            try:
                v1.seed_pose(M, relaxer, pocket, eps_seed=2.0, alpha=8.0, n_rounds=3)
            except Exception:
                continue
            seedX = relaxer.X().copy()
            com = seedX.mean(0)
            # 抖动 → EM（含柔性/系绳）
            for kk in range(6):
                relaxer.set_X(seedX)
                R = rot_axis(rng.normal(size=3), np.radians(rng.uniform(6, 18)))
                relaxer.rigid_all(R, com - R @ com)
                relaxer.rigid_all(np.eye(3), rng.normal(0, 1.2, 3))
                xref = relaxer.X().copy() if flex else None
                v1.dock_flex(M, relaxer, anneal=False, flex=flex,
                             xref=xref, k_teth=k)
            # 原子精修
            xr2 = relaxer.X().copy()
            try:
                sub = ff.near(relaxer.X(), 12.0)
                v2.minimize_atomic(sub, relaxer, n_steps=150, xref=xr2)
            except Exception:
                pass
            Ee, _ = ff.energy_forces(*v2.build_atoms(relaxer))
            cand = {"E": float(sum(Ee.values())), "pose": pose_rmsd(),
                    "internal": internal_rmsd(), "selfclash": self_clash()}
            if best is None or cand["E"] < best["E"]:
                best = cand
        if best:
            out["runs"][name] = best
    return out


def main():
    pids = ["2BBM", "1IWQ", "2I0L", "1A38", "1BE9", "1QJA", "1CDL", "2FLU"]
    rows = []
    print("%-6s %-10s %8s %9s %9s %8s" % ("PDB", "变体", "位姿RMSD", "内部形变", "自碰撞", "E"))
    for pid in pids:
        try:
            r = run_one(pid)
        except Exception as e:
            print(f"{pid}: 失败 {type(e).__name__}: {e}")
            continue
        if not r:
            continue
        for name, v in r["runs"].items():
            print("%-6s %-10s %8.2f %9.2f %9d %8.0f" % (
                pid, name, v["pose"], v["internal"], v["selfclash"], v["E"]))
        rows.append(r)
        print()
    if not rows:
        return
    print("=== 汇总（8 复合物 × 2 重启，按原子能量选优）===")
    import statistics as st
    for name in ("rigid", "flex_k0", "flex_k1", "flex_k4"):
        ps = [r["runs"][name]["pose"] for r in rows if name in r["runs"]]
        ins = [r["runs"][name]["internal"] for r in rows if name in r["runs"]]
        sc = [r["runs"][name]["selfclash"] for r in rows if name in r["runs"]]
        if ps:
            print(f"  {name:9s}: 位姿中位 {st.median(ps):6.2f} Å | "
                  f"内部形变中位 {st.median(ins):5.2f} Å | 自碰撞中位 {st.median(sc):4.0f} | "
                  f"位姿<5Å {sum(1 for x in ps if x<5)}/{len(ps)}")
    json.dump(rows, open(OUT, "w"), ensure_ascii=False, indent=1)
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
