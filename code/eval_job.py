# -*- coding: utf-8 -*-
"""EPT-Dock 算力节点任务：高重启引导式评估
用法：python3 eval_job.py --pdb 2BBM 1IWQ 1MXE 2I0L --restarts 10
输出：eval_result.json（逐例最优位姿 + 逐重启轨迹）
"""
import os
import sys
import json
import time
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import entrobill_proto as eb  # noqa: E402
import entrobill_v1 as v1  # noqa: E402
import entrobill_v2 as v2  # noqa: E402
import benchmark as bm  # noqa: E402


def rot_axis(a, th):
    a = a / np.linalg.norm(a)
    c, s = np.cos(th), np.sin(th)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) * c + s * K + (1 - c) * np.outer(a, a)


def eval_pdb(pid, n_restarts, rng0=9000):
    path = os.path.abspath(os.path.join(HERE, "..", "data", pid + ".pdb"))
    chains = eb.parse_pdb(path)
    pc = bm.pick_complex(chains)
    if pc is None:
        return {"pdb": pid, "error": "not a peptide complex"}
    prot, pep = pc
    n_res = len(pep)
    rec = v2.receptor_atoms(prot)
    ff = v2.AtomFF(rec)
    pk_atoms, Rresof = rec[0], rec[3]
    allpts = v1.chain_field_points(prot)
    keep = eb.surface_filter(allpts, pk_atoms, np.array(Rresof))
    pk_pts = [allpts[k] for k in keep]
    pep_res = v1.mutable_residues(pep)
    M = v1.MatcherV1(v1.chain_field_points(pep_res), pk_pts, n_res, pk_atoms=pk_atoms)
    relaxer = v1.TorsionRelaxer(pep_res)
    X0 = relaxer.X().copy()
    ca0, _ = relaxer.ca_coords()

    def rmsd():
        ca, _ = relaxer.ca_coords()
        return float(np.sqrt(((ca - ca0) ** 2).sum(1).mean()))

    def contacts(X, cut=4.5):
        d = np.sqrt(((X[:, None, :] - pk_atoms[None, :, :]) ** 2).sum(-1))
        m = d < cut
        return set(zip(relaxer.res_of[np.where(m)[0]], Rresof[np.where(m)[1]]))

    C0 = contacts(X0)
    d_nat = np.sqrt(((X0[:, None, :] - pk_atoms[None, :, :]) ** 2).sum(-1))
    site_atoms = np.where((d_nat < 4.5).any(0))[0]
    site_cen = pk_atoms[site_atoms].mean(0)
    rg = float(np.sqrt(((X0 - X0.mean(0)) ** 2).sum(1).mean()))
    prad = max(10.0, rg + 6.0)
    pocket_pts = [p for p in pk_pts if np.linalg.norm(p["x"] - site_cen) < prad]

    best = None
    trace = []
    t_all = time.perf_counter()
    for rs in range(n_restarts):
        rng = np.random.default_rng(rng0 + rs)
        t0 = time.perf_counter()
        relaxer.set_X(X0)
        v1.seed_pose(M, relaxer, pocket_pts, eps_seed=2.0, alpha=8.0, n_rounds=3)
        seedX = relaxer.X().copy()
        com = seedX.mean(0)
        pool = []
        for k in range(12):
            relaxer.set_X(seedX)
            R = rot_axis(rng.normal(size=3), np.radians(rng.uniform(6, 18)))
            relaxer.rigid_all(R, com - R @ com)
            relaxer.rigid_all(np.eye(3), rng.normal(0, 1.2, 3))
            Pn, wres, h = v1.dock_flex(M, relaxer, anneal=False, flex=False)
            pool.append({"score": h[-1], "X": relaxer.X().copy()})
        r_best = None
        for c in sorted(pool, key=lambda r: r["score"])[:3]:
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
                cand = {"E": float(sum(Ee.values())), "X": relaxer.X().copy(), "rmsd": rmsd()}
                if r_best is None or cand["E"] < r_best["E"]:
                    r_best = cand
        if r_best:
            trace.append({"restart": rs, "bestE": round(r_best["E"], 1),
                          "rmsd": round(r_best["rmsd"], 2), "sec": round(time.perf_counter() - t0, 1)})
            if best is None or r_best["E"] < best["E"]:
                best = dict(r_best)
        print(f"  [{pid}] restart {rs+1}/{n_restarts}: 本轮最优 "
              f"{(r_best['rmsd'] if r_best else float('nan')):.2f} Å / E={r_best['E']:.0f}"
              f"  (累计最优 {best['rmsd']:.2f} Å)" if (r_best and best) else f"  [{pid}] restart {rs}", flush=True)

    out = {"pdb": pid, "n_rec": len(prot), "n_pep": n_res, "n_native_pairs": len(C0),
           "restarts": n_restarts, "wall_sec": round(time.perf_counter() - t_all, 1),
           "trace": trace}
    if best:
        X = best["X"]
        C = contacts(X)
        d = np.sqrt(((X[:, None, :] - pk_atoms[None, :, :]) ** 2).sum(-1))
        # E<0 门（分级诊断，阈值待 ΔG 校准）：
        #   pass     E<0        —— 高置信结合位姿（结合净增益为负）
        #   marginal 0≤E<1000   —— 低置信（短肽/力场粗糙可使真位姿 E 略正，如 2I0L +112）
        #   unbound  E≥1000     —— 无结合位姿（悬停/碰撞态，如 1MXE 406 会落 marginal、2BBM 16909 落 unbound）
        E = best["E"]
        gate = "pass" if E < 0 else ("marginal" if E < 1000 else "unbound")
        out.update({"best_E": round(E, 1), "best_rmsd": round(best["rmsd"], 2),
                    "contacts_cover": round(len(C & C0) / max(len(C0), 1), 3),
                    "clash": int((d < 2.7).sum()), "gate": gate})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", nargs="+", required=True)
    ap.add_argument("--restarts", type=int, default=10)
    ap.add_argument("--out", default="eval_result.json")
    args = ap.parse_args()
    out = {}
    for pid in args.pdb:
        print(f"=== {pid}（{args.restarts} 重启）===", flush=True)
        try:
            out[pid] = eval_pdb(pid, args.restarts)
        except Exception as e:
            out[pid] = {"pdb": pid, "error": f"{type(e).__name__}: {e}"}
    json.dump(out, open(os.path.join(HERE, args.out), "w"), ensure_ascii=False, indent=1)
    print("完成 → eval_result.json", flush=True)


if __name__ == "__main__":
    main()
