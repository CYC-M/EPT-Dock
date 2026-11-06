# -*- coding: utf-8 -*-
"""EPT-Dock v1 支柱3实验：场指纹反筛 + 筛选→细化漏斗
  S1  指纹判别：M13 场指纹 vs CaM 真口袋 / CaM 表面诱饵区 / 29 个诱饵蛋白
      度量 = 去偏 Sinkhorn 散度（Feydy 2019，正定非度量）
  S2  速度：单次指纹距离耗时
  E   漏斗合体：指纹 top-3 区域放置肽 → Sinkhorn-EM 细化 → 位姿 RMSD
运行：python3 -I screen_v1.py
"""
import os
import sys
import json
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import entrobill_proto as eb  # noqa: E402
import entrobill_v1 as v1  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PDB = os.path.abspath(os.path.join(HERE, "..", "data", "2BBM.pdb"))
CALDIR = os.path.abspath(os.path.join(HERE, "..", "data", "calib"))
OUT = os.path.join(HERE, "results_screen.json")


def surface_pts(residues, cap=None, rng=None):
    xyz, resl = [], []
    for i, (seq, nm, at) in enumerate(residues):
        for an, (p, e) in at.items():
            xyz.append(p)
            resl.append(i)
    xyz = np.array(xyz)
    pts = eb.chain_field_points(residues)
    keep = eb.surface_filter(pts, xyz, np.array(resl))
    pts = [pts[k] for k in keep]
    if cap and len(pts) > cap and rng is not None:
        sel = rng.choice(len(pts), cap, replace=False)
        pts = [pts[i] for i in sel]
    return pts


def main():
    rng = np.random.default_rng(7)
    chains = eb.parse_pdb(PDB)
    order = sorted(chains, key=lambda c: len(chains[c]), reverse=True)
    prot, pep = chains[order[0]], chains[order[1]]
    n_res = len(pep)
    pk_pts = surface_pts(prot)
    pep_res = v1.mutable_residues(pep)
    pep_pts0 = v1.chain_field_points(pep_res)
    Xp0 = np.array([p["x"] for p in pep_pts0])
    pep_cen = Xp0.mean(0)
    Y = np.array([p["x"] for p in pk_pts])

    # ---------- S1：指纹判别 ----------
    d_true = np.linalg.norm(Y - pep_cen, axis=1)
    true_reg = Y[d_true < 15.0]
    print("=" * 72)
    print(f"S1 场指纹判别：真口袋 {len(true_reg)} 点 | 肽指纹 {len(Xp0)} 点")
    cands = [("TRUE CaM groove", true_reg)]
    centers = []
    while len(centers) < 10:
        c = Y[rng.integers(len(Y))]
        if np.linalg.norm(c - pep_cen) > 20.0 and all(np.linalg.norm(c - x) > 15.0 for x in centers):
            centers.append(c)
    for k, c in enumerate(centers):
        reg = Y[np.linalg.norm(Y - c, axis=1).argsort()[:60]]
        cands.append((f"CaM decoy-{k}", reg))
    n_prot_dec = 0
    for pid in sorted(os.listdir(CALDIR)):
        if not pid.endswith(".pdb"):
            continue
        try:
            ch = eb.parse_pdb(os.path.join(CALDIR, pid))
            best = max(ch.values(), key=lambda r: len(r))
            if len(best) < 60:
                continue
            dec = np.array([p["x"] for p in surface_pts(best, cap=300, rng=rng)])
            if dec.ndim != 2 or len(dec) < 30:
                continue
            cands.append((pid.replace(".pdb", ""), dec))
            n_prot_dec += 1
        except Exception:
            pass
    print(f"候选区总数 {len(cands)}（CaM 区 11 + 诱饵蛋白 {n_prot_dec}）")

    t0 = time.perf_counter()
    scores = []
    for name, reg in cands:
        s = v1.sinkhorn_div(Xp0, reg, eps=100.0, iters=200)
        scores.append((float(s), name))
    dt = (time.perf_counter() - t0) / len(cands)
    scores.sort()
    rank = [i for i, (s, n) in enumerate(scores) if n == "TRUE CaM groove"][0] + 1
    print(f"  真口袋排名: {rank}/{len(cands)}   (单次 {dt*1000:.1f} ms)")
    for s, n in scores[:5]:
        print(f"    {s:9.2f}  {n}")
    results = {"S1_rank": rank, "S1_n_cands": len(cands), "S1_ms": dt * 1000,
               "S1_top5": [{"name": n, "score": s} for s, n in scores[:5]]}

    # ---------- E：筛选→细化漏斗 ----------
    print("\n" + "=" * 72)
    print("E 漏斗合体：指纹 top-3 区域 → 放置 → Sinkhorn-EM 细化（3 取向 × 每区）")
    xyz, resl = [], []
    for i, (seq, nm, at) in enumerate(prot):
        for an, (p, e) in at.items():
            xyz.append(p)
            resl.append(i)
    pk_atoms = np.array(xyz)
    M = v1.MatcherV1(pep_pts0, pk_pts, n_res, pk_atoms=pk_atoms)
    relaxer = v1.TorsionRelaxer(pep_res)
    X0 = relaxer.X()
    ca0, _ = relaxer.ca_coords()

    def clashes():
        Xa = relaxer.X()
        d = np.linalg.norm(Xa[:, None, :] - pk_atoms[None, :, :], axis=2)
        ri = relaxer.res_of
        d2 = np.linalg.norm(Xa[:, None, :] - Xa[None, :, :], axis=2)
        self_m = (d2 < 2.7) & (np.abs(ri[:, None] - ri[None, :]) >= 2)
        return int((d < 2.7).sum()) + int(self_m.sum()) // 2

    top3 = [n for s, n in scores[:3]]
    erecs = []
    for name in top3:
        reg = dict(cands)[name]
        cen = reg.mean(0)
        for ori in range(3):
            relaxer.set_X(X0)
            # 平移肽质心到区域质心，随机取向
            com = X0.mean(0)
            relaxer.rigid_all(np.eye(3), cen - com)
            q = rng.normal(size=4)
            q /= np.linalg.norm(q)
            w_, x_, y_, z_ = q
            R = np.array([[1 - 2 * (y_ * y_ + z_ * z_), 2 * (x_ * y_ - w_ * z_), 2 * (x_ * z_ + w_ * y_)],
                          [2 * (x_ * y_ + w_ * z_), 1 - 2 * (x_ * x_ + z_ * z_), 2 * (y_ * z_ - w_ * x_)],
                          [2 * (x_ * z_ - w_ * y_), 2 * (y_ * z_ + w_ * x_), 1 - 2 * (x_ * x_ + y_ * y_)]])
            com2 = relaxer.X().mean(0)
            relaxer.rigid_all(R, cen - com2)
            Pn, wres, hist = v1.dock_flex(M, relaxer, anneal=True, flex=True)
            ca, _ = relaxer.ca_coords()
            rmsd = float(np.sqrt(((ca - ca0) ** 2).sum(1).mean()))
            erecs.append({"region": name, "ori": ori, "rmsd": round(rmsd, 2),
                          "score": round(hist[-1], 1), "clash": clashes()})
            print(f"  区[{name[:14]:14s}] 取向{ori}: RMSD={rmsd:6.2f} Å  score={hist[-1]:8.1f}  clash={clashes()}")
    best = min(erecs, key=lambda r: r["rmsd"])
    print(f"  漏斗最好: {best['rmsd']:.2f} Å（区域 {best['region']}）")
    results["E"] = erecs
    results["E_best"] = best

    json.dump(results, open(OUT, "w"), ensure_ascii=False, indent=1)
    print(f"\n结果已写入 {OUT}")


if __name__ == "__main__":
    main()
