# -*- coding: utf-8 -*-
"""EPT-Dock v6 探针：异构描述子多种子覆盖率
v5 归因：最终精度由种子决定（r=0.93），单种子仅 15% 命中 <10 Å。
本探针：K=8 个异构描述子配置各产生 1 个种子，测「至少一个种子 <10 Å」的覆盖率。
运行：python3 -I run_v6_probe.py
"""
import os
import sys
import json
import glob

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import entrobill_proto as eb  # noqa: E402
import entrobill_v1 as v1  # noqa: E402
import entrobill_v2 as v2  # noqa: E402
import benchmark as bm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CALDIR = os.path.abspath(os.path.join(HERE, "..", "data", "calib"))
OUT = os.path.join(HERE, "results_v6_probe.json")

# K=8 异构配置：距离壳 / 权重 / 邻接处理 / 受体点子采样
VARIANTS = [
    dict(name="base",     bins=((0, 6), (6, 12)),                alpha=8.0, eps=2.0, excl_adj=True,  sub=1.0),
    dict(name="shell5-10", bins=((0, 5), (5, 10)),               alpha=8.0, eps=2.0, excl_adj=True,  sub=1.0),
    dict(name="shell8-16", bins=((0, 8), (8, 16)),               alpha=8.0, eps=2.0, excl_adj=True,  sub=1.0),
    dict(name="3shell",   bins=((0, 4), (4, 8), (8, 12)),        alpha=8.0, eps=2.0, excl_adj=True,  sub=1.0),
    dict(name="strongChem", bins=((0, 6), (6, 12)),              alpha=3.0, eps=2.0, excl_adj=True,  sub=1.0),
    dict(name="sharp",    bins=((0, 6), (6, 12)),                alpha=8.0, eps=0.8, excl_adj=True,  sub=1.0),
    dict(name="noAdjExcl", bins=((0, 6), (6, 12)),               alpha=8.0, eps=2.0, excl_adj=False, sub=1.0),
    dict(name="sub70",    bins=((0, 6), (6, 12)),                alpha=8.0, eps=2.0, excl_adj=True,  sub=0.7),
]


def setup(path):
    chains = eb.parse_pdb(path)
    pc = bm.pick_complex(chains)
    if pc is None:
        return None
    prot, pep = pc
    rec = v2.receptor_atoms(prot)
    pk_atoms, Rresof = rec[0], rec[3]
    allpts = v1.chain_field_points(prot)
    keep = eb.surface_filter(allpts, pk_atoms, np.array(Rresof))
    pk_pts = [allpts[k] for k in keep]
    if len(pk_pts) < 40:
        return None
    pep_res = v1.mutable_residues(pep)
    M = v1.MatcherV1(v1.chain_field_points(pep_res), pk_pts, len(pep), pk_atoms=pk_atoms)
    relaxer = v1.TorsionRelaxer(pep_res)
    X0 = relaxer.X().copy()
    ca0, _ = relaxer.ca_coords()
    if len(ca0) < 5:
        return None
    return M, relaxer, pk_pts, X0, ca0


def main():
    pids = []
    for p in sorted(glob.glob(os.path.join(CALDIR, "*.pdb"))):
        pid = os.path.basename(p)[:-4]
        try:
            if bm.pick_complex(eb.parse_pdb(p)) is not None:
                pids.append(pid)
        except Exception:
            pass
    print(f"复合物 {len(pids)} 个；每例 {len(VARIANTS)} 个异构种子")
    cases = []
    for i, pid in enumerate(pids):
        s = setup(os.path.join(CALDIR, pid + ".pdb"))
        if s is None:
            continue
        M, relaxer, pk_pts, X0, ca0 = s
        rng = np.random.default_rng(7)
        seeds = []
        for vi, v in enumerate(VARIANTS):
            relaxer.set_X(X0)
            try:
                v1.seed_pose(M, relaxer, pk_pts, eps_seed=v["eps"], alpha=v["alpha"],
                             n_rounds=3, bins=v["bins"], excl_adj=v["excl_adj"],
                             sub=v["sub"], rng=rng)
                ca, _ = relaxer.ca_coords()
                r = float(np.sqrt(((ca - ca0) ** 2).sum(1).mean()))
            except Exception:
                r = float("nan")
            seeds.append({"variant": v["name"], "rmsd": None if np.isnan(r) else round(r, 2)})
        vals = [s["rmsd"] for s in seeds if s["rmsd"] is not None]
        best = min(vals) if vals else float("nan")
        cases.append({"pdb": pid, "seeds": seeds, "best": round(best, 2),
                      "n_under10": sum(1 for v in vals if v < 10),
                      "n_under5": sum(1 for v in vals if v < 5)})
        print(f"  {pid}: 最好 {best:5.2f} Å  <10Å {cases[-1]['n_under10']}/{len(vals)}  "
              f"<5Å {cases[-1]['n_under5']}/{len(vals)}   " +
              " ".join(f"{s['rmsd']:.1f}" if s["rmsd"] else "—" for s in seeds))
        if (i + 1) % 5 == 0:
            json.dump({"cases": cases}, open(OUT, "w"), ensure_ascii=False, indent=1)

    bests = np.array([c["best"] for c in cases])
    print("\n" + "=" * 72)
    print(f"多种子覆盖率（K={len(VARIANTS)}）:")
    print(f"  至少一个种子 <10 Å：{int((bests<10).sum())}/{len(cases)}"
          f"  （v5 单种子：3/20 = 15%）")
    print(f"  至少一个种子 <5 Å ：{int((bests<5).sum())}/{len(cases)}")
    print(f"  最好种子 RMSD 中位：{np.median(bests):.2f} Å（v5 单种子中位 ~17 Å）")
    # 各变体单独命中率
    print("\n各变体单独 <10 Å 命中率：")
    for vi, v in enumerate(VARIANTS):
        hits = sum(1 for c in cases if (c["seeds"][vi]["rmsd"] or 99) < 10)
        med = np.median([c["seeds"][vi]["rmsd"] for c in cases if c["seeds"][vi]["rmsd"]])
        print(f"  {v['name']:12s}: {hits}/{len(cases)}   中位 {med:5.2f} Å")
    json.dump({"cases": cases,
               "summary": {"n": len(cases), "k": len(VARIANTS),
                           "cover10": int((bests < 10).sum()),
                           "cover5": int((bests < 5).sum()),
                           "median_best": float(np.median(bests))}},
              open(OUT, "w"), ensure_ascii=False, indent=1)
    print(f"\n结果已写入 {OUT}")


if __name__ == "__main__":
    main()
