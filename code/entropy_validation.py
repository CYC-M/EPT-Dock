# -*- coding: utf-8 -*-
"""EPT-Dock 熵模型验证（正确设计）：同一肽的构象排序能力
检验：给定晶体构象 + N 个扭角扰动构象，熵账单是否
  ① 把晶体（天然样）构象排在最低熵代价
  ② 账单随构象偏离度单调上升（Spearman）
这是熵模型的本质作用——不是跨复合物比较（那被肽长混淆）。
用法：python3 -I entropy_validation.py
"""
import os
import sys
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import entrobill_proto as eb  # noqa: E402
import entrobill_v1 as v1  # noqa: E402
import benchmark as bm  # noqa: E402

OUT = os.path.join(HERE, "results_entropy_validation.json")


def spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def conformations(pep, n=120, seeds=(11, 22, 33, 44)):
    """用 TorsionRelaxer 生成同肽的扭角扰动构象（不同扰动幅度，混合分布）"""
    relaxer = v1.TorsionRelaxer(v1.mutable_residues(pep))
    X0 = relaxer.X().copy()
    ca0, _ = relaxer.ca_coords()
    out = [("crystal", 0.0, X0.copy(), 0.0)]
    rng = np.random.default_rng(seeds[0])
    sigmas = [5, 10, 15, 20, 30, 40, 60, 90]
    per = max(1, n // len(sigmas))
    for sig in sigmas:
        for _ in range(per):
            relaxer.set_X(X0)
            # 逐扭角独立扰动
            for kind, ri, mask in relaxer.tors:
                relaxer.rotate(kind, ri, np.radians(rng.normal(0, sig)))
            ca, _ = relaxer.ca_coords()
            r = float(np.sqrt(((ca - ca0) ** 2).sum(1).mean()))
            out.append((f"sig{sig}", sig, relaxer.X().copy(), r))
    return out, relaxer, ca0


def bills_of(pep, X):
    """给定肽全原子坐标 → 熵账单（用坐标算 φ/ψ）"""
    relaxer = v1.TorsionRelaxer(v1.mutable_residues(pep))
    relaxer.set_X(X)
    res_now = relaxer.residues
    phi, psi = eb.backbone_torsions([(s, n, {a: (np.array(x), e) for a, (x, e) in at.items()})
                                     for (s, n, at) in res_now])
    cls = [eb.torsion_class(r[1]) for r in pep]
    st = [eb.classify_torsion(a, b) for a, b in zip(phi, psi)]
    bills, jlp, post = v1.entropy_bills_v1(st, cls)
    return float(bills.sum()), jlp


def main():
    pids = ["2BBM", "1CDL", "1MXE", "1QJA", "1BXL", "2FLU", "1YCR", "2I0L", "1BE9", "1A38"]
    rows = []
    print("%-6s %5s %8s %9s %9s %8s" % ("PDB", "构象", "账单范围", "与RMSD相关", "晶体排名", "最优排名"))
    for pid in pids:
        path = os.path.abspath(os.path.join(HERE, "..", "data", "calib", pid + ".pdb"))
        if not os.path.exists(path):
            continue
        try:
            pc = bm.pick_complex(eb.parse_pdb(path))
            if pc is None:
                continue
            prot, pep = pc
            confs, relaxer, ca0 = conformations(pep)
            recs = []
            for tag, sig, X, rmsd in confs:
                tot, jlp = bills_of(pep, X)
                recs.append({"tag": tag, "sigma": sig, "rmsd": rmsd, "bill": tot, "jlp": jlp})
            bl = np.array([r["bill"] for r in recs])
            rm = np.array([r["rmsd"] for r in recs])
            rho = spearman(bl, rm)
            # 晶体构象在账单排序中的位次（1 = 最低账单）
            order = np.argsort(bl)
            crystal_rank = int(np.where(order == 0)[0][0]) + 1
            # 对照：随机选一个构象的期望排名
            rows.append({"pdb": pid, "n_conf": len(recs), "bill_min": float(bl.min()),
                         "bill_max": float(bl.max()), "rho_bill_rmsd": rho,
                         "crystal_rank": crystal_rank,
                         "pct_crystal_rank": crystal_rank / len(recs)})
            print("%-6s %5d %12s %9.2f %9d %10s" % (
                pid, len(recs), f"{bl.min():.0f}-{bl.max():.0f}", rho, crystal_rank,
                f"{crystal_rank}/{len(recs)}"))
        except Exception as e:
            print(f"{pid}: 失败 {type(e).__name__}: {e}")

    if rows:
        rhos = np.array([r["rho_bill_rmsd"] for r in rows])
        pcts = np.array([r["pct_crystal_rank"] for r in rows])
        print()
        print("=== 汇总 ===")
        print(f"  账单 vs 构象偏离度(RMSD) 的 Spearman：中位 {np.median(rhos):+.2f}，"
              f"正相关 {int((rhos>0).sum())}/{len(rows)} 例")
        print(f"  晶体构象的账单排名百分位：中位 {np.median(pcts):.3f}"
              f"（随机期望 0.5；越小越好）")
        print(f"  晶体排名进前 10%：{int((pcts<=0.10).sum())}/{len(rows)} 例")
        json.dump(rows, open(OUT, "w"), ensure_ascii=False, indent=1)
        print(f"→ {OUT}")


if __name__ == "__main__":
    main()
