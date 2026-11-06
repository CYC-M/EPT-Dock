# -*- coding: utf-8 -*-
"""EPT-Dock 熵账单报告生成器（CBM-PepLib 交付形态）
输入：肽-蛋白复合物 PDB + 已知口袋（自动以晶体位点模拟）→ 引导式管线（多重启）→ Markdown 报告
报告内容：结合姿态评估、能量分解、逐残基构象熵账单（标定 coil 统计）、
          关键接触清单、局限声明
用法：python3 -I report.py 2BBM [更多PDB ID]
输出：reports/<PID>-结合评估报告.md
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import entrobill_proto as eb  # noqa: E402
import entrobill_v1 as v1  # noqa: E402
import entrobill_v2 as v2  # noqa: E402
import benchmark as bm  # noqa: E402
import run_v7 as g  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CALDIR = os.path.abspath(os.path.join(HERE, "..", "data", "calib"))
REPO = os.path.join(HERE, "reports")
os.makedirs(REPO, exist_ok=True)

THREE2ONE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
             "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
             "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
             "TYR": "Y", "VAL": "V"}
CRATIC = 1.5  # Page–Jencks 量级占位


def make_report(pid, n_restart=3, rng0=9000):
    path = os.path.join(CALDIR, pid + ".pdb")
    if not os.path.exists(path):
        return None
    chains = eb.parse_pdb(path)
    pc = bm.pick_complex(chains)
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
    pep_res = v1.mutable_residues(pep)
    M = v1.MatcherV1(v1.chain_field_points(pep_res), pk_pts, n_res, pk_atoms=pk_atoms)
    relaxer = v1.TorsionRelaxer(pep_res)
    X0 = relaxer.X().copy()
    ca0, _ = relaxer.ca_coords()

    def contacts(X, cut=4.5):
        d = np.sqrt(((X[:, None, :] - pk_atoms[None, :, :]) ** 2).sum(-1))
        m = d < cut
        return set(zip(relaxer.res_of[np.where(m)[0]], Rresof[np.where(m)[1]]))

    C0 = contacts(X0)

    # 口袋（半径随肽回转半径缩放）
    d_nat = np.sqrt(((X0[:, None, :] - pk_atoms[None, :, :]) ** 2).sum(-1))
    site_atoms = np.where((d_nat < 4.5).any(0))[0]
    site_cen = pk_atoms[site_atoms].mean(0)
    rg = float(np.sqrt(((X0 - X0.mean(0)) ** 2).sum(1).mean()))
    prad = max(10.0, rg + 6.0)
    pocket_pts = [p for p in pk_pts if np.linalg.norm(p["x"] - site_cen) < prad]

    # ---- 多重启引导式管线，按原子能量取最优 ----
    best = None
    for rs in range(n_restart):
        rng = np.random.default_rng(rng0 + rs)
        relaxer.set_X(X0)
        v1.seed_pose(M, relaxer, pocket_pts, eps_seed=2.0, alpha=8.0, n_rounds=3)
        seedX = relaxer.X().copy()
        com = seedX.mean(0)
        pool = []
        for k in range(12):
            relaxer.set_X(seedX)
            R = g.rot_axis(rng.normal(size=3), np.radians(rng.uniform(6, 18)))
            relaxer.rigid_all(R, com - R @ com)
            relaxer.rigid_all(np.eye(3), rng.normal(0, 1.2, 3))
            Pn, wres, h = v1.dock_flex(M, relaxer, anneal=False, flex=False)
            pool.append({"score": h[-1], "X": relaxer.X().copy()})
        for c in sorted(pool, key=lambda r: r["score"])[:3]:
            relaxer.set_X(c["X"])
            for jit in range(2):
                if jit:
                    com2 = relaxer.X().mean(0)
                    R = g.rot_axis(rng.normal(size=3), np.radians(rng.uniform(0, 5)))
                    relaxer.rigid_all(R, com2 - R @ com2)
                    relaxer.rigid_all(np.eye(3), rng.normal(0, 0.8, 3))
                xr2 = relaxer.X().copy()
                try:
                    sub = ff.near(relaxer.X(), 12.0)
                    v2.minimize_atomic(sub, relaxer, n_steps=200, xref=xr2)
                except Exception:
                    break
                Ee, _ = ff.energy_forces(*v2.build_atoms(relaxer))
                cand = {"E": float(sum(Ee.values())), "Eparts": dict(Ee), "X": relaxer.X().copy()}
                if best is None or cand["E"] < best["E"]:
                    best = cand
        relaxer.set_X(X0)
    if best is None:
        return None
    pick = best
    X = pick["X"]
    relaxer.set_X(X)

    # 评估量（与晶体对照）
    ca, _ = relaxer.ca_coords()
    rmsd = float(np.sqrt(((ca - ca0) ** 2).sum(1).mean()))
    predC = contacts(X)
    cov = len(predC & C0) / max(len(C0), 1)
    dmin = np.sqrt(((X[:, None, :] - pk_atoms[None, :, :]) ** 2).sum(-1))
    clash = int((dmin < 2.7).sum())

    # 熵账单（标定 coil 统计）
    phi, psi = eb.backbone_torsions(pep)
    classes = [eb.torsion_class(r[1]) for r in pep]
    states = [eb.classify_torsion(a, b) for a, b in zip(phi, psi)]
    bills, jlp, post = v1.entropy_bills_v1(states, classes)
    bb_total = float(bills.sum())
    rot_total = float(sum(eb.ROT_BILL.get(r[1], 1.0) for r in pep))
    dg_est = pick["E"] * 0.01 + bb_total * 0.15 + rot_total * 0.1 + CRATIC  # 未校准示意换算

    # 关键接触
    contact_list = []
    for (pr, qr) in sorted(predC):
        dmin_pair = min(float(np.linalg.norm(xyz - xyz2))
                        for an, (xyz, e) in pep[pr][2].items()
                        for an2, (xyz2, e2) in prot[qr][2].items())
        contact_list.append((dmin_pair, f"{pep[pr][1]}{pep[pr][0]}", f"{prot[qr][1]}{prot[qr][0]}"))
    pocket_res = sorted({q for (_, q) in predC})

    md = []
    md.append(f"# 肽-靶点结合评估报告：{pid}")
    md.append("")
    md.append("- 生成：EPT-Dock v7（引导式管线 ×3 重启取最优：已知口袋 → OT 种子 → 抖动/EM → 原子精修）")
    md.append(f"- 受体：{len(prot)} 残基；肽：{n_res} 残基"
              f"（{''.join(THREE2ONE.get(r[1], '?') for r in pep)}）；口袋半径 {prad:.0f} Å")
    md.append("")
    md.append("## 1. 结合姿态评估")
    md.append("")
    md.append("| 指标 | 数值 | 说明 |")
    md.append("|---|---|---|")
    md.append(f"| 原子力场总能量 | {pick['E']:.1f} | 任意单位（未校准 kcal/mol） |")
    md.append(f"| LJ / 静电 / 氢键 | {pick['Eparts']['lj']:.1f} / {pick['Eparts']['elec']:.1f} / {pick['Eparts']['hb']:.1f} | 能量分解 |")
    md.append(f"| 重原子碰撞对 | {clash} | <5 为佳 |")
    md.append(f"| 预测界面接触 | {len(predC)} 对 | 晶体对照 {len(C0)} 对 |")
    md.append(f"| 晶体接触覆盖率 | {100*cov:.0f}% | **基准面指标** |")
    md.append(f"| Cα-RMSD（对晶体） | {rmsd:.2f} Å | 评估用；实际使用时无晶体 |")
    md.append("")
    md.append("> ⚠️ RMSD 与接触覆盖率仅用于方法验证。实际筛查场景（无晶体）时，")
    md.append("> 请以能量分解、熵账单与接触清单作为候选间**相对**比较依据。")
    md.append("")
    md.append("## 2. 构象熵账单（标定 coil 统计，前向×后向边缘化）")
    md.append("")
    md.append(f"- 主链构象熵代价合计：**{bb_total:.1f} kcal/mol**（逐残基见下表）")
    md.append(f"- 侧链旋转异构账单（Doig–Sternberg 量级）：{rot_total:.1f} kcal/mol")
    md.append(f"- 平动/转动囚禁项（Page–Jencks 量级）：{CRATIC:.1f} kcal/mol")
    md.append(f"- 示意性 ΔG 估计（**未校准**，系数为占位）：≈ {dg_est:.1f} kcal/mol")
    md.append("")
    md.append("| 残基 | φ | ψ | 态 | 后验 p | 熵代价 (kcal/mol) |")
    md.append("|---|---|---|---|---|---|")
    for i, r in enumerate(pep):
        if states[i] is None:
            md.append(f"| {r[1]}{r[0]} | — | — | 缺失 | — | — |")
        else:
            p = post[i, v1.STATES.index(states[i])]
            md.append(f"| {r[1]}{r[0]} | {phi[i]:.0f} | {psi[i]:.0f} | {states[i]} | {p:.3f} | {bills[i]:.2f} |")
    md.append("")
    hi = sorted([(float(bills[i]), i) for i in range(n_res) if states[i]], reverse=True)[:3]
    if hi:
        md.append("熵代价最高的残基：" + "、".join(
            f"**{pep[i][1]}{pep[i][0]}**（{states[i]} 态，{b:.2f} kcal/mol——结合时被迫离开偏好构象）"
            for b, i in hi))
        md.append("")
    md.append("## 3. 关键接触清单（预测界面）")
    md.append("")
    md.append("| 肽残基 | 口袋残基 | 最近重原子距离 (Å) |")
    md.append("|---|---|---|")
    for dmin_pair, ps, rs in sorted(contact_list)[:15]:
        md.append(f"| {ps} | {rs} | {dmin_pair:.1f} |")
    md.append("")
    md.append(f"口袋残基集合：{', '.join(prot[q][1] + str(prot[q][0]) for q in pocket_res[:20])}"
              + (" …" if len(pocket_res) > 20 else ""))
    md.append("")
    md.append("## 4. 局限声明")
    md.append("")
    md.append("- 能量单位未校准（ΔG 换算系数为占位），**只作候选间相对比较**，不作绝对亲和力预言")
    md.append("- 熵账单的 Ramachandran 先验/侧链/囚禁项为标定 coil 统计与文献量级，不含溶剂熵与结合诱导的口袋构象变化")
    md.append("- 肽以晶体构象为起始（未做从头构象采样）；长肽柔性受限")
    md.append("- 方法学细节与验证数据见 prototype-report-v0~v7.md")
    md.append("")
    out = os.path.join(REPO, f"{pid}-结合评估报告.md")
    open(out, "w").write("\n".join(md))
    print(f"  报告已生成: {out}   (RMSD={rmsd:.2f} Å, 接触覆盖 {100*cov:.0f}%, "
          f"碰撞 {clash}, 熵账单 {bb_total:.1f} kcal/mol)")
    return out


def main():
    pids = sys.argv[1:] or ["2BBM"]
    for pid in pids:
        try:
            make_report(pid)
        except Exception as e:
            print(f"  {pid}: 失败 {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
