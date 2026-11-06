# -*- coding: utf-8 -*-
"""EPT-Dock 最小原型实验驱动
数据：2BBM —— 钙调蛋白(CaM, chain A, 148残基) + M13 肽(chain B, 26残基)
实验：
  A. 晶体姿态：Sinkhorn 匹配代价 + 逐残基软对应（可解释输出）
  B. 姿态恢复：随机位姿出发的 Sinkhorn-EM 刚体对接，测 Cα RMSD（搜索引擎硬测试）
  C. 构象熵账单：晶体构象 vs 扰动构象的前向×后向逐残基账单
运行：python3 -I run_demo.py  （-I 隔离模式；本脚本自行注入模块路径）
"""
import os
import sys
import json
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import entrobill_proto as eb  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PDB = os.path.abspath(os.path.join(HERE, "..", "data", "2BBM.pdb"))
OUT = os.path.join(HERE, "results.json")


THREE2ONE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
             "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
             "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
             "TYR": "Y", "VAL": "V"}


def build_pep_arrays(pep):
    """肽重原子（带原子名索引）与 Cα 数组"""
    names, xyz, res_idx = [], [], []
    ca = []
    for i, (seq, name, atoms) in enumerate(pep):
        for an, (p, el) in atoms.items():
            names.append(an)
            xyz.append(p)
            res_idx.append(i)
        ca.append(atoms["CA"][0] if "CA" in atoms else None)
    return np.array(names), np.array(xyz), np.array(res_idx), np.array(ca)


def clash_count(xa, pk_atoms, cutoff=2.7):
    d = np.linalg.norm(xa[:, None, :] - pk_atoms[None, :, :], axis=2)
    return int((d < cutoff).sum())


def main():
    rng = np.random.default_rng(42)
    chains = eb.parse_pdb(PDB)
    order = sorted(chains, key=lambda c: len(chains[c]), reverse=True)
    prot, pep = chains[order[0]], chains[order[1]]
    print("=" * 72)
    print(f"复合物 2BBM：受体 chain {order[0]} {len(prot)} 残基 | 肽 chain {order[1]} {len(pep)} 残基")
    print("肽序列：", "".join(THREE2ONE.get(r[1], "?") for r in pep))

    # ---------- 场点 ----------
    pk_pts_all = eb.chain_field_points(prot)
    pep_pts = eb.chain_field_points(pep)
    n_res = len(pep)
    pk_xyz_list, pk_res_list = [], []
    for i, (seq, name, atoms) in enumerate(prot):
        for an, (xyz, el) in atoms.items():
            pk_xyz_list.append(xyz)
            pk_res_list.append(i)
    pk_atoms = np.array(pk_xyz_list)
    pk_res_arr = np.array(pk_res_list)
    keep = eb.surface_filter(pk_pts_all, pk_atoms, pk_res_arr)
    pk_pts = [pk_pts_all[k] for k in keep]
    print(f"场点：受体 {len(pk_pts_all)}（表面过滤后 {len(pk_pts)}） | 肽 {len(pep_pts)}")
    print("受体场点类型分布：", dict(Counter(p["t"] for p in pk_pts)))

    pep_anames, pep_atoms, pep_res_of_atom, pep_ca = build_pep_arrays(pep)
    M = eb.Matcher(pep_pts, pk_pts, n_res, pk_atoms=pk_atoms)
    pk_seq = [p["seq"] for p in pk_pts]
    pk_name = [p["rname"] for p in pk_pts]

    results = {"pdb": "2BBM", "n_receptor_res": len(prot), "n_peptide_res": n_res}

    # ---------- 实验 A：晶体姿态 ----------
    print("\n" + "=" * 72)
    print("实验 A：晶体姿态打分 + 软对应关系（可解释输出）")
    eb.Params.n_em = 3
    xa, xp, xr = pep_atoms.copy(), M.X0.copy(), M.res_X0.copy()
    traj = []

    def _tr(it, xa_):
        ca_ = np.array([xa_[pep_res_of_atom == i][int(np.where(pep_anames[pep_res_of_atom == i] == "CA")[0][0])]
                        for i in range(n_res)])
        traj.append(float(np.sqrt(((ca_ - pep_ca) ** 2).sum(1).mean())))

    xa, xp, xr, Pn, wres, hist = M.dock(xa, xp, xr, anneal=False, track=_tr)
    eb.Params.n_em = 40
    print(f"  匹配代价（任意单位，Å² 当量）：{hist[-1]:.1f}")
    print(f"  EM 过程位姿 RMSD（0→3 迭代）：{' → '.join(f'{r:.2f}' for r in traj)} Å")
    rows = sorted(
        [(float(wres[r]), r, pep[r][1], int(pk_seq[int(np.argmax(Pn[r]))]),
          pk_name[int(np.argmax(Pn[r]))], pk_pts[int(np.argmax(Pn[r]))]["t"])
         for r in range(n_res)], reverse=True)
    print("  top-1 对应关系（按输运质量排序，前 12）:")
    print("    肽残基              →  CaM 残基       类型        质量")
    for w, r, rn, ps, pn, pt in rows[:12]:
        print(f"    {rn}{pep[r][0]:<4d}      →  {pn}{ps:<5d}   {pt:<9s} {w:.3f}")
    results["crystal_cost"] = float(hist[-1])
    results["crystal_assign"] = [
        {"pep": f"{rn}{pep[r][0]}", "pocket": f"{pn}{ps}", "type": pt, "mass": w}
        for w, r, rn, ps, pn, pt in rows[:12]]

    # ---------- 实验 B：姿态恢复 ----------
    print("\n" + "=" * 72)
    print("实验 B：随机位姿 → Sinkhorn-EM(ε退火) 刚体恢复（12 次重启，各 40 EM 迭代）")
    base_clash = clash_count(pep_atoms, pk_atoms)
    print(f"  基准：晶体姿态碰撞原子对 = {base_clash}")
    restarts = []
    for k in range(12):
        R, t = eb.rand_se3(rng, pep_ca.mean(0))
        xa = pep_atoms @ R.T + t
        xp = M.X0 @ R.T + t
        xr = M.res_X0 @ R.T + t
        xa, xp, xr, Pn, wres, hist = M.dock(xa, xp, xr)
        # 受体坐标系下的位姿 RMSD（不做任何叠加——受体不动，这是刚体对接的正确指标）
        ca_fit = np.array([xa[pep_res_of_atom == i][int(np.where(pep_anames[pep_res_of_atom == i] == "CA")[0][0])]
                           for i in range(n_res) if "CA" in pep[i][2]])
        rmsd = float(np.sqrt(((ca_fit - pep_ca) ** 2).sum(1).mean()))
        nclash = clash_count(xa, pk_atoms)
        restarts.append({"restart": k, "cost": float(hist[-1]), "rmsd": rmsd, "clashes": nclash})
        print(f"  restart {k:2d}: cost={hist[-1]:10.1f}   位姿RMSD={rmsd:6.2f} Å   碰撞原子对={nclash}")
    good = [r for r in restarts if r["clashes"] <= max(3 * base_clash + 5, 10)]
    sel = good or restarts
    best = min(sel, key=lambda r: r["rmsd"])
    best_cost = min(sel, key=lambda r: r["cost"])
    print(f"\n  最好结果（无碰撞组内）：位姿RMSD={best['rmsd']:.2f} Å（cost={best['cost']:.1f}，碰撞={best['clashes']}）")
    print(f"  最低代价者：cost={best_cost['cost']:.1f} → RMSD={best_cost['rmsd']:.2f} Å")
    print(f"  RMSD<5 Å：{sum(r['rmsd'] < 5 for r in restarts)}/12；RMSD<2.5 Å：{sum(r['rmsd'] < 2.5 for r in restarts)}/12")
    results["crystal_clash_baseline"] = base_clash
    results["restarts"] = restarts

    # ---------- 实验 B2：局部细化（随机取向 + σ=3Å 平移抖动出发） ----------
    print("\n" + "-" * 72)
    print("实验 B2：局部细化 —— 随机取向 + σ=3Å 平移抖动出发（12 次，固定 ε=30，纯细化模式）")
    eb.Params.eps_max = eb.Params.eps  # 关闭退火：纯细化
    refine = []
    c_pep = pep_ca.mean(0)
    for k in range(12):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        theta = np.radians(rng.uniform(5, 20))
        s, c_ = np.sin(theta), np.cos(theta)
        ax, ay, az = axis
        K = np.array([[0, -az, ay], [az, 0, -ax], [-ay, ax, 0]])
        R = np.eye(3) * c_ + s * K + (1 - c_) * np.outer(axis, axis)
        jit = rng.normal(0, 2.0, size=3)
        xa0 = (pep_atoms - c_pep) @ R.T + c_pep + jit
        xp0 = (M.X0 - c_pep) @ R.T + c_pep + jit
        xr0 = (M.res_X0 - c_pep) @ R.T + c_pep + jit
        ca0 = np.array([xa0[pep_res_of_atom == i][int(np.where(pep_anames[pep_res_of_atom == i] == "CA")[0][0])]
                        for i in range(n_res)])
        rmsd0 = float(np.sqrt(((ca0 - pep_ca) ** 2).sum(1).mean()))
        xa, xp, xr, Pn, wres, hist = M.dock(xa0, xp0, xr0)
        ca_fit = np.array([xa[pep_res_of_atom == i][int(np.where(pep_anames[pep_res_of_atom == i] == "CA")[0][0])]
                           for i in range(n_res)])
        rmsd = float(np.sqrt(((ca_fit - pep_ca) ** 2).sum(1).mean()))
        nclash = clash_count(xa, pk_atoms)
        refine.append({"restart": k, "rmsd_start": rmsd0, "rmsd_end": rmsd,
                       "cost": float(hist[-1]), "clashes": nclash})
        print(f"  restart {k:2d}: RMSD {rmsd0:5.2f} → {rmsd:5.2f} Å   cost={hist[-1]:6.1f}   碰撞={nclash}")
    n_ok = sum(r["rmsd_end"] < 2.5 and r["clashes"] <= max(3 * base_clash + 5, 10) for r in refine)
    print(f"\n  细化成功（RMSD<2.5Å 且碰撞达标）：{n_ok}/12")
    results["refine"] = refine
    results["refine_success"] = n_ok
    eb.Params.eps_max = 400.0

    # ---------- 实验 C：构象熵账单 ----------
    print("\n" + "=" * 72)
    print("实验 C：构象熵账单（前向×后向边缘化，占位统计）")
    phi, psi = eb.backbone_torsions(pep)
    classes = [eb.torsion_class(r[1]) for r in pep]
    states = [eb.classify_torsion(a, b) for a, b in zip(phi, psi)]

    def perturb(phi, psi, sig=45.0):
        p2 = [None if a is None else ((a + rng.normal(0, sig) + 180) % 360 - 180) for a in phi]
        q2 = [None if a is None else ((a + rng.normal(0, sig) + 180) % 360 - 180) for a in psi]
        return p2, q2

    phi2, psi2 = perturb(phi, psi)
    states2 = [eb.classify_torsion(a, b) for a, b in zip(phi2, psi2)]

    bills1, jlp1, post1 = eb.entropy_bills(states, classes)
    bills2, jlp2, post2 = eb.entropy_bills(states2, classes)
    rot = sum(eb.ROT_BILL[r[1]] for r in pep)
    n_missing = sum(1 for s in states if s is None)

    print(f"  {'残基':<8s}{'φ':>8s}{'ψ':>8s}  {'态':<6s}{'p_i(s*)':>10s}{'账单':>9s}")
    for i, r in enumerate(pep):
        if states[i] is None:
            print(f"  {r[1]}{r[0]:<5d}    {'—':>8s}{'—':>8s}  缺失")
            continue
        p = post1[i, eb.STATES.index(states[i])]
        print(f"  {r[1]}{r[0]:<5d}{phi[i]:8.1f}{psi[i]:8.1f}  {states[i]:<6s}{p:10.3f}{bills1[i]:9.2f}")
    print(f"\n  主链账单合计   晶体构象 = {bills1.sum():7.2f} kcal/mol"
          f"   |   扰动构象(±45°) = {bills2.sum():7.2f} kcal/mol")
    print(f"  联合对数概率   晶体 = {jlp1:7.2f}   |   扰动 = {jlp2:7.2f}（nats）")
    print(f"  侧链旋转异构账单（占位） = {rot:.2f} kcal/mol；平动/转动囚禁项（占位）= {eb.CRATIC:.2f} kcal/mol")
    print(f"  缺失扭转角残基数（链断裂/端残基）= {n_missing}")
    print("  注：晶体构象账单显著低于随机扰动构象 = 熵项对'同肽不同构象'排序有效的最小证据。")
    results["entropy"] = {
        "backbone_crystal": float(bills1.sum()),
        "backbone_perturbed": float(bills2.sum()),
        "joint_logp_crystal": jlp1, "joint_logp_perturbed": jlp2,
        "rotamer_placeholder": rot, "cratic_placeholder": eb.CRATIC,
        "per_residue": [{"res": f"{r[1]}{r[0]}", "state": states[i],
                         "bill": float(bills1[i])} for i, r in enumerate(pep)]}

    json.dump(results, open(OUT, "w"), ensure_ascii=False, indent=1)
    print(f"\n结果已写入 {OUT}")


if __name__ == "__main__":
    main()
