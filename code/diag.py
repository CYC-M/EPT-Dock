# -*- coding: utf-8 -*-
"""晶体姿态能量面解剖：找出 EM 在晶体姿态第一步就被推离的原因"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import entrobill_proto as eb  # noqa: E402

PDB = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "2BBM.pdb"))

chains = eb.parse_pdb(PDB)
order = sorted(chains, key=lambda c: len(chains[c]), reverse=True)
prot, pep = chains[order[0]], chains[order[1]]
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

M = eb.Matcher(pep_pts, pk_pts, n_res, pk_atoms=pk_atoms)

E = M.pair_energy(M.X0, M.U1)
d = np.sqrt(((M.X0[:, None, :] - M.Y[None, :, :]) ** 2).sum(-1))

print("=== 每个肽点的能量分量（前 5 候选）===")
interesting = [i for i, p in enumerate(pep_pts) if p["label"] in ("TRP-sc", "NZ", "ARG-guan", "N", "O", "OD1", "OE2")][:18]
for i in interesting:
    p = pep_pts[i]
    order_i = np.argsort(E[i])[:5]
    print(f"\n肽点 {p['rname']}{p['seq']}.{p['label']}[{p['t']}]  min E={E[i].min():8.2f}")
    for j in order_i:
        q = pk_pts[j]
        # 重算能量分量
        dd = (d[i, j] - M.dst[i, j]) / M.sig[i, j]
        morse = M.epw[i, j] * ((1 - np.exp(-dd)) ** 2 - 1)
        wall = Params_wall = 100.0 * max(0.0, 3.0 - d[i, j]) ** 2
        base = morse if M.compat[i, j] else wall
        loc = 0.15 * d[i, j] ** 2
        dot = M.U1[i] @ M.U2[j]
        df = 0.35 + 0.65 * min(1.0, max(0.0, -dot)) if M.dir_mask[i, j] else 1.0
        print(f"   CaM {q['rname']}{q['seq']}.{q['label']}[{q['t']}] d={d[i, j]:5.2f} "
              f"morse/wall={base:8.2f} dirfac={df:4.2f} loc={loc:5.2f} E={E[i, j]:8.2f} "
              f"u_dot={dot:6.2f}")

print("\n=== 能量面统计 ===")
mins = E.min(1)
print(f"attract = mean_i min_p E = {mins.mean():8.2f}")
print(f"min E < -10 的肽点比例: {(mins < -10).mean():.2f};  < 0: {(mins < 0).mean():.2f}")
print(f"晶体下 min E 分布: p10={np.percentile(mins,10):.1f} p50={np.percentile(mins,50):.1f} p90={np.percentile(mins,90):.1f}")

print("\n=== 晶体姿态下的计划与 targets 漂移 ===")
P = M.sinkhorn(M.cost(M.X0, M.U1), 5.0)
Pagg = np.zeros((n_res, len(pk_pts)))
for r, ix in enumerate(M.res_idx):
    if len(ix):
        Pagg[r] = P[ix].sum(0)
Pn = Pagg / (Pagg.sum(1, keepdims=True) + 1e-12)
targets = Pn @ M.Y
x_res = M.res_X0
drift = np.linalg.norm(targets - x_res, axis=1)
print("残基  |target−x_res|  质量和")
for r in range(n_res):
    print(f"  {pep[r][1]}{pep[r][0]:<4d}  {drift[r]:6.2f} Å   {Pagg[r].sum():.4f}")

# Kabsch 步长
R, t = eb.weighted_kabsch(x_res, targets, np.maximum(Pagg.sum(1), 1e-9))
moved = (x_res @ R.T + t) - x_res
print(f"\nKabsch 刚体步平均位移 = {np.linalg.norm(moved, axis=1).mean():.2f} Å")
# 排斥弛豫步（晶体姿态）
d_atoms = None
xa = np.array([xyz for (seq, name, atoms) in pep for an, (xyz, el) in atoms.items()])
dd2 = np.sqrt(((xa[:, None, :] - pk_atoms[None, :, :]) ** 2).sum(-1))
print(f"晶体肽-受体原子对 <2.9Å: {(dd2 < 2.9).sum()}；<3.2Å: {(dd2 < 3.2).sum()}；最小 {dd2.min():.2f} Å")

print("\n=== 晶体接触距离统计（d_pref 标定依据）===")
import entrobill_proto as _eb
P = _eb.Params
for (ta, tb) in sorted(P.MORSE):
    tm = np.array([[(a == ta and b == tb) for b in M.pkt] for a in M.pt]) & (d < 8) & M.compat
    vals = d[tm]
    if len(vals) > 5:
        print(f"{ta:9s}×{tb:9s}: n={len(vals):4d}  p10={np.percentile(vals,10):5.2f}  "
              f"p25={np.percentile(vals,25):5.2f}  p50={np.percentile(vals,50):5.2f}")
nc = (~M.compat) & (d < 6)
vals = d[nc]
print(f"{'non-compat':18s} d<6: n={len(vals):4d}  p10={np.percentile(vals,10):5.2f}  "
      f"p25={np.percentile(vals,25):5.2f}  p50={np.percentile(vals,50):5.2f}")

