# -*- coding: utf-8 -*-
"""EPT-Dock v4：原子级局部极小化层

定位（v3 确诊）：场点 OT 层的职责 = 搜索/对应/熵（精度上限 ~5 Å，晶体位姿亦被漂走 ~2 Å）；
Å 级精度交给经典原子项层——正是设计文档统一打分公式里的 E_vdw + E_HB + E_elec。

实现：
  E = Σ_LJ  [eps_ij((rmin/r)^12 − 2(rmin/r)^6)]
    + Σ_elec 332·q_i q_j / (4·r²)     （距离依赖介电 ε=4r）
    + Σ_HB  −depth·exp(−((d−d0)/σ)²) ·(0.5 + 0.5·(−û_i·û_j))   （方向性氢键）
自由度 = 刚体(6) + 主链 φ/ψ；解析力 → 扭矩投影（复用 TorsionRelaxer）
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import entrobill_proto as eb  # noqa: E402
import entrobill_v1 as v1  # noqa: E402

# ---------------- 参数 ----------------
LJ = {"C": (1.70, 0.086), "N": (1.55, 0.170), "O": (1.50, 0.210), "S": (1.80, 0.250),
      "P": (1.80, 0.200), "H": (1.20, 0.030)}
CHARGE = {  # 简化的残基/原子部分电荷（e）
    "N": -0.35, "CA": 0.10, "C": 0.45, "O": -0.45, "OXT": -0.55,
    "OD1": -0.55, "OD2": -0.55, "OE1": -0.55, "OE2": -0.55,
    "ND1": -0.10, "NE2": -0.10, "ND2": -0.10, "NE1": -0.20, "OE1_GLN": -0.45,
    "NZ": 1.00, "NH1": 0.33, "NH2": 0.33, "NE": 0.34,
    "OG": -0.35, "OG1": -0.35, "OH": -0.35, "SG": -0.15,
    "SD": 0.06, "CG": 0.00, "CD": 0.00, "CE": 0.02, "CZ": 0.04,
    "CB": 0.02, "CG1": 0.00, "CG2": 0.00, "CD1": 0.00, "CD2": 0.00,
    "CE1": 0.00, "CE2": 0.00, "CE3": 0.00, "CZ2": 0.00, "CZ3": 0.00, "CH2": 0.00,
    "NH": 0.00, "OB": 0.00,
}
DONOR_ATOMS = {"N", "NZ", "NE", "NH1", "NH2", "ND1", "NE2", "ND2", "OG", "OG1", "OH", "SG"}
ACCEPTOR_ATOMS = {"O", "OXT", "OD1", "OD2", "OE1", "OE2", "OD2", "OH", "OG", "OG1", "SD", "ND1", "NE2"}


class Params4:
    lj_cut = 8.0
    hb_depth = 3.0
    hb_d0 = 2.95
    hb_sig = 0.55
    hb_cut = 4.2
    intra_mask_min_dres = 2
    n_steps = 150
    lr_rigid = 0.02
    lr_tors = 3e-4
    tors_max = 0.06
    max_disp = 0.15     # 单步最大原子位移（Å）
    k_teth = 0.5        # 构象系绳（防远离参考构象）
    soft_max = 0.9      # 软化核初始半径（Å，逐步退火到 0）


# ---------------- 原子化 ----------------
def build_atoms(relaxer):
    """返回 pos(n,3), el(n), charge(n), resof(n), names(n)"""
    pos, el, ch, resof, names = [], [], [], [], []
    for k, (ri, an) in enumerate(relaxer.idx):
        xyz, e = relaxer.residues[ri][2][an]
        pos.append(xyz)
        el.append("C" if e.upper().startswith("C") else
                  "N" if e.upper().startswith("N") else
                  "O" if e.upper().startswith("O") else
                  "S" if e.upper().startswith("S") else "C")
        q = CHARGE.get(an, 0.0)
        nm = relaxer.residues[ri][1]
        if an == "OE1" and nm == "GLN":
            q = -0.45
        ch.append(q)
        resof.append(ri)
        names.append(an)
    return (np.array(pos, dtype=float), np.array(el), np.array(ch),
            np.array(resof), np.array(names))


def receptor_atoms(residues):
    pos, el, ch, resof, names = [], [], [], [], []
    for ri, (seq, nm, at) in enumerate(residues):
        for an, (xyz, e) in at.items():
            pos.append(xyz)
            el.append("C" if e.upper().startswith("C") else
                      "N" if e.upper().startswith("N") else
                      "O" if e.upper().startswith("O") else
                      "S" if e.upper().startswith("S") else "C")
            q = CHARGE.get(an, 0.0)
            if an == "OE1" and nm == "GLN":
                q = -0.45
            ch.append(q)
            resof.append(ri)
            names.append(an)
    return (np.array(pos, dtype=float), np.array(el), np.array(ch),
            np.array(resof), np.array(names))


def _lj_sigma_eps(el_a, el_b):
    ra, ea = LJ.get(el_a, LJ["C"])
    rb, eb_ = LJ.get(el_b, LJ["C"])
    return ra + rb, float(np.sqrt(ea * eb_))


class AtomFF:
    """向量化原子力场（平滑切换函数消除硬截断，保证能量面可微）"""

    def __init__(self, rec):
        self.Rpos, self.Rel, self.Rch, self.Rresof, self.Rnames = rec
        m = len(self.Rpos)
        self.is_acc = np.array([n in ACCEPTOR_ATOMS for n in self.Rnames])
        self.acc_idx = np.where(self.is_acc)[0]
        # 受体侧 LJ 参数表
        self.r_sig = np.array([LJ.get(e, LJ["C"])[0] for e in self.Rel])
        self.r_eps = np.array([LJ.get(e, LJ["C"])[1] for e in self.Rel])

    def near(self, Ppos, cutoff=12.0):
        """返回仅含 Ppos 附近受体重原子的子力场（邻居列表加速）"""
        d = np.sqrt(((Ppos[:, None, :] - self.Rpos[None, :, :]) ** 2).sum(-1))
        keep = np.where((d < cutoff).any(0))[0]
        sub = (self.Rpos[keep], self.Rel[keep], self.Rch[keep], self.Rresof[keep], self.Rnames[keep])
        return AtomFF(sub)

    @staticmethod
    def _switch(r, rc):
        """平滑切换 (1-(r/rc)^2)^2 与其导数"""
        t = np.clip(r / rc, 0.0, 1.0)
        s = (1.0 - t * t) ** 2
        ds = np.where(r < rc, -4.0 * r * (1.0 - t * t) / (rc * rc), 0.0)
        return s, ds

    def energy_forces(self, Ppos, Pel, Pch, Presof, Pnames, soft=0.0):
        n = len(Ppos)
        F = np.zeros_like(Ppos)
        soft2 = soft * soft

        # ---------- 分子间 LJ + 静电（平滑切换 + 软化核）----------
        diff = Ppos[:, None, :] - self.Rpos[None, :, :]
        d_raw = np.sqrt(np.maximum((diff ** 2).sum(-1), 1e-8))
        d = np.sqrt(d_raw * d_raw + soft2)      # 软化半径：r_eff = sqrt(r²+δ²)
        jac = d_raw / d                          # d r_eff / d r
        sig = np.array([LJ.get(e, LJ["C"])[0] for e in Pel])[:, None] + self.r_sig[None, :]
        eps = np.sqrt(np.array([LJ.get(e, LJ["C"])[1] for e in Pel])[:, None] * self.r_eps[None, :])
        sr = sig / d
        sr6 = sr ** 6
        e_lj = eps * (sr6 * sr6 - 2.0 * sr6)
        g_lj = eps * (-12.0 * sr6 * sr6 + 12.0 * sr6) / d * jac
        qq = 332.0 * Pch[:, None] * self.Rch[None, :]
        e_el = qq / (4.0 * d ** 2)
        g_el = -2.0 * qq / (4.0 * d ** 3) * jac
        S, Sd = self._switch(d, Params4.lj_cut)
        E_lj = float((e_lj * S).sum())
        E_elec = float((e_el * S).sum())
        g_pair = (g_lj + g_el) * S + (e_lj + e_el) * Sd * jac
        F -= (g_pair[:, :, None] * diff / d_raw[:, :, None]).sum(1)

        # ---------- 肽内 LJ 自排斥（硬掩码但距离远离截断）----------
        diff2 = Ppos[:, None, :] - Ppos[None, :, :]
        dd = np.sqrt(np.maximum((diff2 ** 2).sum(-1), 1e-8))
        intra = (np.abs(Presof[:, None] - Presof[None, :]) >= Params4.intra_mask_min_dres)
        np.fill_diagonal(intra, False)
        sig2 = np.array([LJ.get(e, LJ["C"])[0] for e in Pel])[:, None] + \
            np.array([LJ.get(e, LJ["C"])[0] for e in Pel])[None, :]
        eps2 = np.sqrt(np.array([LJ.get(e, LJ["C"])[1] for e in Pel])[:, None] *
                       np.array([LJ.get(e, LJ["C"])[1] for e in Pel])[None, :])
        sr6_i = (sig2 / dd) ** 6
        e_i = eps2 * (sr6_i * sr6_i - 2.0 * sr6_i)
        g_i = eps2 * (-12.0 * sr6_i * sr6_i + 12.0 * sr6_i) / dd
        S2, Sd2 = self._switch(dd, 6.0)
        E_intra = float((e_i * S2 * intra).sum()) * 0.5
        Fi = (g_i * S2 + e_i * Sd2) * intra
        F -= (Fi[:, :, None] * diff2 / dd[:, :, None]).sum(1) * 0.5

        # ---------- 方向性氢键（平滑）----------
        E_hb = 0.0
        don_idx = [i for i in range(n) if Pnames[i] in DONOR_ATOMS]
        if don_idx and len(self.acc_idx):
            uh = np.zeros((len(don_idx), 3))
            for k, i in enumerate(don_idx):
                u = self._h_dir(Ppos, Pnames, Presof, i)
                uh[k] = u if u is not None else 0.0
            Pi = Ppos[don_idx]
            dv = self.Rpos[self.acc_idx][None, :, :] - Pi[:, None, :]
            rr = np.sqrt(np.maximum((dv ** 2).sum(-1), 1e-8))
            dhat = dv / rr[:, :, None]
            dot = (uh[:, None, :] * dhat).sum(-1)
            ang = np.clip(dot, 0.0, 1.0) ** 2          # 平滑方向因子
            dr = (rr - Params4.hb_d0) / Params4.hb_sig
            env = np.exp(-dr * dr)
            Sh, Shd = self._switch(rr, Params4.hb_cut)
            w = ang * env * Sh
            E_hb = float((-Params4.hb_depth * w).sum())
            # dE/dr（忽略 dAng/dx 与 dSh 交叉项的高阶贡献，保留主项）
            g_hb = -Params4.hb_depth * ang * (env * (-2.0 * dr / Params4.hb_sig) * Sh + env * Shd)
            Fh = (g_hb[:, :, None] * dhat).sum(1)
            np.add.at(F, np.array(don_idx), -Fh)
        return {"lj": E_lj, "elec": E_elec, "hb": E_hb, "intra": E_intra}, F

    def _h_dir(self, Ppos, Pnames, Presof, i):
        """供体 H 方向估计：N 原子用邻居 (前残基C, CA) 的二分角反向；其它用重原子外指"""
        ri = Presof[i]
        an = Pnames[i]
        same = np.where(Presof == ri)[0]
        pos_i = Ppos[i]
        if an == "N":
            ca = [k for k in same if Pnames[k] == "CA"]
            cprev = [k for k in np.where(Presof == ri - 1)[0] if Pnames[k] == "C"]
            if ca and cprev:
                d1 = Ppos[cprev[0]] - pos_i
                d2 = Ppos[ca[0]] - pos_i
                s = d1 / (np.linalg.norm(d1) + 1e-9) + d2 / (np.linalg.norm(d2) + 1e-9)
                n_ = np.linalg.norm(s)
                return -s / n_ if n_ > 1e-6 else None
            if ca:
                v = pos_i - Ppos[ca[0]]
                n_ = np.linalg.norm(v)
                return v / n_ if n_ > 1e-6 else None
            return None
        # 侧链供体：沿"从残基质心向外"方向
        cen = Ppos[same].mean(0)
        v = pos_i - cen
        n_ = np.linalg.norm(v)
        return v / n_ if n_ > 1e-6 else None


def minimize_atomic(ff, relaxer, n_steps=None, xref=None, verbose=False):
    """刚体 + 主链扭角的原子力场极小化（能量回溯线搜索：降能才接受，否则回滚减半步长）"""
    n_steps = n_steps or Params4.n_steps
    lr_t, lr_r = 1.0, 1.0          # 平移/旋转步长缩放因子（自适应）
    hist = []
    cur = None
    for step in range(n_steps):
        # 软化核退火：前 40% 步数逐渐从 0.9 Å 收敛到 0（先脱离重叠再精修）
        frac = step / max(n_steps - 1, 1)
        soft = Params4.soft_max * max(0.0, 1.0 - frac / 0.4) if Params4.soft_max > 0 else 0.0
        Ppos, Pel, Pch, Presof, Pnames = build_atoms(relaxer)
        E, F = ff.energy_forces(Ppos, Pel, Pch, Presof, Pnames, soft=soft)
        tot = E["lj"] + E["elec"] + E["hb"] + E["intra"]
        if xref is not None:
            tot += 0.5 * Params4.k_teth * float(((Ppos - xref) ** 2).sum())
            F = F - Params4.k_teth * (Ppos - xref)
        hist.append(tot)
        cur = tot
        # 刚度上限：位移步长与总力大小解耦（平坦面上限制单步位移）
        Fn = F.sum(0)
        fnorm = float(np.linalg.norm(Fn))
        cen = Ppos.mean(0)
        tau = (np.cross(Ppos - cen, F)).sum(0)
        tnorm = float(np.linalg.norm(tau))
        move = (Fn / fnorm * min(0.05, 0.004 * fnorm)) if fnorm > 1e-9 else np.zeros(3)
        ang = min(0.004, 0.0002 * tnorm)

        state = (relaxer.X().copy(), [(np.array(relaxer.residues[ri][2][an][0]),) for ri, an in relaxer.idx])
        # 试探性位移：刚体平移 + 旋转 + 扭角
        relaxer.rigid_all(np.eye(3), move * lr_t)
        if tnorm > 1e-9:
            ax = tau / tnorm
            th = ang * lr_r
            c_, s_ = np.cos(th), np.sin(th)
            K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
            R = np.eye(3) * c_ + s_ * K + (1 - c_) * np.outer(ax, ax)
            c2 = relaxer.X().mean(0)
            relaxer.rigid_all(R, c2 - R @ c2)
        for kind, ri, mask in relaxer.tors:
            pivot, u = relaxer._axis(kind, ri)
            if u is None:
                continue
            X = relaxer.X()
            r = X[mask] - pivot
            tq = float((np.cross(r, F[mask]) @ u).sum())
            if abs(tq) < 1e-9:
                continue
            delta = float(np.clip(Params4.lr_tors * tq, -Params4.tors_max, Params4.tors_max)) * lr_r
            relaxer.rotate(kind, ri, delta)

        P2 = build_atoms(relaxer)
        E2, _ = ff.energy_forces(*P2, soft=soft)
        tot2 = E2["lj"] + E2["elec"] + E2["hb"] + E2["intra"]
        if xref is not None:
            tot2 += 0.5 * Params4.k_teth * float(((P2[0] - xref) ** 2).sum())
        if tot2 <= cur:                      # 接受 → 适度放大步长
            lr_t = min(1.0, lr_t * 1.15)
            lr_r = min(1.0, lr_r * 1.15)
        else:                                # 拒绝 → 回滚并缩小
            for k, (ri, an) in enumerate(relaxer.idx):
                relaxer.residues[ri][2][an][0][:] = state[1][k][0]
            lr_t = max(0.02, lr_t * 0.5)
            lr_r = max(0.02, lr_r * 0.5)
        if verbose and step % 25 == 0:
            print(f"    step {step:3d}: E={hist[-1]:9.2f} (lr_t={lr_t:.2f} lr_r={lr_r:.2f}) "
                  f"lj {E['lj']:7.1f} el {E['elec']:7.1f} hb {E['hb']:6.1f} intra {E['intra']:6.1f}")
    return hist
