# -*- coding: utf-8 -*-
"""EPT-Dock v1 核心库（在 entrobill_proto 基础上扩展）

新增：
  * 标定参数接入（data/calib/calib.json；缺失项回退文献/占位值并如实标注）
  * MatcherV1：Morse + 通用 vdW + 位阻墙 + 链可达 代价，附解析力（dE/dd）
  * TorsionRelaxer：主链 φ/ψ 扭角弛豫（罗德里格斯旋转 + 场点力→原子力→扭矩下降）
  * dock_flex：Sinkhorn-EM（阻尼刚体 M 步 + 刚体排斥 + 扭角弛豫交替；场点逐迭代重建，
    方向帧自动同步——修复 v0 的 U1 同步与"平衡位置 targets"问题）
  * sinkhorn_div：去偏 Sinkhorn 散度场指纹距离（支柱3 反筛）
"""
import os
import json

import numpy as np

from entrobill_proto import (
    parse_pdb, _norm, dihedral, backbone_torsions, weighted_kabsch, kabsch_rmsd,
    residue_field_points, chain_field_points, surface_filter, chain_neighbors,
    reach_L, classify_torsion, torsion_class, type_cost,
    COMPAT, ANGULAR, HYDRO_ATOMS, PRIOR as PRIOR_V0, trans_matrix, RT,
)

HERE = os.path.dirname(os.path.abspath(__file__))
CAL_PATH = os.path.abspath(os.path.join(HERE, "..", "data", "calib", "calib.json"))


def load_calib():
    if os.path.exists(CAL_PATH):
        try:
            with open(CAL_PATH) as fh:
                return json.load(fh)
        except Exception:
            return {}
    return {}


CAL = load_calib()


# ---------------- 参数（标定接入 + 回退） ----------------
class P1:
    w_c = 0.6
    eps = 2.0          # Sinkhorn 终值
    eps_max = 400.0    # 退火起点
    tau_u = 1.0
    tau_q = 0.05
    n_sink = 30
    n_em = 40
    eta_att = 0.75
    topk = 10
    ATTR_MIN = 2.0
    WALL = 100.0
    WALL_D0 = 3.0
    w_local = 0.15
    loc_d0 = 8.0
    vdw_d = 4.6
    vdw_sig = 1.0
    vdw_epw = 4.0
    rep_d0 = 2.9
    rep_eta = 0.6
    rep_steps = 8
    tors_steps = 8
    tors_lr = 1e-4
    tors_max = 0.08    # 单步最大扭角变化（rad）
    # 文献回退值 (d*, σ, 阱深)
    MORSE_FALLBACK = {
        ("donor", "acceptor"): (2.9, 0.45, 45.0),
        ("acceptor", "donor"): (2.9, 0.45, 45.0),
        ("donor", "neg"): (3.0, 0.45, 45.0),
        ("neg", "donor"): (3.0, 0.45, 45.0),
        ("pos", "acceptor"): (3.4, 0.60, 50.0),
        ("acceptor", "pos"): (3.4, 0.60, 50.0),
        ("pos", "neg"): (4.3, 0.75, 60.0),
        ("neg", "pos"): (4.3, 0.75, 60.0),
        ("hydro", "hydro"): (5.5, 1.10, 20.0),
    }
    DEPTH = {("donor", "acceptor"): 45.0, ("acceptor", "donor"): 45.0,
             ("donor", "neg"): 45.0, ("neg", "donor"): 45.0,
             ("pos", "acceptor"): 50.0, ("acceptor", "pos"): 50.0,
             ("pos", "neg"): 60.0, ("neg", "pos"): 60.0,
             ("hydro", "hydro"): 20.0}


def _cal_pair(a, b):
    """从标定数据取 (d*, σ)；供体对用氢键窗口 mode_hb，其余用全窗 mode"""
    e = CAL.get("contacts", {}).get(a + "|" + b)
    if not e:
        return None
    if "donor" in (a, b):
        m, n = e.get("mode_hb"), e.get("n_hb", 0)
        if m is not None and n >= 10 and 2.5 <= m <= 4.5:
            return float(m), 0.45
        return None
    m = e.get("mode")
    if m is not None and e.get("n", 0) >= 40 and 2.5 <= m <= 6.5:
        p25, p75 = e.get("p25", m), e.get("p75", m)
        sig = float(np.clip(max(0.45, (p75 - p25) / 1.35, 0.2 * m), 0.45, 1.5))
        return float(m), sig
    return None


def _build_morse():
    out = {}
    for (a, b), (d0, s0, w0) in P1.MORSE_FALLBACK.items():
        c = _cal_pair(a, b)
        out[(a, b)] = (c[0], c[1], w0) if c else (d0, s0, w0)
    return out


P1.MORSE = _build_morse()
_nc = CAL.get("contacts", {}).get("noncompat")
if _nc and _nc.get("n", 0) >= 50 and _nc.get("p25") is not None:
    P1.vdw_d = float(np.clip(_nc["p25"], 3.8, 5.5))

# ---------------- 熵模型（标定版；回退占位） ----------------
STATES = ["aR", "beta", "PPII", "gamma", "aL", "delta"]
_ent = CAL.get("entropy", {})
CALIB_ENTROPY = bool(_ent.get("prior") and _ent.get("trans"))
if CALIB_ENTROPY:
    PRIOR_C = {}
    for c in ("std", "GLY", "PRO"):
        p = np.array([_ent["prior"].get(c, {}).get(s, 0.0) for s in STATES])
        p = np.maximum(p, 1e-3)
        PRIOR_C[c] = p / p.sum()
    T_C = np.maximum(np.array(_ent["trans"], dtype=float), 1e-3)
    T_C = T_C / T_C.sum(1, keepdims=True)


def entropy_bills_v1(states, classes):
    """前向×后向边缘化（标定版：coil 库先验 + 相邻转移矩阵；缺标定时回退占位）"""
    n, S = len(states), len(STATES)
    sidx = {s: k for k, s in enumerate(STATES)}
    f = np.zeros((n, S))
    b = np.zeros((n, S))
    if CALIB_ENTROPY:
        p0 = PRIOR_C[classes[0]]
        TT = lambda i: T_C
    else:
        p0 = np.array([PRIOR_V0[classes[0]][s] for s in STATES])
        TT = lambda i: trans_matrix(classes[i])
    f[0] = p0
    for i in range(1, n):
        T = TT(i)
        pi = PRIOR_C[classes[i]] if CALIB_ENTROPY else np.array([PRIOR_V0[classes[i]][s] for s in STATES])
        f[i] = pi * (T.T @ f[i - 1])
    Z = f[-1].sum()
    b[-1] = 1.0
    for i in range(n - 2, -1, -1):
        T = TT(i + 1)
        pi1 = PRIOR_C[classes[i + 1]] if CALIB_ENTROPY else np.array([PRIOR_V0[classes[i + 1]][s] for s in STATES])
        b[i] = T @ (pi1 * b[i + 1])
    post = f * b / Z
    bills = np.zeros(n)
    obs = np.array([sidx[s] if s is not None else -1 for s in states])
    for i in range(n):
        p = post[i, obs[i]] if obs[i] >= 0 else post[i].mean()
        bills[i] = -RT * np.log(max(p, 1e-12))
    joint_logp = np.log(max(p0[obs[0]], 1e-12))
    for i in range(1, n):
        T = TT(i)
        joint_logp += np.log(max(T[obs[i - 1], obs[i]], 1e-12))
    return bills, float(joint_logp), post


# ---------------- MatcherV1 ----------------
class MatcherV1:
    def __init__(self, pep_pts, pk_pts, n_res, pk_atoms=None):
        self.pk_atoms = pk_atoms
        self.pt = [p["t"] for p in pep_pts]
        self.pkt = [p["t"] for p in pk_pts]
        self.Y = np.array([p["x"] for p in pk_pts])
        self.U2 = np.array([p["u"] if p["u"] is not None else np.zeros(3) for p in pk_pts])
        self.p_res = np.array([p["r"] for p in pep_pts])
        self.n_res = n_res
        n, m = len(pep_pts), len(pk_pts)
        self.compat = np.array([[(a, b) in COMPAT for b in self.pkt] for a in self.pt])
        self.dst = np.zeros((n, m))
        self.sig = np.ones((n, m))
        self.epw = np.zeros((n, m))
        for i, a in enumerate(self.pt):
            for j, b in enumerate(self.pkt):
                if (a, b) in P1.MORSE:
                    self.dst[i, j], self.sig[i, j], self.epw[i, j] = P1.MORSE[(a, b)]
        self.U1_0 = np.array([p["u"] if p["u"] is not None else np.zeros(3) for p in pep_pts])
        has1 = np.array([p["u"] is not None for p in pep_pts])
        has2 = np.array([p["u"] is not None for p in pk_pts])
        self.dir_pair = np.array([[((a, b) in ANGULAR) for b in self.pkt] for a in self.pt])
        self.dir_mask = has1[:, None] & has2[None, :] & self.dir_pair & self.compat
        self.nbrs = [chain_neighbors(r, n_res) for r in range(n_res)]
        self.res_idx = [np.where(self.p_res == r)[0] for r in range(n_res)]

    def pair_energy(self, X, U1_cur):
        d = np.sqrt(((X[:, None, :] - self.Y[None, :, :]) ** 2).sum(-1))
        dd = (d - self.dst) / self.sig
        morse = self.epw * ((1.0 - np.exp(-dd)) ** 2 - 1.0)
        ddv = (d - P1.vdw_d) / P1.vdw_sig
        vdw = P1.vdw_epw * ((1.0 - np.exp(-ddv)) ** 2 - 1.0)
        wall = P1.WALL * np.maximum(0.0, P1.WALL_D0 - d) ** 2
        E = np.where(self.compat, morse, vdw + wall)
        E = E + P1.w_local * np.maximum(0.0, d - P1.loc_d0) ** 2
        dot = U1_cur @ self.U2.T
        dirfac = np.where(self.dir_mask,
                          0.35 + 0.65 * np.clip(-dot, 0.0, 1.0), 1.0)
        return np.where(self.dir_mask, dirfac * E, E)

    def cost(self, X, U1_cur, targets=None):
        E = self.pair_energy(X, U1_cur)
        if targets is not None:
            dists = np.linalg.norm(self.Y[:, None, :] - targets[None, :, :], axis=2)
            for r in range(self.n_res):
                pen = np.zeros(len(self.Y))
                for j in self.nbrs[r]:
                    pen += np.maximum(0.0, dists[:, j] - reach_L(abs(r - j))) ** 2
                E[self.res_idx[r]] += P1.w_c * pen
        return E

    def sinkhorn(self, C, eps):
        K = np.exp(-C / eps)
        k = min(P1.topk, C.shape[1])
        idx = np.argpartition(C, k - 1, axis=1)[:, :k]
        mask = np.zeros_like(K, dtype=bool)
        np.put_along_axis(mask, idx, True, axis=1)
        K = K * (mask & (C < -P1.ATTR_MIN))
        empty = K.sum(1) <= 0
        if empty.any():
            idx0 = np.argmin(C, axis=1)
            K[empty, idx0[empty]] = 1.0
        n, m = C.shape
        a = np.full(n, 1.0 / n)
        b = np.full(m, 1.0 / m)
        v = np.ones(m)
        for _ in range(P1.n_sink):
            u = (a / np.maximum(K @ v, 1e-300)) ** P1.tau_u
            v = (b / np.maximum(K.T @ u, 1e-300)) ** P1.tau_q
        return u[:, None] * K * v[None, :]

    def point_targets(self, X, P):
        """平衡位置 targets：T = y_p + d_pref·unit(x_i − y_p)"""
        d = np.sqrt(((X[:, None, :] - self.Y[None, :, :]) ** 2).sum(-1))
        d_pref = np.where(self.compat, self.dst, P1.vdw_d)
        u = (X[:, None, :] - self.Y[None, :, :]) / np.maximum(d, 1e-6)[:, :, None]
        T = self.Y[None, :, :] + d_pref[:, :, None] * u
        Pn = P / (P.sum(1, keepdims=True) + 1e-12)
        return (Pn[:, :, None] * T).sum(1)

    def score(self, Xp, U1_cur, Xa):
        attract = float(self.pair_energy(Xp, U1_cur).min(1).mean())
        rep = 0.0
        if self.pk_atoms is not None:
            d = np.sqrt(((Xa[:, None, :] - self.pk_atoms[None, :, :]) ** 2).sum(-1))
            rep = 40.0 * float((np.maximum(0.0, 2.7 - d) ** 2).sum())
        return attract + rep

    def dEdd(self, X, U1_cur):
        """逐对 dE/dd（解析；忽略方向因子导数）"""
        d = np.sqrt(((X[:, None, :] - self.Y[None, :, :]) ** 2).sum(-1))
        dd = (d - self.dst) / self.sig
        g_morse = np.where(self.compat,
                           2.0 * self.epw * (1.0 - np.exp(-dd)) * np.exp(-dd) / self.sig, 0.0)
        ddv = (d - P1.vdw_d) / P1.vdw_sig
        g_vdw = np.where(~self.compat,
                         2.0 * P1.vdw_epw * (1.0 - np.exp(-ddv)) * np.exp(-ddv) / P1.vdw_sig, 0.0)
        g_wall = np.where((~self.compat) & (d < P1.WALL_D0),
                          -2.0 * P1.WALL * (P1.WALL_D0 - d), 0.0)
        g_loc = np.where(d > P1.loc_d0, 2.0 * P1.w_local * (d - P1.loc_d0), 0.0)
        return g_morse + g_vdw + g_wall + g_loc, d

    def forces_on_pts(self, X, U1_cur, P=None):
        """场点下降力；P（输运计划）给定时按计划质量加权（n·P：好行满力、坏行自弱）"""
        g, d = self.dEdd(X, U1_cur)
        uvec = (X[:, None, :] - self.Y[None, :, :]) / np.maximum(d, 1e-6)[:, :, None]
        if P is None:
            F = -(g[:, :, None] * uvec).sum(1)
        else:
            F = -(len(X) * P[:, :, None] * g[:, :, None] * uvec).sum(1)
        fn = np.linalg.norm(F, axis=1)
        scale = np.minimum(1.0, 10.0 / np.maximum(fn, 1e-9))  # 力截断
        return F * scale[:, None]


# ---------------- 扭角弛豫 ----------------
def mutable_residues(residues):
    return [(seq, nm, {an: [np.array(xyz), el] for an, (xyz, el) in at.items()})
            for (seq, nm, at) in residues]


class TorsionRelaxer:
    """主链 φ/ψ 扭角空间中的肽坐标（可变；场点由坐标现场重建，方向帧自动同步）"""

    def __init__(self, residues):
        self.residues = residues
        self.idx = []
        for ri, (seq, nm, at) in enumerate(residues):
            for an, (xyz, el) in at.items():
                self.idx.append((ri, an))
        res_of = np.array([ri for ri, _ in self.idx])
        anames = np.array([an for _, an in self.idx])
        self.tors = []
        for ri in range(len(residues)):
            seq, nm, at = residues[ri]
            has_prev = ri > 0 and residues[ri - 1][0] == seq - 1
            has_next = ri < len(residues) - 1 and residues[ri + 1][0] == seq + 1
            if has_prev and "N" in at and "CA" in at:
                self.tors.append(("phi", ri, res_of >= ri))
            if has_next and "CA" in at and "C" in at:
                m = (res_of > ri) | ((res_of == ri) & np.isin(anames, ["C", "O", "OXT"]))
                self.tors.append(("psi", ri, m))
        self.res_of = res_of

    def X(self):
        return np.array([self.residues[ri][2][an][0] for (ri, an) in self.idx])

    def set_X(self, Xnew):
        for k, (ri, an) in enumerate(self.idx):
            self.residues[ri][2][an][0][:] = Xnew[k]

    def rigid_all(self, R, t):
        self.set_X(self.X() @ R.T + t)

    def _axis(self, kind, ri):
        at = self.residues[ri][2]
        if kind == "phi":
            pivot, u = at["N"][0], at["CA"][0] - at["N"][0]
        else:
            pivot, u = at["CA"][0], at["C"][0] - at["CA"][0]
        nu = np.linalg.norm(u)
        return np.asarray(pivot, dtype=float), (u / nu if nu > 1e-8 else None)

    def rotate(self, kind, ri, delta):
        mask = next(m for (k, r, m) in self.tors if k == kind and r == ri)
        pivot, u = self._axis(kind, ri)
        if u is None or abs(delta) < 1e-9:
            return
        X = self.X()
        v = X[mask] - pivot
        c, s = np.cos(delta), np.sin(delta)
        vr = v * c + np.cross(u, v) * s + np.outer(v @ u, u) * (1 - c)
        X[mask] = pivot + vr
        self.set_X(X)

    def torsion_scramble(self, rng, sig_deg=25.0):
        for kind, ri, _ in self.tors:
            self.rotate(kind, ri, np.radians(rng.normal(0, sig_deg)))

    def ca_coords(self):
        out, ris = [], []
        for ri, (seq, nm, at) in enumerate(self.residues):
            if "CA" in at:
                out.append(at["CA"][0])
                ris.append(ri)
        return np.array(out), np.array(ris)


def build_pt2atom(relaxer, pts):
    m = {(ri, an): k for k, (ri, an) in enumerate(relaxer.idx)}
    out = []
    for p in pts:
        ri, label = p["r"], p["label"]
        if (ri, label) in m:
            out.append(m[(ri, label)])
        elif label.endswith("-sc"):
            an = HYDRO_ATOMS.get(p["rname"], ["CB"])[0]
            out.append(m.get((ri, an), m.get((ri, "CB"), 0)))
        elif label == "ARG-guan":
            out.append(m.get((ri, "NE"), m.get((ri, "CZ"), 0)))
        else:
            out.append(m.get((ri, "CB"), 0))
    return np.array(out)


def _pts_arrays(relaxer):
    pts = chain_field_points(relaxer.residues)
    Xp = np.array([p["x"] for p in pts])
    U1 = np.array([p["u"] if p["u"] is not None else np.zeros(3) for p in pts])
    return pts, Xp, U1


def torsion_steps(M, relaxer, P=None, n_steps=None, lr=None, xref=None, k_teth=1.0):
    n_steps = n_steps or P1.tors_steps
    lr = lr or P1.tors_lr
    # 链内自排斥掩码（|Δres| ≥ 2 的原子对）
    res_of = relaxer.res_of
    self_mask = np.abs(res_of[:, None] - res_of[None, :]) >= 2
    np.fill_diagonal(self_mask, False)
    for _ in range(n_steps):
        pts, Xp, U1 = _pts_arrays(relaxer)
        pt2atom = build_pt2atom(relaxer, pts)  # 每步重建映射，防场点数变化错位
        Fp = M.forces_on_pts(Xp, U1, P)
        Xa = relaxer.X()
        Fa = np.zeros_like(Xa)
        np.add.at(Fa, pt2atom, Fp)
        if xref is not None:  # 构象系绳（= 构象熵先验的搜索内体现：偏离参考构象受罚）
            Fa += k_teth * (xref - Xa)
        if M.pk_atoms is not None:  # 受体原子软球排斥
            d = np.sqrt(((Xa[:, None, :] - M.pk_atoms[None, :, :]) ** 2).sum(-1))
            close = d < 2.7
            diff = Xa[:, None, :] - M.pk_atoms[None, :, :]
            Fa += (np.where(close, 80.0 * (2.7 - d) / np.maximum(d, 1e-6), 0.0)[:, :, None] * diff).sum(1)
        d2 = np.sqrt(((Xa[:, None, :] - Xa[None, :, :]) ** 2).sum(-1))  # 链内自排斥
        close2 = (d2 < 3.2) & self_mask
        if close2.any():
            diff2 = Xa[:, None, :] - Xa[None, :, :]
            Fa += (np.where(close2, 80.0 * (3.2 - d2) / np.maximum(d2, 1e-6), 0.0)[:, :, None] * diff2).sum(1)
        for kind, ri, mask in relaxer.tors:
            pivot, u = relaxer._axis(kind, ri)
            if u is None:
                continue
            Xa = relaxer.X()
            r = Xa[mask] - pivot
            tau = float((np.cross(r, Fa[mask]) @ u).sum())
            delta = float(np.clip(lr * tau, -P1.tors_max, P1.tors_max))
            if abs(delta) > 1e-6:
                relaxer.rotate(kind, ri, delta)


def rigid_repulse(M, relaxer):
    if M.pk_atoms is None:
        return
    for _ in range(P1.rep_steps):
        Xa = relaxer.X()
        d = np.linalg.norm(Xa[:, None, :] - M.pk_atoms[None, :, :], axis=2)
        close = d < P1.rep_d0
        if not close.any():
            break
        diff = Xa[:, None, :] - M.pk_atoms[None, :, :]
        w = np.where(close, (P1.rep_d0 - d) / np.maximum(d, 1e-6), 0.0)
        disp = (w[:, :, None] * diff).sum(1)
        R, t = weighted_kabsch(Xa, Xa + P1.rep_eta * disp, np.ones(len(Xa)))
        relaxer.rigid_all(R, t)


def dock_flex(M, relaxer, anneal=True, flex=True, track=None, xref=None, k_teth=1.0):
    """Sinkhorn-EM：E 步带链可行约束的稀疏 Sinkhorn；M 步 = 阻尼刚体 + 刚体排斥 + 扭角弛豫
    xref: 构象系绳参考坐标（柔性搜索不偏离起始构象系综）"""
    targets = None
    hist = []
    n_res = M.n_res
    Pn = wres = None
    for it in range(P1.n_em):
        eps = P1.eps
        if anneal:
            frac = it / max(P1.n_em - 1, 1)
            eps = P1.eps_max * (P1.eps / P1.eps_max) ** frac
        _, Xp, U1 = _pts_arrays(relaxer)
        C = M.cost(Xp, U1, targets)
        P = M.sinkhorn(C, eps)
        Pagg = np.zeros((n_res, len(M.Y)))
        for r, ix in enumerate(M.res_idx):
            if len(ix):
                Pagg[r] = P[ix].sum(0)
        wres = Pagg.sum(1)
        Pn = Pagg / (Pagg.sum(1, keepdims=True) + 1e-12)
        tgt_pts = M.point_targets(Xp, P)
        reps = np.array([Xp[ix].mean(0) if len(ix) else np.zeros(3) for ix in M.res_idx])
        targets = np.array([tgt_pts[ix].mean(0) if len(ix) else np.zeros(3) for ix in M.res_idx])
        ok = wres > 1e-6
        targets[~ok] = reps[~ok]
        R, t = weighted_kabsch(reps, targets, np.maximum(wres, 1e-9))
        relaxer.rigid_all((1 - P1.eta_att) * np.eye(3) + P1.eta_att * R, P1.eta_att * t)
        rigid_repulse(M, relaxer)
        if flex and (not anneal or eps <= 8.0):
            # 柔性只在锐化相启用：退火早期计划近乎均匀，满力吸引会令链屈曲坍缩
            torsion_steps(M, relaxer, P, xref=xref, k_teth=k_teth)
        if track is not None:
            track(it, relaxer)
        _, Xp, U1 = _pts_arrays(relaxer)
        hist.append(M.score(Xp, U1, relaxer.X()))
    return Pn, wres, hist


# ---------------- 支柱3：场指纹 Sinkhorn 散度 ----------------
SEED_TYPES = ("donor", "acceptor", "pos", "neg", "hydro")


def local_descriptors(pts, bins=((0.0, 6.0), (6.0, 12.0)), excl_adj=False):
    """位姿无关的场点局部环境描述子：分类型邻居计数直方图（L1 归一化）"""
    X = np.array([p["x"] for p in pts])
    R = np.array([p["r"] for p in pts])
    T = np.array([p["t"] for p in pts])
    d = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    resdiff = np.abs(R[:, None] - R[None, :])
    valid = d > 0.5
    valid &= resdiff > 0            # 排除同残基
    if excl_adj:
        valid &= resdiff > 1        # 肽侧再排除相邻残基（链本身主导邻域）
    D = np.zeros((len(pts), len(SEED_TYPES) * len(bins)))
    for bi, (r0, r1) in enumerate(bins):
        m = valid & (d >= r0) & (d < r1)
        for ti, t in enumerate(SEED_TYPES):
            D[:, bi * len(SEED_TYPES) + ti] = (m & (T[:, None] == t)).sum(1)
    return D / np.maximum(D.sum(1, keepdims=True), 1e-9)


def _sinkhorn_plain(C, eps, iters=120, tau_q=0.15):
    """肽侧硬边际 + 口袋侧松弛的非平衡 Sinkhorn（种子匹配用；
    平衡边际会强制每个口袋点接收质量 → 计划弥散、化学错配）"""
    K = np.exp(-C / eps)
    n, m = C.shape
    a = np.full(n, 1.0 / n)
    b = np.full(m, 1.0 / m)
    v = np.ones(m)
    for _ in range(iters):
        u = a / np.maximum(K @ v, 1e-300)
        v = (b / np.maximum(K.T @ u, 1e-300)) ** tau_q
    return u[:, None] * K * v[None, :]


def soft_procrustes(X, Y, P):
    """软对应 Procrustes：min_{R,t} Σ_ij P_ij ||R x_i + t − y_j||² 的闭式解"""
    a = P.sum(1)
    b = P.sum(0)
    W = P.sum()
    xbar = (a @ X) / W
    ybar = (b @ Y) / W
    H = (X - xbar).T @ P @ (Y - ybar)
    U, S, Vt = np.linalg.svd(H)
    dsign = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, dsign]) @ U.T
    return R, ybar - R @ xbar


def seed_pose(M, relaxer, pk_pts, eps_seed=2.0, alpha=5.0, n_rounds=3, verbose=False,
              bins=None, excl_adj=True, sub=1.0, rng=None):
    """OT 计划 → 软对应 Procrustes 确定性姿态种子：
    第 1 轮用位姿无关的局部描述子代价匹配（类型代价全权重，化学优先）；
    后续轮用完整位姿相关代价（方向项进入）。
    bins/excl_adj/sub: 异构描述子配置（多种子多样化用）；sub<1 时对受体点下采样"""
    pts0, Xp0, U1_0 = _pts_arrays(relaxer)
    q_pts = pk_pts
    if sub < 1.0:                                  # 受体点子采样 → 不同种子
        r = rng if rng is not None else np.random.default_rng(0)
        k = max(30, int(len(pk_pts) * sub))
        sel = np.sort(r.choice(len(pk_pts), k, replace=False))
        q_pts = [pk_pts[i] for i in sel]
    kw = {"excl_adj": excl_adj}
    if bins is not None:
        kw["bins"] = bins
    Dp = local_descriptors(pts0, **kw)
    Dq = local_descriptors(q_pts, **kw)
    pkt = [p["t"] for p in q_pts]
    tc = np.array([[type_cost(a, b) for b in pkt] for a in M.pt])
    C = alpha * ((Dp[:, None, :] - Dq[None, :, :]) ** 2).sum(-1) + tc
    P = _sinkhorn_plain(C, eps_seed)
    Yr = np.array([p["x"] for p in q_pts])  # 第 1 轮目标（子采样/区域点集）
    hist = []
    for it in range(n_rounds):
        R, t = soft_procrustes(Xp0, Yr if it == 0 else M.Y, P)
        relaxer.rigid_all(R, t)
        _, Xp0, U1_0 = _pts_arrays(relaxer)
        C2 = M.cost(Xp0, U1_0, None)
        P = M.sinkhorn(C2, P1.eps)
        hist.append(float((P * np.where(C2 < -P1.ATTR_MIN, C2, 0.0)).sum()))
        if verbose:
            print(f"    seed round {it}: plan cost = {hist[-1]:.1f}")
    return hist


def gw_seed(M, relaxer, pk_pts, n_iter=6, alpha=1.0, beta=0.6, eps=2.0,
            use_desc=True, verbose=False):
    """Gromov-Wasserstein 种子：强制相对几何一致
    C_ij = β·Σ_kl (D1_ik − D2_jl)² P_kl    + 化学类型代价
    —— 描述子只看单点局部环境（丢失相对几何），GW 补上"肽内点距 ≈ 匹配口袋点距"
    用平方损失的 GW 线性化：C = (D1²@行边际)₁ᵀ − 2·D1·P·D2ᵀ + (D2²@列边际)ᵀ₂
    初值 P 由局部描述子匹配给出（α 权重），之后纯 GW 迭代 + 软 Procrustes"""
    pts0, Xp0, U1_0 = _pts_arrays(relaxer)
    Y = np.array([p["x"] for p in pk_pts])
    pkt = [p["t"] for p in pk_pts]
    D1 = np.sqrt(((Xp0[:, None, :] - Xp0[None, :, :]) ** 2).sum(-1))
    D2 = np.sqrt(((Y[:, None, :] - Y[None, :, :]) ** 2).sum(-1))
    D1, D2 = np.minimum(D1, 30.0), np.minimum(D2, 30.0)
    tc = np.array([[type_cost(a, b) for b in pkt] for a in M.pt])

    if use_desc:
        Dp = local_descriptors(pts0, excl_adj=True)
        Dq = local_descriptors(pk_pts)
        C0 = alpha * ((Dp[:, None, :] - Dq[None, :, :]) ** 2).sum(-1) + tc
        P = _sinkhorn_plain(C0, eps, tau_q=0.15)
    else:
        P = np.outer(np.full(len(Xp0), 1.0 / len(Xp0)), np.full(len(Y), 1.0 / len(Y)))

    hist = []
    for it in range(n_iter):
        r = P.sum(1)
        c = P.sum(0)
        gw = (D1 ** 2 @ r)[:, None] - 2.0 * (D1 @ P @ D2.T) + (D2 ** 2 @ c)[None, :]
        C = beta * gw + tc + (alpha * 0 if not use_desc else 0)
        P = _sinkhorn_plain(C, eps, tau_q=0.15)
        R, t = soft_procrustes(Xp0, Y, P)
        relaxer.rigid_all(R, t)
        _, Xp0, U1_0 = _pts_arrays(relaxer)
        hist.append(float((P * gw).sum()))
        if verbose:
            print(f"    gw iter {it}: GW cost = {hist[-1]:9.2f}")
    return hist


def detect_pockets(pk_pts, pk_atoms, kmax=6, rho=10.0, sep=9.0, rad=12.0, min_pts=12):
    """第 0 级口袋检测（确定性场点埋藏度聚类，设计文档 §3.1）：
    返回 [(center, [pocket_field_points])]，按埋藏度降序"""
    Y = np.array([p["x"] for p in pk_pts])
    d = np.sqrt(((Y[:, None, :] - pk_atoms[None, :, :]) ** 2).sum(-1))
    bury = (d < rho).sum(1).astype(float)
    order = np.argsort(-bury)
    chosen = []
    for i in order:
        if all(np.linalg.norm(Y[i] - Y[j]) > sep for j in chosen):
            chosen.append(i)
        if len(chosen) >= kmax:
            break
    out = []
    for ci in chosen:
        near = np.linalg.norm(Y - Y[ci], axis=1) < rad
        pts = [pk_pts[k] for k in np.where(near)[0]]
        if len(pts) >= min_pts:
            out.append((Y[ci], pts))
    return out


def pocket_seed(M, relaxer, pocket_pts, eps_seed=2.0, alpha=8.0, n_rounds=3, gw_extra=2):
    """凹穴内种子：描述子计划限定在凹穴点集内（强制局域化），随后 GW 项收相对几何"""
    v1_ok = seed_pose(M, relaxer, pocket_pts, eps_seed=eps_seed, alpha=alpha, n_rounds=n_rounds)
    if gw_extra:
        gw_seed(M, relaxer, pocket_pts, n_iter=gw_extra, alpha=0.0, beta=0.6, use_desc=False)
    return relaxer.X().copy()


def pocket_seed_score(M, relaxer, pk_pts, X_seed):
    """凹穴种子的场点锐化分数（排序用）"""
    relaxer_set = None
    _, Xp, U1 = _pts_arrays(relaxer)
    return M.score(Xp, U1, X_seed)


def detect_pockets_wrapper(pk_pts, pk_atoms):  # 兼容别名
    return detect_pockets(pk_pts, pk_atoms)


def anchor_vote_seeds(M, relaxer, pk_pts, n_cand=25, n_hyp=400, n_out=4,
                      eps_seed=2.0, n_rounds=2, verbose=False):
    """锚点假设投票种子（几何 hashing）：
    1) 描述子最独特的 3 个肽锚点 × 各自类型相容+描述子最近的 top-K 口袋候选
    2) 三点 Kabsch 生成刚性假设（几何由构造保证）
    3) 全链快速评估（含方向的 pair_energy 最小值均值），取 top 假设
    4) 每个胜出假设跑 n_rounds 种子-EM
    返回 [(X_seed, hyp_score)]"""
    from itertools import product
    pts0, Xp0, U1_0 = _pts_arrays(relaxer)
    Y = np.array([p["x"] for p in pk_pts])
    Dp = local_descriptors(pts0, excl_adj=True)
    Dq = local_descriptors(pk_pts)
    pkt = [p["t"] for p in pk_pts]

    # 锚点：描述子最独特（最优匹配描述子距离最小）且类型为疏水/电荷的点优先
    dmin = ((Dp[:, None, :] - Dq[None, :, :]) ** 2).sum(-1).min(1)
    pref = [i for i in np.argsort(dmin) if M.pt[i] in ("hydro", "pos", "neg")]
    anchors = pref[:3] if len(pref) >= 3 else list(np.argsort(dmin)[:3])
    # 各锚点的口袋候选
    cands = {}
    for a in anchors:
        ccost = np.array([type_cost(M.pt[a], b) for b in pkt])
        dcost = ((Dp[a][None, :] - Dq) ** 2).sum(-1)
        cands[a] = list(np.argsort(dcost + 0.02 * ccost)[:n_cand])
    # 假设枚举 + 快速评估
    a1, a2, a3 = anchors[0], anchors[1], anchors[2]
    Xa = np.array([Xp0[a1], Xp0[a2], Xp0[a3]])
    results = []
    seen_R = []
    cnt = 0
    for p1, p2, p3 in product(cands[a1], cands[a2], cands[a3]):
        if len({p1, p2, p3}) < 3 or cnt >= n_hyp:
            continue
        cnt += 1
        Yh = np.array([Y[p1], Y[p2], Y[p3]])
        R, t = soft_procrustes(Xa, Yh, np.ones((3, 3)) / 3.0)
        Xt = Xp0 @ R.T + t
        U1t = U1_0 @ R.T
        sc = float(M.pair_energy(Xt, U1t).min(1).mean())
        results.append((sc, R, t))
    results.sort(key=lambda x: x[0])
    # 非极大值抑制：假设间差异太小的合并
    out = []
    for sc, R, t in results:
        dup = False
        for _, R2, t2 in out:
            if np.linalg.norm(t - t2) < 4.0 and float(np.abs(np.trace(R @ R2.T) - 1)) < 0.6:
                dup = True
                break
        if not dup:
            out.append((sc, R, t))
        if len(out) >= n_out:
            break
    # 胜出假设 → 种子-EM 精化
    seeds = []
    for sc, R, t in out:
        relaxer.rigid_all(R, t)
        for _ in range(n_rounds):
            _, Xp, U1 = _pts_arrays(relaxer)
            C2 = M.cost(Xp, U1, None)
            P = M.sinkhorn(C2, eps_seed)
            R2, t2 = soft_procrustes(Xp, M.Y, P)
            relaxer.rigid_all(R2, t2)
        _, Xp, U1 = _pts_arrays(relaxer)
        sc2 = M.score(Xp, U1, relaxer.X())
        seeds.append((relaxer.X().copy(), sc2))
        if verbose:
            print(f"    hyp score {sc:.1f} → seed score {sc2:.1f}")
    seeds.sort(key=lambda x: x[1])
    return seeds


def _ot_eps(A, B, eps, iters):
    C = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    K = np.exp(-C / eps)
    n, m = C.shape
    a = np.full(n, 1.0 / n)
    b = np.full(m, 1.0 / m)
    v = np.ones(m)
    for _ in range(iters):
        u = a / np.maximum(K @ v, 1e-300)
        v = b / np.maximum(K.T @ u, 1e-300)
    P = u[:, None] * K * v[None, :]
    return float((P * C).sum())


def sinkhorn_div(Pf, Qf, eps=100.0, iters=200):
    """去偏 Sinkhorn 散度（Feydy 2019：正定，非度量）；Pf/Qf 为 (n,3)/(m,3) 点集"""
    return _ot_eps(Pf, Qf, eps, iters) - 0.5 * _ot_eps(Pf, Pf, eps, iters) \
        - 0.5 * _ot_eps(Qf, Qf, eps, iters)
