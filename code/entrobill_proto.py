# -*- coding: utf-8 -*-
"""EPT-Dock 最小原型核心库（v0.1，纯 numpy）

对应设计文档 v0.2 的三大支柱的最小可实现版本：
  支柱1  各向异性场点 + 链可达性(线性化)惩罚的 Sinkhorn 匹配 + Sinkhorn-EM（加权 Kabsch + 刚体排斥）
  支柱2  转移矩阵构象熵账单（前向×后向边缘化，逐残基 posterior）
  支柱3  （本原型暂不含反筛，场点指纹即其数据基础）

v0.1 关键升级（相对 v0）：
  * OT 代价从"纯二次吸引"升级为**类型特异相互作用势**：
      兼容对 → Morse 势（有最优接触距离与结合深度；方向性调制阱深）
      非兼容对 → 空间位阻墙（纯排斥）
    —— 修复"吸引项偏好把肽拍在蛋白表面"的错误能量面（match what you score 的正确实现）
  * 位姿相关的方向点积逐迭代更新（旋转肽时方向帧随之旋转）
  * M 步刚体更新加阻尼（η=0.75）+ 刚体排斥弛豫

诚实声明：
  * Ramachandran 先验/转移相关/侧链熵/囚禁项均为占位统计，待 PDB coil 库替换。
  * 势参数（阱深/宽度/墙高）为手调量级，未做系统校准。
"""
import numpy as np

RT = 0.0019872041 * 300.0  # kcal/mol @ 300K


# ---------------- 参数 ----------------
class Params:
    w_c = 0.6        # 链可达惩罚权重
    eps = 2.0        # Sinkhorn 熵正则终值（小 ε + top-k 掩码 → 势阱配对主导）
    eps_max = 400.0  # 退火起点
    topk = 10        # 稀疏对应：每个肽点只与能量最优的 top-k 口袋点竞争
    tau_u = 1.0      # 肽侧边际硬度（=1 近似硬约束）
    tau_q = 0.05     # 口袋侧边际松弛（接近自由——点云不强制接收质量）
    n_sink = 30
    n_em = 40
    eta_att = 0.75   # M 步刚体更新阻尼
    # Morse 势参数：type-pair → (d*, σ, εw)
    MORSE = {
        ("donor", "acceptor"): (3.0, 0.50, 45.0),
        ("acceptor", "donor"): (3.0, 0.50, 45.0),
        ("donor", "neg"): (3.0, 0.50, 45.0),
        ("neg", "donor"): (3.0, 0.50, 45.0),
        ("pos", "acceptor"): (3.4, 0.60, 50.0),
        ("acceptor", "pos"): (3.4, 0.60, 50.0),
        ("pos", "neg"): (4.3, 0.75, 60.0),
        ("neg", "pos"): (4.3, 0.75, 60.0),
        ("hydro", "hydro"): (5.5, 1.1, 20.0),
    }
    WALL = 100.0     # 位阻墙系数（非兼容对，d < 3.0 Å 才算真碰撞）
    WALL_D0 = 3.0
    w_local = 0.15   # 空间局部性项（Å⁻²；从 loc_d0 起才生效，只压制远处云质量）
    loc_d0 = 8.0     # 局部性项起始距离（Å）——接触区（<8Å）不受污染
    # 通用 vdW 堆积势（非兼容对）：接触距离按晶体统计标定
    vdw_d = 4.6
    vdw_sig = 1.0
    vdw_epw = 4.0
    ATTR_MIN = 2.0  # 计划准入阈值：增益 < 2 的候选不进入输运计划（E≈0 的候选不产生拉力）
    dir_perp = 0.35  # 方向垂直时阱深保留比例（反平行=1.0）
    # 排斥弛豫（原子级，EM 内）
    rep_d0 = 2.9
    rep_eta = 0.6
    rep_steps = 8
    # 打分中的原子级软球排斥
    lam_rep = 40.0


# ---------------- PDB 解析 ----------------
def parse_pdb(path):
    """解析 ATOM 记录 → {chain: [(resSeq, resname, {atom: (xyz, element)})]}（按 resSeq 排序）"""
    chains = {}
    with open(path) as fh:
        for line in fh:
            if not line.startswith("ATOM  "):
                continue
            if line[16] not in (" ", "A"):  # altloc
                continue
            atom = line[12:16].strip()
            resname = line[17:20].strip()
            ch = line[21]
            try:
                resseq = int(line[22:26])
                xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            except ValueError:
                continue
            element = (line[76:78].strip() or atom[0])
            if element.upper().startswith("H"):
                continue
            res = chains.setdefault(ch, {}).setdefault(resseq, {"name": resname, "atoms": {}})
            res["atoms"][atom] = (xyz, element)
    return {ch: [(s, m[s]["name"], m[s]["atoms"]) for s in sorted(m)] for ch, m in chains.items()}


# ---------------- 几何 ----------------
def _norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else None


def dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    return np.degrees(np.arctan2(np.dot(np.cross(b1n, v), w), np.dot(v, w)))


def backbone_torsions(residues):
    n = len(residues)
    phi, psi = [None] * n, [None] * n
    for i, (seq, name, a) in enumerate(residues):
        if all(k in a for k in ("N", "CA", "C")):
            if i > 0 and residues[i - 1][0] == seq - 1 and "C" in residues[i - 1][2]:
                phi[i] = dihedral(residues[i - 1][2]["C"][0], a["N"][0], a["CA"][0], a["C"][0])
            if i < n - 1 and residues[i + 1][0] == seq + 1 and "N" in residues[i + 1][2]:
                psi[i] = dihedral(a["N"][0], a["CA"][0], a["C"][0], residues[i + 1][2]["N"][0])
    return phi, psi


def weighted_kabsch(X, Y, W):
    W = np.maximum(W, 1e-9)
    cx = (W[:, None] * X).sum(0) / W.sum()
    cy = (W[:, None] * Y).sum(0) / W.sum()
    H = ((X - cx) * W[:, None]).T @ (Y - cy)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, cy - cx @ R.T


def kabsch_rmsd(P, Q):
    R, t = weighted_kabsch(P, Q, np.ones(len(P)))
    diff = P @ R.T + t - Q
    return float(np.sqrt((diff ** 2).sum(1).mean()))


def rand_se3(rng, center, rmin=6.0, rmax=18.0):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    d = rng.normal(size=3)
    d /= np.linalg.norm(d)
    return R, center + d * rng.uniform(rmin, rmax)


# ---------------- 支柱1：各向异性场点 ----------------
MILD = ({"donor", "pos"}, {"acceptor", "neg"}, {"donor", "hydro"}, {"acceptor", "hydro"})


def type_cost(t1, t2):
    """类型相容性代价（种子匹配用）：相容 0 / 同类相斥 30 / 温和 12 / 不相容 80"""
    if (t1, t2) in COMPAT:
        return 0.0
    if t1 == t2:
        return 30.0
    if frozenset((t1, t2)) in map(frozenset, MILD):
        return 12.0
    return 80.0


COMPAT = {("donor", "acceptor"), ("acceptor", "donor"), ("donor", "neg"), ("neg", "donor"),
          ("pos", "acceptor"), ("acceptor", "pos"), ("pos", "neg"), ("neg", "pos"),
          ("hydro", "hydro")}
# 需要方向调制的配对（给体 u=N→H 外指，受体 u=孤对方向外指；最优 H 键 = 反平行）
ANGULAR = {("donor", "acceptor"), ("acceptor", "donor"), ("donor", "neg"), ("neg", "donor"),
           ("pos", "acceptor"), ("acceptor", "pos"), ("pos", "neg"), ("neg", "pos")}
HYDRO_ATOMS = {
    "ALA": ["CB"], "VAL": ["CG1", "CG2"], "LEU": ["CD1", "CD2"], "ILE": ["CG1", "CG2", "CD1"],
    "PHE": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"], "MET": ["CG", "SD", "CE"],
    "TRP": ["CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    "TYR": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"], "PRO": ["CB", "CG", "CD"]}


def residue_field_points(resname, atoms, prev_atoms):
    """一个残基的场点列表 [(type, xyz, dir|None, weight, label)]"""
    pts, A = [], atoms

    def add(t, aname, u=None, w=1.0):
        if aname in A:
            pts.append((t, A[aname][0], u, w, aname))

    # 主链
    if "N" in A and resname != "PRO":
        ca = A.get("CA")
        u = None
        if ca is not None:
            if prev_atoms and "C" in prev_atoms:
                d1, d2 = _norm(prev_atoms["C"][0] - A["N"][0]), _norm(ca[0] - A["N"][0])
                if d1 is not None and d2 is not None:
                    s = d1 + d2
                    if np.linalg.norm(s) > 1e-6:
                        u = -s / np.linalg.norm(s)
            else:
                u = _norm(A["N"][0] - ca[0])
        if u is not None:
            pts.append(("donor", A["N"][0], u, 1.0, "N"))
    if "O" in A and "C" in A:
        pts.append(("acceptor", A["O"][0], _norm(A["O"][0] - A["C"][0]), 1.0, "O"))
    if "OXT" in A and "C" in A:
        pts.append(("neg", A["OXT"][0], _norm(A["OXT"][0] - A["C"][0]), 1.5, "OXT"))

    # 侧链
    if resname == "LYS" and "NZ" in A:
        pts.append(("pos", A["NZ"][0], _norm(A["NZ"][0] - A["CE"][0]) if "CE" in A else None, 1.5, "NZ"))
    if resname == "ARG" and all(k in A for k in ("CZ", "NE", "NH1", "NH2")):
        cen = (A["NE"][0] + A["NH1"][0] + A["NH2"][0]) / 3.0
        pts.append(("pos", cen, _norm(cen - A["CZ"][0]), 1.5, "ARG-guan"))
    if resname in ("ASP", "GLU"):
        c, o1, o2 = {"ASP": ("CG", "OD1", "OD2"), "GLU": ("CD", "OE1", "OE2")}[resname]
        for o in (o1, o2):
            if o in A and c in A:
                pts.append(("neg", A[o][0], _norm(A[o][0] - A[c][0]), 1.5, o))
    if resname == "ASN":
        add("acceptor", "OD1", _norm(A["OD1"][0] - A["CG"][0]) if "OD1" in A and "CG" in A else None)
        add("donor", "ND2", _norm(A["ND2"][0] - A["CG"][0]) if "ND2" in A and "CG" in A else None)
    if resname == "GLN":
        add("acceptor", "OE1", _norm(A["OE1"][0] - A["CD"][0]) if "OE1" in A and "CD" in A else None)
        add("donor", "NE2", _norm(A["NE2"][0] - A["CD"][0]) if "NE2" in A and "CD" in A else None)
    if resname == "HIS":
        add("donor", "ND1", _norm(A["ND1"][0] - A["CG"][0]) if "ND1" in A and "CG" in A else None)
        add("donor", "NE2", _norm(A["NE2"][0] - A["CE1"][0]) if "NE2" in A and "CE1" in A else None)
    if resname == "TRP":
        add("donor", "NE1", _norm(A["NE1"][0] - A["CD1"][0]) if "NE1" in A and "CD1" in A else None)
    for rn, an, src in (("SER", "OG", "CB"), ("THR", "OG1", "CB"), ("TYR", "OH", "CZ"), ("CYS", "SG", "CB")):
        if resname == rn and an in A:
            add("donor", an, _norm(A[an][0] - A[src][0]) if src in A else None)

    if resname in HYDRO_ATOMS:
        hs = [A[k][0] for k in HYDRO_ATOMS[resname] if k in A]
        if hs:
            pts.append(("hydro", np.mean(hs, axis=0), None, 1.2, resname + "-sc"))
    return pts


def chain_field_points(residues):
    out = []
    for i, (seq, name, atoms) in enumerate(residues):
        prev = residues[i - 1][2] if i > 0 else None
        for t, x, u, w, label in residue_field_points(name, atoms, prev):
            out.append(dict(t=t, x=x, u=u, w=w, r=i, seq=seq, rname=name, label=label))
    return out


def surface_filter(pts, all_xyz, all_res, cutoff=3.5, max_neigh=2):
    """剔除深埋场点；主链给体/受体（N/O/OXT）是肽主链氢键的主要伙伴，不参与埋藏过滤"""
    keep = []
    for k, p in enumerate(pts):
        if p["label"] in ("N", "O", "OXT"):
            keep.append(k)
            continue
        d = np.linalg.norm(all_xyz - p["x"], axis=1)
        if int(((d < cutoff) & (all_res != p["r"])).sum()) <= max_neigh:
            keep.append(k)
    return keep


# ---------------- 支柱1：链约束 Sinkhorn-EM（Morse 势代价） ----------------
def chain_neighbors(r, n_res, window=3):
    return [j for j in range(max(0, r - window), min(n_res, r + window + 1)) if j != r]


def reach_L(dr):
    return 3.8 * dr + 4.0


class Matcher:
    """场点级 Sinkhorn-EM 匹配器（Morse 势 + 位阻墙 + 链可达惩罚）"""

    def __init__(self, pep_pts, pk_pts, n_res, pk_atoms=None):
        self.pk_atoms = pk_atoms
        self.pt = [p["t"] for p in pep_pts]
        self.pkt = [p["t"] for p in pk_pts]
        self.X0 = np.array([p["x"] for p in pep_pts])
        self.Y = np.array([p["x"] for p in pk_pts])
        self.p_res = np.array([p["r"] for p in pep_pts])
        self.n_res = n_res
        n, m = len(pep_pts), len(pk_pts)
        # Morse 参数矩阵（静态）与兼容掩码
        self.compat = np.array([[(a, b) in COMPAT for b in self.pkt] for a in self.pt])
        self.dst = np.zeros((n, m))
        self.sig = np.ones((n, m))
        self.epw = np.zeros((n, m))
        for i, a in enumerate(self.pt):
            for j, b in enumerate(self.pkt):
                if (a, b) in Params.MORSE:
                    self.dst[i, j], self.sig[i, j], self.epw[i, j] = Params.MORSE[(a, b)]
        # 方向矩阵（静态部分）
        self.U1 = np.array([p["u"] if p["u"] is not None else np.zeros(3) for p in pep_pts])
        self.U2 = np.array([p["u"] if p["u"] is not None else np.zeros(3) for p in pk_pts])
        has1 = np.array([p["u"] is not None for p in pep_pts])
        has2 = np.array([p["u"] is not None for p in pk_pts])
        self.dir_pair = np.array([[((a, b) in ANGULAR) for b in self.pkt] for a in self.pt])
        self.dir_mask = has1[:, None] & has2[None, :] & self.dir_pair & self.compat
        # 链邻居
        self.nbrs = [chain_neighbors(r, n_res) for r in range(n_res)]
        self.res_idx = [np.where(self.p_res == r)[0] for r in range(n_res)]
        self.res_X0 = np.array([self.X0[ix].mean(0) if len(ix) else np.zeros(3) for ix in self.res_idx])

    def pair_energy(self, X, U1_cur):
        """逐场点对相互作用能（Morse + 位阻墙）"""
        d = np.sqrt(((X[:, None, :] - self.Y[None, :, :]) ** 2).sum(-1))
        dd = (d - self.dst) / self.sig
        morse = self.epw * ((1.0 - np.exp(-dd)) ** 2 - 1.0)  # 阱底 −εw，远处 →0
        ddv = (d - Params.vdw_d) / Params.vdw_sig
        vdw = Params.vdw_epw * ((1.0 - np.exp(-ddv)) ** 2 - 1.0)  # 通用堆积势
        wall = Params.WALL * np.maximum(0.0, Params.WALL_D0 - d) ** 2
        E = np.where(self.compat, morse, vdw + wall)
        # 局部性项：接触区（<8Å）为零，远处二次增长——无伙伴点贴最近邻，不飘向点云质心
        E = E + Params.w_local * np.maximum(0.0, d - Params.loc_d0) ** 2
        # 方向调制：反平行全阱深，垂直保留 dir_perp
        dot = U1_cur @ self.U2.T
        dirfac = np.where(self.dir_mask,
                          Params.dir_perp + (1 - Params.dir_perp) * np.clip(-dot, 0.0, 1.0), 1.0)
        return np.where(self.dir_mask, dirfac * E, E)

    def cost(self, X, U1_cur, targets=None):
        """Sinkhorn 代价 = 相互作用能 + 链可达惩罚（线性化）"""
        E = self.pair_energy(X, U1_cur)
        if targets is not None:
            dists = np.linalg.norm(self.Y[:, None, :] - targets[None, :, :], axis=2)
            for r in range(self.n_res):
                pen = np.zeros(len(self.Y))
                for j in self.nbrs[r]:
                    pen += np.maximum(0.0, dists[:, j] - reach_L(abs(r - j))) ** 2
                E[self.res_idx[r]] += Params.w_c * pen
        return E

    def sinkhorn(self, C, eps):
        K = np.exp(-C / eps)
        # 稀疏化：每行只保留 top-k 候选；且只允许有实质吸引增益（C < −ATTR_MIN）的候选，
        # 避免"E≈0 的远处候选"产生虚假 target 拉力
        k = min(Params.topk, C.shape[1])
        idx = np.argpartition(C, k - 1, axis=1)[:, :k]
        mask = np.zeros_like(K, dtype=bool)
        np.put_along_axis(mask, idx, True, axis=1)
        K = K * (mask & (C < -Params.ATTR_MIN))
        empty = K.sum(1) <= 0  # 无吸引候选的行：退化为最近候选（保持行质量）
        if empty.any():
            idx0 = np.argmin(C, axis=1)
            K[empty, idx0[empty]] = 1.0
        n, m = C.shape
        a = np.full(n, 1.0 / n)
        b = np.full(m, 1.0 / m)
        v = np.ones(m)
        for _ in range(Params.n_sink):
            u = (a / np.maximum(K @ v, 1e-300)) ** Params.tau_u
            v = (b / np.maximum(K.T @ u, 1e-300)) ** Params.tau_q
        return u[:, None] * K * v[None, :]

    def score(self, x_pts, x_atoms, U1_cur):
        """统一打分 = 场点相互作用能（逐肽点最优配对均值） + 原子级软球排斥"""
        E = self.pair_energy(x_pts, U1_cur)
        attract = float(E.min(1).mean())
        rep = 0.0
        if self.pk_atoms is not None:
            d = np.sqrt(((x_atoms[:, None, :] - self.pk_atoms[None, :, :]) ** 2).sum(-1))
            rep = Params.lam_rep * float((np.maximum(0.0, 2.7 - d) ** 2).sum())
        return attract + rep

    def _repulse_rigid(self, x_atoms, x_pts, x_res):
        """刚体排斥弛豫（保持肽为刚体）"""
        if self.pk_atoms is None:
            return x_atoms, x_pts, x_res
        for _ in range(Params.rep_steps):
            d = np.linalg.norm(x_atoms[:, None, :] - self.pk_atoms[None, :, :], axis=2)
            close = d < Params.rep_d0
            if not close.any():
                break
            diff = x_atoms[:, None, :] - self.pk_atoms[None, :, :]
            w = np.where(close, (Params.rep_d0 - d) / np.maximum(d, 1e-6), 0.0)
            disp = (w[:, :, None] * diff).sum(1)
            R, t = weighted_kabsch(x_atoms, x_atoms + Params.rep_eta * disp, np.ones(len(x_atoms)))
            x_atoms = x_atoms @ R.T + t
            x_pts = x_pts @ R.T + t
            x_res = x_res @ R.T + t
        return x_atoms, x_pts, x_res

    def point_targets(self, X, P):
        """平衡位置 targets：T(i,p) = y_p + d_pref(i,p)·unit(x_i − y_p)
        —— M 步把肽点拉向势阱平衡点而非伙伴原子本身（EM 的真不动点）"""
        d = np.sqrt(((X[:, None, :] - self.Y[None, :, :]) ** 2).sum(-1))
        d_pref = np.where(self.compat, self.dst, Params.vdw_d)
        u = (X[:, None, :] - self.Y[None, :, :]) / np.maximum(d, 1e-6)[:, :, None]
        T = self.Y[None, :, :] + d_pref[:, :, None] * u
        Pn = P / (P.sum(1, keepdims=True) + 1e-12)
        return (Pn[:, :, None] * T).sum(1)

    def dock(self, x_atoms, x_pts, x_res, verbose=False, anneal=True, track=None):
        """Sinkhorn-EM：ε 几何退火 + 阻尼刚体 M 步 + 刚体排斥弛豫
        track: 可选回调 fn(x_atoms)，逐迭代观测"""
        U1_cur = self.U1.copy()
        targets = None
        hist = []
        Pn = Pagg = None
        wres = None
        for it in range(Params.n_em):
            eps = Params.eps
            if anneal:
                frac = it / max(Params.n_em - 1, 1)
                eps = Params.eps_max * (Params.eps / Params.eps_max) ** frac
            C = self.cost(x_pts, U1_cur, targets if it > 0 else None)
            P = self.sinkhorn(C, eps)
            Pagg = np.zeros((self.n_res, len(self.Y)))
            for r, ix in enumerate(self.res_idx):
                if len(ix):
                    Pagg[r] = P[ix].sum(0)
            wres = Pagg.sum(1)
            Pn = Pagg / (Pagg.sum(1, keepdims=True) + 1e-12)
            # 平衡位置 targets（逐点→残基聚合）
            tgt_pts = self.point_targets(x_pts, P)
            targets = np.array([tgt_pts[ix].mean(0) if len(ix) else x_res[r]
                                for r, ix in enumerate(self.res_idx)])
            ok = wres > 1e-6
            targets[~ok] = x_res[~ok]
            R, t = weighted_kabsch(x_res, targets, np.maximum(wres, 1e-9))
            # 阻尼刚体更新 + 方向帧同步旋转
            Rd, td = (1 - Params.eta_att) * np.eye(3) + Params.eta_att * R, Params.eta_att * t
            U1_cur = U1_cur @ Rd.T
            x_res = x_res @ Rd.T + td
            x_pts = x_pts @ Rd.T + td
            x_atoms = x_atoms @ Rd.T + td
            x_atoms, x_pts, x_res = self._repulse_rigid(x_atoms, x_pts, x_res)
            if track is not None:
                track(it, x_atoms)
            hist.append(self.score(x_pts, x_atoms, U1_cur))
            if verbose and (it % 10 == 0 or it == Params.n_em - 1):
                print(f"    iter {it:2d}  score={hist[-1]:10.1f}")
        return x_atoms, x_pts, x_res, Pn, wres, hist


# ---------------- 支柱2：构象熵账单 ----------------
STATES = ["aR", "beta", "PPII", "gamma", "aL", "delta"]
KAPPA = 0.7

PRIOR = {
    "std": {"aR": 0.40, "beta": 0.25, "PPII": 0.15, "gamma": 0.05, "aL": 0.05, "delta": 0.10},
    "GLY": {"aR": 0.18, "beta": 0.10, "PPII": 0.07, "gamma": 0.10, "aL": 0.16, "delta": 0.39},
    "PRO": {"aR": 0.33, "beta": 0.02, "PPII": 0.55, "gamma": 0.03, "aL": 0.01, "delta": 0.06},
}

ROT_BILL = {
    "ALA": 0.0, "GLY": 0.0, "PRO": 0.4, "SER": 0.6, "CYS": 0.6, "THR": 0.7, "VAL": 0.8,
    "HIS": 0.9, "PHE": 1.0, "ASN": 1.1, "ASP": 1.1, "ILE": 1.2, "LEU": 1.2, "GLN": 1.3,
    "GLU": 1.3, "MET": 1.4, "TRP": 1.4, "TYR": 1.4, "LYS": 1.5, "ARG": 1.7}

CRATIC = 1.5


def classify_torsion(phi, psi):
    if phi is None or psi is None:
        return None
    if phi > 20 and -40 < psi < 100:
        return "aL"
    if -120 < phi < -35 and -90 < psi < 0:
        return "aR"
    if -100 < phi < -35 and 0 <= psi < 95:
        return "gamma"
    if -100 < phi < -55 and 95 <= psi < 165:
        return "PPII"
    if phi <= -100 and psi >= 95:
        return "beta"
    return "delta"


def torsion_class(aa):
    return "GLY" if aa == "GLY" else ("PRO" if aa == "PRO" else "std")


def trans_matrix(cls):
    p = np.array([PRIOR[cls][s] for s in STATES])
    T = p[None, :] * (1.0 + KAPPA * np.eye(len(STATES)))
    return T / T.sum(1, keepdims=True)


def entropy_bills(states, classes):
    n, S = len(states), len(STATES)
    sidx = {s: k for k, s in enumerate(STATES)}
    prior = {c: np.array([PRIOR[c][s] for s in STATES]) for c in ("std", "GLY", "PRO")}
    f = np.zeros((n, S))
    b = np.zeros((n, S))
    f[0] = prior[classes[0]]
    for i in range(1, n):
        T = trans_matrix(classes[i])
        f[i] = prior[classes[i]] * (T.T @ f[i - 1])
    Z = f[-1].sum()
    b[-1] = 1.0
    for i in range(n - 2, -1, -1):
        T = trans_matrix(classes[i + 1])
        b[i] = T @ (prior[classes[i + 1]] * b[i + 1])
    post = f * b / Z
    bills = np.zeros(n)
    obs = np.array([sidx[s] if s is not None else -1 for s in states])
    for i in range(n):
        if obs[i] >= 0:
            bills[i] = -RT * np.log(max(post[i, obs[i]], 1e-12))
        else:
            bills[i] = -RT * np.log(max(post[i].mean(), 1e-12))
    joint_logp = np.log(max(prior[classes[0]][obs[0]], 1e-12))
    for i in range(1, n):
        T = trans_matrix(classes[i])
        joint_logp += np.log(max(T[obs[i - 1], obs[i]], 1e-12))
    return bills, float(joint_logp), post
