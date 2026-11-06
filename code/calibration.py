# -*- coding: utf-8 -*-
"""EPT-Dock v1 参数标定：从 PDB 校准集统计
  A) 肽-蛋白复合物场点接触距离 -> MORSE d*/σ 与通用 vdW d*
  B) 球蛋白 coil 库（HELIX/SHEET 记录排除二级结构）-> Ramachandran 状态先验与相邻转移矩阵
所有 PDB ID 仅为候选，经程序化验证（解析成功/链形态符合）后才进入统计。
输出：data/calib/calib.json
"""
import os
import sys
import json
import subprocess
import concurrent.futures as cf

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import entrobill_proto as eb  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CALDIR = os.path.abspath(os.path.join(HERE, "..", "data", "calib"))
os.makedirs(CALDIR, exist_ok=True)

# ---- 候选集（程序化验证，失败即跳过）----
PEPTIDE_CANDIDATES = [
    # CaM-肽
    "2BBM", "1CDL", "1CT8", "2VAY", "1IWQ", "1MXE", "2F3Y",
    # p53 系
    "1YCR", "1YCQ", "1DT7",
    # PDZ / SH2 / SH3 / WW / BRCT / BIR
    "1BE9", "1QAU", "3GSL", "2I0L", "1JYR", "1SBB", "1LKK", "2B3O", "1SHA",
    "1BBZ", "2H96", "1RLQ", "1CKA", "1H8S", "1F8A", "1JNX", "1G3F", "1G73",
    # BH3 / MDM2 / 其它
    "1BXL", "1R0D", "1PQ1", "2NLW", "3MQP", "2B4J", "1QJA", "3P21",
    "1AKJ", "1ABO", "1AWC", "2F4L", "3NKJ", "1OQD",
]
PROTEIN_CANDIDATES = [
    "1UBQ", "1LYZ", "1AKI", "1HEL", "2LZM", "7RSA", "1FS3", "1MBO", "1A3N",
    "1EMA", "1FKG", "1TIM", "4AKE", "1CLL", "2CI2", "1BPI", "1PGB", "3CHY",
    "1CSP", "1HRC", "256B", "1CCP", "1PNT", "5CNA", "1LMB", "1ENH", "1HDD",
    "1TEN", "1FNF", "2ERL", "1CTF", "1TIG",
]

STD_AA = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
          "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"}


def fetch(pid):
    path = os.path.join(CALDIR, pid + ".pdb")
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return pid, True
    r = subprocess.run(["curl", "-sS", "-m", "40", "-o", path,
                        f"https://files.rcsb.org/download/{pid}.pdb"],
                       capture_output=True)
    ok = r.returncode == 0 and os.path.exists(path) and os.path.getsize(path) > 1000
    if not ok and os.path.exists(path):
        os.remove(path)
    return pid, ok


def is_std_chain(residues, frac=0.9):
    if not residues:
        return False
    n_std = sum(1 for (_, nm, _) in residues if nm in STD_AA)
    return n_std / len(residues) >= frac


def secstruct_marks(path):
    """PDB HELIX/SHEET 记录 → {(chain, seqnum)}"""
    marks = set()
    with open(path) as fh:
        for line in fh:
            if line.startswith("HELIX ") or line.startswith("SHEET "):
                try:
                    ci, si, ce, se = line[19], int(line[21:25]), line[31], int(line[33:37])
                except ValueError:
                    continue
                if ci == ce:
                    for s in range(si, se + 1):
                        marks.add((ci, s))
    return marks


def receptor_peptide(chains):
    cands = {c: rs for c, rs in chains.items() if is_std_chain(rs)}
    if not cands:
        return None
    rec = max(cands, key=lambda c: len(cands[c]))
    if len(cands[rec]) < 60:
        return None
    peps = [c for c, rs in cands.items() if c != rec and 5 <= len(rs) <= 35]
    if not peps:
        return None
    return cands[rec], cands[peps[0]]


def contact_stats(path, acc):
    chains = eb.parse_pdb(path)
    rp = receptor_peptide(chains)
    if rp is None:
        return False
    rec, pep = rp
    pk_all = eb.chain_field_points(rec)
    pep_pts = eb.chain_field_points(pep)
    xyz, resl = [], []
    for i, (seq, nm, at) in enumerate(rec):
        for an, (p, e) in at.items():
            xyz.append(p)
            resl.append(i)
    xyz = np.array(xyz)
    resl = np.array(resl)
    keep = eb.surface_filter(pk_all, xyz, resl)
    pk = [pk_all[k] for k in keep]
    if len(pk) < 40 or len(pep_pts) < 10:
        return False
    X = np.array([p["x"] for p in pep_pts])
    Y = np.array([p["x"] for p in pk])
    D = np.sqrt(((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1))
    types_p = [p["t"] for p in pep_pts]
    types_q = [p["t"] for p in pk]
    U_p = [p["u"] for p in pep_pts]
    U_q = [p["u"] for p in pk]
    for i, a in enumerate(types_p):
        for j, b in enumerate(types_q):
            # 各向异性预筛：仅对含供体的方向性配对要求氢键几何（反平行 dot ≤ −0.3），
            # 否则真氢键子群被"靠近但未成键"的背景淹没；盐桥几何松散不做预筛
            if (a, b) in eb.ANGULAR and "donor" in (a, b):
                up, uq = U_p[i], U_q[j]
                if up is None or uq is None or float(np.dot(up, uq)) > -0.3:
                    continue
            d = D[i, j]
            if d < 8.0:
                acc.setdefault(a + "|" + b, []).append(float(d))
    return True


def coil_stats(path, acc_prior, acc_trans, used):
    chains = eb.parse_pdb(path)
    marks = secstruct_marks(path)
    n_ch = 0
    for ch, residues in chains.items():
        if not is_std_chain(residues, 0.95) or len(residues) < 40:
            continue
        covered = sum(1 for (seq, nm, at) in residues if (ch, seq) in marks)
        if covered < 0.05 * len(residues):
            continue  # 无有效二级结构记录，无法定义 coil
        phi, psi = eb.backbone_torsions(residues)
        states = [(ch, seq) not in marks and eb.classify_torsion(phi[i], psi[i])
                  for i, (seq, nm, at) in enumerate(residues)]
        n = 0
        for i, (seq, nm, at) in enumerate(residues):
            if states[i]:
                cls = "GLY" if nm == "GLY" else ("PRO" if nm == "PRO" else "std")
                acc_prior.setdefault(cls, []).append(states[i])
                n += 1
        for i in range(len(residues) - 1):
            if residues[i + 1][0] == residues[i][0] + 1 and states[i] and states[i + 1]:
                acc_trans.append((states[i], states[i + 1]))
        if n:
            n_ch += 1
    if n_ch:
        used.append(os.path.basename(path))


def hist_mode(vals, lo=2.0, hi=8.0):
    v = np.asarray(vals)
    v = v[(v >= lo) & (v <= hi)]
    if len(v) < 10:
        return None
    counts, edges = np.histogram(v, bins=np.arange(lo, hi + 0.01, 0.25))
    sm = np.convolve(counts, [0.25, 0.5, 0.25], mode="same")
    k = int(np.argmax(sm))
    return float(edges[k] + 0.125)


def summarize(vals):
    v = np.asarray(vals)
    q = np.percentile(v, [10, 25, 50, 75])
    counts, edges = np.histogram(v, bins=np.arange(2.0, 8.01, 0.25))
    hb = v[v < 4.5]  # 氢键窗口子群（供体对用）
    return {"n": int(len(v)), "p10": round(float(q[0]), 2), "p25": round(float(q[1]), 2),
            "p50": round(float(q[2]), 2), "p75": round(float(q[3]), 2),
            "mode": hist_mode(v), "mode_hb": hist_mode(hb, 2.0, 4.5),
            "n_hb": int(len(hb)), "hist": [int(x) for x in counts]}


def main():
    pids = sorted(set(PEPTIDE_CANDIDATES + PROTEIN_CANDIDATES))
    print(f"下载 {len(pids)} 个候选 PDB ...")
    with cf.ThreadPoolExecutor(8) as ex:
        res = dict(ex.map(fetch, pids))
    ok = [p for p, f in res.items() if f]
    print(f"成功下载 {len(ok)}/{len(pids)}")

    acc_contact = {}
    n_complex = 0
    fail_complex = []
    for pid in sorted(set(PEPTIDE_CANDIDATES) & set(ok)):
        path = os.path.join(CALDIR, pid + ".pdb")
        try:
            if contact_stats(path, acc_contact):
                n_complex += 1
            else:
                fail_complex.append(pid)
        except Exception:
            fail_complex.append(pid)
    print(f"肽-复合物统计：{n_complex} 个有效（跳过 {len(fail_complex)}: {' '.join(fail_complex[:12])}）")

    # 镜像对在原始数据层合并（相互作用对称；规范方向：donor/pos 在前）
    CANON = {("donor", "acceptor"), ("donor", "neg"), ("pos", "acceptor"), ("pos", "neg")}
    raw = {}
    for k, v in acc_contact.items():
        a, b = k.split("|")
        if (a, b) in CANON or (b, a) not in CANON:
            ck = k
        else:
            ck = b + "|" + a
        raw.setdefault(ck, []).extend(v)
    contacts = {k: summarize(v) for k, v in sorted(raw.items())}

    acc_prior, acc_trans, used = {}, [], []
    n_prot = 0
    for pid in sorted(set(PROTEIN_CANDIDATES) & set(ok)):
        path = os.path.join(CALDIR, pid + ".pdb")
        try:
            before = sum(len(v) for v in acc_prior.values())
            coil_stats(path, acc_prior, acc_trans, used)
            if sum(len(v) for v in acc_prior.values()) > before:
                n_prot += 1
        except Exception:
            pass
    print(f"coil 统计：{n_prot} 个蛋白，coil 残基 {sum(len(v) for v in acc_prior.values())}，"
          f"相邻对 {len(acc_trans)}")

    STATES = eb.STATES
    prior = {}
    for cls, lst in acc_prior.items():
        cnt = {s: 0 for s in STATES}
        for s in lst:
            cnt[s] += 1
        tot = sum(cnt.values())
        prior[cls] = {s: round(cnt[s] / tot, 4) for s in STATES} if tot else None
    sidx = {s: i for i, s in enumerate(STATES)}
    T = np.ones((6, 6))  # Laplace 先验
    for a, b in acc_trans:
        T[sidx[a], sidx[b]] += 1
    T = T / T.sum(1, keepdims=True)

    calib = {"meta": {"n_complex": n_complex, "n_protein": n_prot,
                      "n_coil_res": sum(len(v) for v in acc_prior.values()),
                      "n_pairs": len(acc_trans), "date": "2026-11-05"},
             "contacts": contacts,
             "entropy": {"prior": prior, "trans": [[round(x, 4) for x in row] for row in T],
                         "states": STATES}}
    out = os.path.join(CALDIR, "calib.json")
    json.dump(calib, open(out, "w"), ensure_ascii=False, indent=1)
    print(f"已写入 {out}")
    for k in ("donor|acceptor", "donor|neg", "pos|neg", "hydro|hydro", "noncompat"):
        if k in contacts:
            c = contacts[k]
            print(f"  {k:18s} n={c['n']:5d} p25={c['p25']:.2f} p50={c['p50']:.2f} mode={c['mode']}")
    print("coil 先验：", json.dumps(prior.get("std", {}), ensure_ascii=False))


if __name__ == "__main__":
    main()
