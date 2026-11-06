# -*- coding: utf-8 -*-
"""EPT-Bench50 构建（节点端）：从 PepBDB 官方全库选 50 例标准基准
准则（透明可复现，seed=42）：
  1) peptidelist 预筛：resolution 0 < r <= 2.5 Å（X-ray/EM，排除 NMR -1.00）
  2) 解析验证：受体 50-350 残基、肽 5-30 残基、肽/receptor 均为标准氨基酸
  3) 合并为单一复合物 PDB（受体=chain A，肽=chain B）
  4) 随机抽样 50（seed 42），清单与全部被拒原因写盘
"""
import os
import sys
import json
import glob
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import entrobill_proto as eb  # noqa: E402

ROOT = "/data/home/pepbdb"
OUTDIR = "/data/home/ept-dock/bench50"
STD = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
       "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"}


def std_frac(residues):
    if not residues:
        return 0.0
    return sum(1 for (_, nm, _) in residues if nm in STD) / len(residues)


def merge_complex(entry_dir, out_path):
    """receptor.pdb→chain A，peptide.pdb→chain B，重写 ATOM 行"""
    rec_lines, pep_lines = [], []
    for fn, ch in (("receptor.pdb", "A"), ("peptide.pdb", "B")):
        p = os.path.join(entry_dir, fn)
        if not os.path.exists(p):
            return False
        lines = rec_lines if ch == "A" else pep_lines
        with open(p) as fh:
            for ln in fh:
                if ln.startswith("ATOM"):
                    ln = ln[:21] + ch + ln[22:]
                    lines.append(ln)
    if not rec_lines or not pep_lines:
        return False
    with open(out_path, "w") as fh:
        fh.writelines(rec_lines)
        fh.writelines(pep_lines)
        fh.write("END\n")
    return True


def main():
    # 1) 预筛
    cands = []
    with open(os.path.join(ROOT, "peptidelist.txt")) as fh:
        for ln in fh:
            parts = ln.split()
            if len(parts) < 10:
                continue
            try:
                res = float(parts[-2])
            except ValueError:
                continue
            if not (0.0 < res <= 2.5) or parts[-1] != "prot":
                continue
            try:
                peplen = int(parts[2])
            except ValueError:
                continue
            if not (5 <= peplen <= 30):
                continue
            cands.append(f"{parts[0]}_{parts[1]}")  # 目录名格式 {pdb}_{pepchain}
    print(f"预筛（分辨率≤2.5Å，肽 5-30 残基）：{len(cands)} 条", flush=True)

    # 2) 解析验证 + 合并
    os.makedirs(OUTDIR, exist_ok=True)
    valid = []
    rejects = {"parse": 0, "size": 0, "std": 0, "merge": 0, "torsion": 0}
    for i, pid in enumerate(cands):
        entry = os.path.join(ROOT, "pepbdb", pid)  # 清单 col0 即为 {pdb}_{pepchain} 格式
        if not os.path.isdir(entry):
            rejects["parse"] += 1
            continue
        try:
            rec = eb.parse_pdb(os.path.join(entry, "receptor.pdb"))
            pep = eb.parse_pdb(os.path.join(entry, "peptide.pdb"))
        except Exception:
            rejects["parse"] += 1
            continue
        rec_chains = {c: r for c, r in rec.items() if std_frac(r) >= 0.9}
        pep_chains = {c: r for c, r in pep.items() if std_frac(r) >= 0.9}
        if not rec_chains or not pep_chains:
            rejects["std"] += 1
            continue
        prot = max(rec_chains.values(), key=len)
        pep = max(pep_chains.values(), key=len)
        if not (50 <= len(prot) <= 350) or not (5 <= len(pep) <= 30):
            rejects["size"] += 1
            continue
        # 扭角可评估性（至少 60% 残基有 φ/ψ）
        phi, psi = eb.backbone_torsions(pep)
        n_obs = sum(1 for a, b in zip(phi, psi) if a is not None and b is not None)
        if n_obs < 0.6 * len(pep):
            rejects["torsion"] += 1
            continue
        out = os.path.join(OUTDIR, pid + ".pdb")
        if not merge_complex(entry, out):
            rejects["merge"] += 1
            continue
        valid.append({"pid": pid, "n_rec": len(prot), "n_pep": len(pep)})
        if (i + 1) % 1000 == 0:
            print(f"  扫描 {i+1}/{len(cands)}，有效 {len(valid)}", flush=True)
    print(f"验证通过：{len(valid)} 个（拒绝：{rejects}）", flush=True)

    # 3) 随机抽样 50
    random.seed(42)
    sample = random.sample(valid, min(50, len(valid)))
    json.dump({"criteria": "PepBDB 2020-03-18; X-ray/EM res<=2.5A; receptor 50-350 res; "
                           "peptide 5-30 res; >=90% std AA; >=60% torsions observable; seed=42",
               "n_candidates": len(cands), "n_valid": len(valid),
               "rejects": rejects, "sample": sample},
              open(os.path.join(OUTDIR, "bench50_meta.json"), "w"),
              ensure_ascii=False, indent=1)
    # 删除未抽中的合并文件
    keep = {s["pid"] for s in sample}
    for f in glob.glob(os.path.join(OUTDIR, "*.pdb")):
        pid = os.path.basename(f)[:-4]
        if pid not in keep:
            os.remove(f)
    print(f"抽样 {len(sample)} 例 → {OUTDIR}", flush=True)
    print("样例:", " ".join(s["pid"] for s in sample[:10]), flush=True)


if __name__ == "__main__":
    main()
