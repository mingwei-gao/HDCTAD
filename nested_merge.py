import os
import argparse
import time
import numpy as np
import pandas as pd


# ╔══════════════════════════════════════════════════════════════╗
# ║                        路径配置                              ║
# ╚══════════════════════════════════════════════════════════════╝

# ★ 项目根目录（与 randomwalk.py / pipeline.py 一致）
PROJECT_BASE = "/mnt/sdb/gao/tad_project"

# Hi-C 接触矩阵: {hic_base}/{res}KB/chr{N}_{res}KB.txt
HIC_BASE = "/mnt/sde/gmw/GM12878/Hi-C/KR"


# ╔══════════════════════════════════════════════════════════════╗
# ║                     DeepTAD 合并算法                         ║
# ╚══════════════════════════════════════════════════════════════╝

def cosine_similarity_manual(x, y):
    """余弦相似度: dot(x,y) / (||x|| * ||y||)"""
    zero = [0] * len(x)
    if x == zero or y == zero:
        return 1.0 if x == y else 0.0
    arr = np.array([[x[i]*y[i], x[i]*x[i], y[i]*y[i]] for i in range(len(x))])
    return arr[:, 0].sum() / (np.sqrt(arr[:, 1].sum()) * np.sqrt(arr[:, 2].sum()) + 1e-10)


def calc_tad_pair_cs(a, b, df, hic_mat):
    """
    计算 TAD_a 与 TAD_b 的余弦相似度。

    对每个其他 TAD i:
        vec[k] = mean(inter_contact(TAD, k)) / mean(intra_contact(TAD))
    """
    r1 = [df.iloc[a, 0] - 1, df.iloc[a, 1] - 1]
    r2 = [df.iloc[b, 0] - 1, df.iloc[b, 1] - 1]
    vec1, vec2 = [], []
    for i in range(len(df)):
        if i == a or i == b:
            continue
        fr = [df.iloc[i, 0] - 1, df.iloc[i, 1] - 1]
        intra1 = np.mean(hic_mat[r1[0]:r1[1]+1, r1[0]:r1[1]+1])
        intra2 = np.mean(hic_mat[r2[0]:r2[1]+1, r2[0]:r2[1]+1])
        vec1.append(np.mean(hic_mat[r1[0]:r1[1]+1, fr[0]:fr[1]+1]) / (intra1 + 1e-10))
        vec2.append(np.mean(hic_mat[r2[0]:r2[1]+1, fr[0]:fr[1]+1]) / (intra2 + 1e-10))
    return cosine_similarity_manual(vec1, vec2)


def merge_tads(df, hic_mat, cs_threshold=0.8):
    """
    DeepTAD 嵌套 TAD 合并。

    1. 计算每对相邻 TAD 的余弦相似度
    2. 贪心扫描: CS >= threshold 且接触强度 >= 全局均值 → 合并
    3. 原始 TAD ∪ 合并 TAD → 去重 → 嵌套层次

    参数:
        df           : DataFrame [left, right]，1-indexed 闭区间
        hic_mat      : Hi-C 矩阵，0-indexed
        cs_threshold : 合并阈值

    返回:
        list[list[int]] : 嵌套 TAD [[left, right], ...]
    """
    n = len(df)
    avg_contact = np.mean(hic_mat)

    # 相邻 TAD 的余弦相似度
    cs_values = []
    for i in range(n - 1):
        cs_values.append(calc_tad_pair_cs(i, i + 1, df, hic_mat))
    cs_values.append(0)

    df = df.copy()
    df.insert(2, "cs", cs_values)

    # 贪心合并
    core_lst = []
    row = 0
    while row < n:
        if row == n - 1:
            core_lst.append([df.iloc[row, 0], df.iloc[row, 1]])
            row += 1
            continue

        left, right = df.iloc[row, 0], df.iloc[row, 1]
        cs = df.iloc[row, 2]
        new_region = [left, right]

        while cs >= cs_threshold:
            next_right = df.iloc[row + 1, 1]
            strength = np.mean(
                hic_mat[df.iloc[row+1, 0]-1:next_right,
                        df.iloc[row, 0]-1:df.iloc[row, 1]]
            )
            if strength >= avg_contact:
                cs = df.iloc[row + 1, 2]
                row += 1
                new_region = [left, next_right]
            else:
                break

        core_lst.append(new_region)
        row += 1

    # 原始 ∪ 合并 → 嵌套 TAD
    original = [[df.iloc[i, 0], df.iloc[i, 1]] for i in range(n)]
    seen = set()
    nested = []
    for tad in original + core_lst:
        key = (tad[0], tad[1])
        if key not in seen:
            seen.add(key)
            nested.append(tad)
    nested.sort(key=lambda x: (x[0], -x[1]))
    return nested


# ╔══════════════════════════════════════════════════════════════╗
# ║                        I/O 工具                              ║
# ╚══════════════════════════════════════════════════════════════╝

def load_hic(path):
    """加载 Hi-C 矩阵（纯数值，无表头）"""
    return pd.read_csv(path, sep='\\s+', header=None).values.astype(np.float32)


def load_tads(path):
    """加载 TAD 边界文件 → list[(start, end)]，跳过注释行"""
    tads = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or line.startswith("Start"):
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                s, e = int(parts[0]), int(parts[1])
                if s < e:
                    tads.append((s, e))
    return tads


def save_nested_tads(path, tads, threshold):
    """保存嵌套 TAD 结果"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(f"# deeptad nested TADs, cos_threshold={threshold}\n")
        f.write("Start\tEnd\n")
        for s, e in tads:
            f.write(f"{s}\t{e}\n")


# ╔══════════════════════════════════════════════════════════════╗
# ║                           主流程                             ║
# ╚══════════════════════════════════════════════════════════════╝

def run(chr_name, res, threshold):
    """单条染色体: 读 TAD → 合并 → 保存"""
    chr_num = chr_name.replace("chr", "")
    chr_dir = os.path.join(PROJECT_BASE, f"res{res}", f"chr{chr_num}")

    # 路径
    hic_path = os.path.join(HIC_BASE, f"{res}KB", f"{chr_name}_{res}KB.txt")
    minima_path = os.path.join(chr_dir, "result", "TADs.txt")
    output_path = os.path.join(chr_dir, "v3_refine_nested", "final_tads.txt")

    # 检查文件
    if not os.path.exists(hic_path):
        print(f"❌ Hi-C 文件不存在: {hic_path}")
        return
    if not os.path.exists(minima_path):
        print(f"❌ 精修 TAD 文件不存在: {minima_path}")
        return

    # 执行
    print(f"📌 {chr_name} | res={res}KB | threshold={threshold}")
    print(f"   Hi-C:  {hic_path}")
    print(f"   输入:  {minima_path}")
    print(f"   输出:  {output_path}")

    t0 = time.time()
    hic_mat = load_hic(hic_path)
    tads = load_tads(minima_path)
    print(f"   加载: {time.time() - t0:.1f}s | Hi-C {hic_mat.shape} | TAD {len(tads)} 个")

    if len(tads) < 2:
        print(f"   ⚠ TAD 数量 < 2，无法合并，跳过")
        return

    t0 = time.time()
    df = pd.DataFrame(tads, columns=["left", "right"])
    result = merge_tads(df, hic_mat, cs_threshold=threshold)
    save_nested_tads(output_path, result, threshold)
    print(f"   合并: {time.time() - t0:.1f}s | {len(tads)} → {len(result)} 个嵌套 TAD")
    print(f"✅ 完成")


def main():
    parser = argparse.ArgumentParser(description="DeepTAD 层级嵌套 TAD 合并（单染色体）")
    parser.add_argument("--chr",       type=int, required=True, help="染色体编号，如 22")
    parser.add_argument("--res",       type=int, default=25,    help="分辨率 KB (默认 25)")
    parser.add_argument("--threshold", type=float, default=0.8, help="余弦相似度阈值 (默认 0.8)")
    args = parser.parse_args()

    run(f"chr{args.chr}", args.res, args.threshold)


if __name__ == "__main__":
    main()