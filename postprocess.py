import os

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

REFINE_WINDOW = 5
INSULATION_HALF = 3
RANKSUM_SIZE = 5
PVALUE_THRESHOLD = 0.05
MIN_MERGE_BINS = 3
COS = 0.8


def extract_tads(preds):
    tads, start = [], 0
    for i in range(1, len(preds)):
        if preds[i] != preds[start]:
            tads.append((start, i - 1))
            start = i
    tads.append((start, len(preds) - 1))
    return tads


def to_shared(tads):
    t = [list(x) for x in tads]
    for i in range(len(t) - 1):
        t[i][1] = t[i + 1][0]
    return [(a, b) for a, b in t]


def to_adjacent(tads):
    t = [list(x) for x in tads]
    for i in range(len(t) - 1):
        t[i][1] = t[i + 1][0] - 1
    return [(a, b) for a, b in t]


def merge_pass1(tads, min_bins=MIN_MERGE_BINS):
    out, i = [], 0
    while i < len(tads):
        s, e = tads[i]
        if e - s < min_bins:
            ms, me, i = s, e, i + 1
            while i < len(tads) and tads[i][0] == me and tads[i][1] - tads[i][0] < min_bins:
                me = tads[i][1]
                i += 1
            out.append((ms, me))
        else:
            out.append((s, e))
            i += 1
    return out


def merge_pass2(tads, min_bins=MIN_MERGE_BINS):
    out, i = [], 0
    while i < len(tads):
        s, e = tads[i]
        if e - s >= min_bins:
            out.append([s, e])
        elif out and out[-1][1] == s:
            out[-1][1] = e
        elif i + 1 < len(tads) and tads[i + 1][0] == e:
            tads[i + 1] = (s, tads[i + 1][1])
        i += 1
    return [(a, b) for a, b in out]


def merge_short(tads, min_bins=MIN_MERGE_BINS):
    return merge_pass2(merge_pass1(list(tads), min_bins), min_bins)


def insulation_score(pos, hic, half_w=INSULATION_HALF):
    n = hic.shape[0]
    s, e = max(0, pos - half_w), min(n, pos + half_w)
    if s >= pos or pos >= e:
        return 1.0
    cross = hic[s:pos, pos:e]
    return float(np.mean(cross)) if cross.size > 0 else 1.0


def ranksum_pvalue(hic, pos, size=RANKSUM_SIZE):
    n = hic.shape[0]
    dia, ref = [], []
    for k in range(size):
        r = pos - k
        if 0 <= r < n - 1:
            for c in range(pos + 1, min(n, pos + size + 1)):
                dia.append(hic[r, c])
    for start, end in [(max(0, pos - size), pos + 1),
                       (pos + 1, min(n, pos + size + 1))]:
        if end - start >= 2:
            b = hic[start:end, start:end]
            for r in range(b.shape[0]):
                for c in range(r + 1, b.shape[1]):
                    ref.append(b[r, c])
    x, y = np.array(dia, np.float64), np.array(ref, np.float64)
    if len(x) < 2 or len(y) < 2:
        return 1.0
    try:
        _, p = mannwhitneyu(x, y, alternative="less", method="asymptotic")
        return p
    except Exception:
        return 1.0


def boundary_candidates(pos, hic, window=REFINE_WINDOW, half_w=INSULATION_HALF,
                        rs_size=RANKSUM_SIZE, p_thresh=PVALUE_THRESHOLD,
                        lo_limit=0, hi_limit=None):
    n = hic.shape[0]
    hi_cap = n - 1 if hi_limit is None else min(n - 1, hi_limit)
    lo = max(0, pos - window, lo_limit)
    hi = min(n - 1, pos + window, hi_cap)
    if lo > hi:
        return []
    scores = [(c, insulation_score(c, hic, half_w)) for c in range(lo, hi + 1)]
    vals = [v for _, v in scores]
    if max(vals) - min(vals) <= 1e-9:
        return [(min(max(pos, lo), hi), vals[0])]
    minima = []
    for i, (c, ins) in enumerate(scores):
        if (i == 0 or scores[i - 1][1] >= ins) and \
           (i == len(scores) - 1 or scores[i + 1][1] >= ins):
            minima.append((c, ins))
    passed = [(c, ins) for c, ins in minima if ranksum_pvalue(hic, c, rs_size) < p_thresh]
    passed.sort(key=lambda x: (abs(x[0] - pos), x[1]))
    return passed


def refine_sequential(hic, tads_closed):
    out, prev = [], -1
    n_tads = len(tads_closed)
    for i, (s, e) in enumerate(tads_closed):
        next_s = tads_closed[i + 1][0] if i + 1 < n_tads else hic.shape[0] - 1
        A = [c for c, _ in boundary_candidates(
            s, hic, lo_limit=max(prev, 0), hi_limit=s + REFINE_WINDOW)]
        B = [c for c, _ in boundary_candidates(
            e, hic, lo_limit=e - REFINE_WINDOW, hi_limit=next_s)]
        if not A and not B:
            continue
        pairs = [(a, b) for a in A for b in B if b - a >= 1]
        if pairs:
            ns, ne = min(pairs, key=lambda x: abs(x[0] - s) + abs(x[1] - e))
        else:
            alts = [(a, e) for a in A] + [(s, b) for b in B]
            alts = [(a, b) for a, b in alts if b - a >= 1]
            ns, ne = (min(alts, key=lambda x: abs(x[0] - s) + abs(x[1] - e))
                      if alts else (s, e))
        if ns < prev:
            ns = prev
        if ne < ns + 1:
            ns, ne = max(s, prev), e
        if ne < ns + 1:
            continue
        out.append((ns, ne))
        prev = ne
    return np.array(out, dtype=np.int64).reshape(-1, 2)


def _cos(x, y):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    if nx == 0 or ny == 0:
        return 1.0 if np.array_equal(x, y) else 0.0
    return float(x @ y / (nx * ny))


def _tad_cs(a, b, df, M):
    r1 = [int(df.iloc[a, 0]) - 1, int(df.iloc[a, 1]) - 1]
    r2 = [int(df.iloc[b, 0]) - 1, int(df.iloc[b, 1]) - 1]
    v1, v2 = [], []
    for i in range(len(df)):
        if i in (a, b):
            continue
        f = [int(df.iloc[i, 0]) - 1, int(df.iloc[i, 1]) - 1]
        intra1 = np.mean(M[r1[0]:r1[1] + 1, r1[0]:r1[1] + 1])
        intra2 = np.mean(M[r2[0]:r2[1] + 1, r2[0]:r2[1] + 1])
        c1 = np.mean(M[r1[0]:r1[1] + 1, f[0]:f[1] + 1])
        c2 = np.mean(M[r2[0]:r2[1] + 1, f[0]:f[1] + 1])
        v1.append(c1 / intra1 if intra1 > 0 else 0.0)
        v2.append(c2 / intra2 if intra2 > 0 else 0.0)
    return _cos(v1, v2)


def merge_tads(df, M, cs_threshold=COS):
    n = len(df)
    avg = float(np.mean(M))
    cs_list = [_tad_cs(i, i + 1, df, M) for i in range(n - 1)] + [0.0]
    df = df.copy()
    df.insert(2, "cosine_similarity", cs_list)
    row, core = 0, []
    while row < n:
        if row == n - 1:
            core.append([df.iloc[row, 0], df.iloc[row, 1]])
            row += 1
            continue
        left, right = df.iloc[row, 0], df.iloc[row, 1]
        cs = df.iloc[row, 2]
        new = [left, right]
        while cs >= cs_threshold and row + 1 < n:
            l2, r2 = df.iloc[row + 1, 0], df.iloc[row + 1, 1]
            strength = np.mean(M[l2 - 1:r2, left - 1:right])
            if not np.isfinite(strength) or strength < avg:
                break
            cs = df.iloc[row + 1, 2]
            row += 1
            new = [left, r2]
        core.append(new)
        row += 1
    original = [[df.iloc[i, 0], df.iloc[i, 1]] for i in range(n)]
    seen, out = set(), []
    for t in original + core:
        key = (int(t[0]), int(t[1]))
        if key not in seen:
            seen.add(key)
            out.append(list(key))
    out.sort(key=lambda x: (x[0], -x[1]))
    return np.array(out, dtype=np.int64)


def build_nested(leaf, hic, cos=COS):
    leaf = np.asarray(leaf, dtype=np.int64).reshape(-1, 2)
    if len(leaf) == 0:
        return leaf, np.array([], dtype=object)
    df = pd.DataFrame({"left": leaf[:, 0] + 1, "right": leaf[:, 1] + 1})
    nested = merge_tads(df, hic, cs_threshold=cos) - 1
    leaves = set(map(tuple, leaf.tolist()))
    types = np.array(["leaf" if tuple(r) in leaves else "parent"
                      for r in nested.tolist()], dtype=object)
    order = np.lexsort((-nested[:, 1], nested[:, 0]))
    return nested[order], types[order]


def save_nested(path, nested, types):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("# nested TAD: 0-based closed [Start, End]; Type = leaf|parent\n")
        f.write("Start\tEnd\tType\n")
        for (a, b), t in zip(nested.tolist(), types.tolist()):
            f.write(f"{a}\t{b}\t{t}\n")
