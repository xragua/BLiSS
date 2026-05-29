#!/usr/bin/env python3
"""
clean_egauss.py

Usage:
    python clean_egauss.py input.par output.par

What it does:
- Finds egauss(N) parameter triplets (area, center, sigma)
- Drops any triplet where area == 0 (numerically)
- Renumbers the remaining egauss indices in ascending order starting from 1
- Renumbers the leading "idx" column sequentially starting at 1
- Preserves column formatting as closely as practical
- Then: loads clean_lines.csv, adds "stat_diff" column from stat_diff.txt,
  padding the rest with NaN, and writes clean_lines_with_statdiff.csv
"""
import sys
import re
from pathlib import Path

import pandas as pd
import numpy as np

AREA_RE = re.compile(r'(egauss\()\s*(\d+)\s*(\)\.area)')
CENTER_RE = re.compile(r'(egauss\()\s*(\d+)\s*(\)\.center)')
SIGMA_RE = re.compile(r'(egauss\()\s*(\d+)\s*(\)\.sigma)')


def parse_float_safe(s: str):
    try:
        return float(s)
    except Exception:
        return None


def is_egauss_area(line: str):
    return bool(AREA_RE.search(line))


def get_area_value(line: str):
    """
    Extract the 'value' field from a parameter line like:
      "  45  egauss(1).area         0     0     9.613535e-06           0       1e+08  photons/s/cm^2"
    We need the 5th column (value). We'll split by whitespace and read the column positions:
    idx  name  tie  freeze  value  min  max  [unit]
    """
    parts = line.strip().split()
    if len(parts) < 7:
        return None
    # parts[0]=idx, parts[1]=name, parts[2]=tie-to, parts[3]=freeze, parts[4]=value
    return parse_float_safe(parts[4])


def renumber_idx_column(lines):
    """Rewrite the first numeric column (idx) to be 1..N, preserving right alignment."""
    out = []
    idx_counter = 1
    for line in lines:
        m = re.match(r'\s*\d+\s+(.*)$', line)
        if not m:
            out.append(line)
            continue
        rest = m.group(1)
        out.append(f"{idx_counter:4d}  {rest}")
        idx_counter += 1
    return out


def main(inp: Path, outp: Path):
    text = inp.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    # Keep header lines (typically first 2 lines). We'll process from line 3 onward.
    if len(lines) < 3:
        outp.write_text(text, encoding="utf-8")
        return

    header0 = lines[0]
    header1 = lines[1]
    body = lines[2:]

    kept_lines = []
    i = 0
    old_to_new = {}
    next_egauss_index = 1

    while i < len(body):
        line = body[i]
        if is_egauss_area(line):
            # Expect the following two lines to be center and sigma with the same index
            if i + 2 >= len(body):
                kept_lines.append(line)
                i += 1
                continue

            area_line = body[i]
            center_line = body[i + 1]
            sigma_line = body[i + 2]

            ma = AREA_RE.search(area_line)
            mc = CENTER_RE.search(center_line)
            ms = SIGMA_RE.search(sigma_line)

            same_index = False
            if ma and mc and ms:
                ia = int(ma.group(2))
                ic = int(mc.group(2))
                isg = int(ms.group(2))
                same_index = (ia == ic == isg)

            if same_index:
                val = get_area_value(area_line)

                # Drop triplet if area is exactly zero (the ones we froze in ISIS)
                if val is not None and val == 0.0:
                    i += 3
                    continue
                else:
                    old_idx = int(ma.group(2))
                    new_idx = next_egauss_index
                    next_egauss_index += 1
                    old_to_new[old_idx] = new_idx

                    def renumber_line(l, pat):
                        return pat.sub(
                            lambda m: f"{m.group(1)}{new_idx}{m.group(3)}", l
                        )

                    kept_lines.append(renumber_line(area_line, AREA_RE))
                    kept_lines.append(renumber_line(center_line, CENTER_RE))
                    kept_lines.append(renumber_line(sigma_line, SIGMA_RE))
                    i += 3
                    continue
            else:
                kept_lines.append(line)
                i += 1
                continue
        else:
            kept_lines.append(line)
            i += 1

    # Renumber idx column in the body
    body_renumbered = renumber_idx_column(kept_lines)

    # ---- Rebuild header0 using raw_model.par template ----
    area_pat = re.compile(r'egauss\((\d+)\)\.area')
    idxs = sorted(
        {
            int(m.group(1))
            for m in map(lambda mm: re.search(area_pat, mm), kept_lines)
            if m
        }
    )

    # Read first line of raw_model.par, e.g.
    # (tbnew(1)+constant(1)*tbnew(2))*(powerlaw(1)+bbody(1)+bbody(2)+linemodel)
    raw_model_path = Path("raw_model.par")
    if raw_model_path.exists():
        base_header0 = raw_model_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()[0].strip()
    else:
        # Fallback to whatever was in the original header0
        base_header0 = header0

    if idxs:
        # Build the line model as a sum of the surviving egauss components
        line_model = "+".join(f"egauss({i})" for i in idxs)
        if "linemodel" in base_header0:
            header0 = base_header0.replace("linemodel", line_model)
        else:
            # If for some reason there's no 'linemodel' token, just keep base_header0
            header0 = base_header0
    else:
        # No gaussians left: drop 'linemodel' term if it exists
        if "linemodel" in base_header0:
            # Try to remove '+linemodel' or 'linemodel+' cleanly, then any remaining 'linemodel'
            header0 = (
                base_header0.replace("+linemodel", "")
                .replace("linemodel+", "")
                .replace("linemodel", "")
            )
        else:
            header0 = base_header0

    final_lines = [header0, header1] + body_renumbered
    outp.write_text("\n".join(final_lines) + "\n", encoding="utf-8")

    # ---- pandas / stat_diff part ----
    try:
        df = pd.read_csv("clean_lines.csv")
        stat_diff = np.loadtxt("stat_diff.txt")
    except FileNotFoundError as e:
        print(f"Warning: {e}. Skipping CSV/stat_diff update.")
        return

    df["stat_diff"] = np.nan
    n = len(stat_diff)
    df.loc[: n - 1, "stat_diff"] = stat_diff
    df.to_csv("clean_lines.csv", index=False)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python clean_egauss.py input.par output.par")
        sys.exit(1)
    inp = Path(sys.argv[1])
    outp = Path(sys.argv[2])
    main(inp, outp)


# Files you want to delete
to_delete = [
    "raw_model.par",
    "spec_0.dat",
    "set_line_parameters_.sl",
    "set_line_model_.sl",   # if you also want this removed
    #"stat_diff.txt"         # only if you want to remove this too
]

for fname in to_delete:
    Path(fname).unlink(missing_ok=True)
