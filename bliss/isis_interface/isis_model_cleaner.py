"""Utilities for removing unused Gaussian components from ISIS parameter files."""
from __future__ import annotations
from pathlib import Path
import re
AREA_RE = re.compile('(egauss\\()\\s*(\\d+)\\s*(\\)\\.area)')
CENTER_RE = re.compile('(egauss\\()\\s*(\\d+)\\s*(\\)\\.center)')
SIGMA_RE = re.compile('(egauss\\()\\s*(\\d+)\\s*(\\)\\.sigma)')

def _parse_float(value: str):
    """Convert a token from an ISIS parameter row to a float.

    Parameters
    ----------
    value : str
        Text token read from a ``.par`` file, normally the parameter value column.

    Returns
    -------
    float or None
        Parsed floating-point value, or ``None`` when the token cannot be
        interpreted as a number.
    """
    try:
        return float(value)
    except Exception:
        return None

def _area_value(line: str):
    """Extract the numerical area value from an ``egauss`` parameter row.

    Parameters
    ----------
    line : str
        Single line from an ISIS ``.par`` file describing an ``egauss`` area
        parameter.

    Returns
    -------
    float or None
        Area value stored in the row, or ``None`` if the row is too short or the
        value field is not numeric.
    """
    parts = line.strip().split()
    if len(parts) < 7:
        return None
    return _parse_float(parts[4])

def _renumber_parameter_rows(lines: list[str]) -> list[str]:
    """Renumber ISIS parameter rows after components have been removed.

    Parameters
    ----------
    lines : list of str
        Body rows from an ISIS ``.par`` file, excluding the model-expression and
        column-header lines.

    Returns
    -------
    list of str
        Parameter rows with the leading integer index rewritten sequentially while
        preserving non-parameter lines unchanged.
    """
    out = []
    counter = 1
    for line in lines:
        match = re.match('\\s*\\d+\\s+(.*)$', line)
        if match is None:
            out.append(line)
        else:
            out.append(f'{counter:4d}  {match.group(1)}')
            counter += 1
    return out

def clean_zero_area_egauss_model(input_par: str | Path, output_par: str | Path) -> Path:
    """Remove zero-area ``egauss`` components from an ISIS parameter file.

    Parameters
    ----------
    input_par : str or pathlib.Path
        Existing ISIS ``.par`` file to inspect.
    output_par : str or pathlib.Path
        Destination path for the cleaned parameter file.

    Returns
    -------
    pathlib.Path
        Path to the written cleaned file.

    Notes
    -----
    The cleaner expects each Gaussian component to appear as consecutive
    ``area``, ``center``, and ``sigma`` rows. Components with exactly zero area are
    removed; the remaining components and parameter-row numbers are renumbered so
    that the output model expression remains consistent.
    """
    input_path = Path(input_par)
    output_path = Path(output_par)
    lines = input_path.read_text(encoding='utf-8', errors='replace').splitlines()
    if len(lines) < 3:
        output_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        return output_path
    header1 = lines[1]
    body = lines[2:]
    kept = []
    next_index = 1
    i = 0
    while i < len(body):
        line = body[i]
        if AREA_RE.search(line) and i + 2 < len(body):
            area_line, center_line, sigma_line = (body[i], body[i + 1], body[i + 2])
            ma, mc, ms = (AREA_RE.search(area_line), CENTER_RE.search(center_line), SIGMA_RE.search(sigma_line))
            same_component = ma and mc and ms and (int(ma.group(2)) == int(mc.group(2)) == int(ms.group(2)))
            if same_component:
                area = _area_value(area_line)
                if area is not None and abs(area) == 0.0:
                    i += 3
                    continue

                def renumber(line_text: str, pattern: re.Pattern) -> str:
                    """Replace the component index inside one ISIS Gaussian parameter row.

                    Parameters
                    ----------
                    line_text : str
                        Parameter row whose ``egauss(<index>)`` expression is being updated.
                    pattern : re.Pattern
                        Compiled expression matching one Gaussian parameter type, such as area,
                        center, or sigma.

                    Returns
                    -------
                    str
                        Row with the matched component index replaced by the next retained index.
                    """
                    return pattern.sub(lambda m: f'{m.group(1)}{next_index}{m.group(3)}', line_text)
                kept.append(renumber(area_line, AREA_RE))
                kept.append(renumber(center_line, CENTER_RE))
                kept.append(renumber(sigma_line, SIGMA_RE))
                next_index += 1
                i += 3
                continue
        kept.append(line)
        i += 1
    body_renumbered = _renumber_parameter_rows(kept)
    component_indices = sorted({int(m.group(1)) for line in kept for m in [re.search('egauss\\((\\d+)\\)\\.area', line)] if m})
    if component_indices:
        header0 = 'tbnew(1)*(powerlaw(1)+' + '+'.join((f'egauss({i})' for i in component_indices)) + ')'
    else:
        header0 = 'tbnew(1)*(powerlaw(1))'
    output_path.write_text('\n'.join([header0, header1] + body_renumbered) + '\n', encoding='utf-8')
    return output_path

def main() -> None:
    """Run the command-line cleaner for ISIS ``.par`` files."""
    import argparse
    parser = argparse.ArgumentParser(description='Remove zero-area egauss components from an ISIS .par model.')
    parser.add_argument('input_par')
    parser.add_argument('output_par')
    args = parser.parse_args()
    clean_zero_area_egauss_model(args.input_par, args.output_par)
if __name__ == '__main__':
    main()
